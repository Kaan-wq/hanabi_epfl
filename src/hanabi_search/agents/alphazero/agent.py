"""AlphaZero-style agent: information-set MCTS guided by a policy/value network."""

import random
from collections.abc import Generator
from math import inf
from typing import override

import numpy as np
import numpy.typing as npt

from hanabi_search.agents.alphazero.tree import SearchTree, TreeNode
from hanabi_search.agents.base import Evaluation, EvaluationRequest
from hanabi_search.agents.mcts.agent import MCTSAgent
from hanabi_search.agents.mcts.node import MCTSNode
from hanabi_search.config import (
    AlphaZeroAgentConfig,
    GameConfig,
    MCTSAgentConfig,
    SelfPlayConfig,
)
from hanabi_search.env import HanabiMove, HanabiState, ObservationDict
from hanabi_search.networks import AlphaZeroNetwork
from hanabi_search.training.transition import TransitionSink


def as_search_config(config: AlphaZeroAgentConfig) -> MCTSAgentConfig:
    """Returns the MCTS settings an AlphaZero agent searches with (no rollout simulation)."""
    return MCTSAgentConfig(
        rollouts=config.rollouts,
        max_simulation_steps=0,
        max_depth=config.max_depth,
        exploration_weight=config.exploration_weight,
        use_rules=False,
        determinization=config.determinization,
        collect_data=True,
    )


class AlphaZeroAgent(MCTSAgent):
    """MCTS where leaves are valued by the network and children weighted by its policy.

    Every decision is recorded as a training example (root visit counts as the
    policy target, labelled with the final score after the game).

    Args:
        player_id: Absolute seat of this agent.
        game: The game being played.
        config: Search hyperparameters.
        network: Policy/value network; shared with the trainer, if any. `act` evaluates
            leaves with it one at a time; batching callers use `act_steps` instead.
        transition_sink: Receives this agent's labelled decisions after each game.
        self_play: Exploration for training games: Dirichlet noise on the root priors
            and the first moves drawn in proportion to root visits. `None` searches
            and plays greedily.
    """

    def __init__(
        self,
        player_id: int,
        game: GameConfig,
        config: AlphaZeroAgentConfig,
        network: AlphaZeroNetwork,
        transition_sink: TransitionSink | None = None,
        self_play: SelfPlayConfig | None = None,
    ) -> None:
        super().__init__(player_id, game, as_search_config(config), transition_sink)
        self.alphazero_config = config
        self.network = network
        self.num_actions = self.environment.num_moves()
        self.self_play = self_play
        self.tree = SearchTree(config.exploration_weight, game.max_score)
        self._moves_this_game = 0

    @override
    def end_episode(self, final_score: int) -> None:
        super().end_episode(final_score)
        self._moves_this_game = 0

    @override
    def reset(self, state: HanabiState) -> None:
        super().reset(state)
        self.tree = SearchTree(self.alphazero_config.exploration_weight, self.game.max_score)

    @override
    def search_steps(
        self, node: MCTSNode, observation: ObservationDict
    ) -> Generator[EvaluationRequest, Evaluation]:
        """One search iteration: yields the leaf for the network, then expands it.

        The path is selected by PUCT first, then replayed in this determinization.
        Finished games are valued by their score and impossible paths by the reward
        reached; leaves at the depth limit get the network's value but no children.
        The path is replayed in one engine call (`DeterminizedEnv.play_moves`).

        Args:
            node: The root (AlphaZero always searches from it).
            observation: The searching player's observation at the root.
        """
        tree = self.tree
        path, move_uids = tree.select()
        environment = self.environment
        # The tree only grows below max_depth, so paths fit the depth limit.
        depth = environment.play_moves(move_uids)
        state = environment.state
        if depth < len(move_uids) or state.is_terminal():
            reward = float(environment.reward())
        else:
            if depth == 0:
                leaf_observation = observation["pyhanabi"]
            else:
                # Chance (-1) wraps to the last player, as in the original implementation.
                leaf_observation = state.observation(state.cur_player() % environment.num_players)
            encoded = environment.vectorized_observation(leaf_observation)
            policy, value = yield np.asarray(encoded, dtype=np.float32)
            reward = (value + 1) / 2 * self.game.max_score
            if depth < self.max_depth and not tree.is_expanded(path):
                self._expand(path, state, policy)
        tree.backpropagate(path, reward)

    @override
    def evaluate(self, request: EvaluationRequest) -> Evaluation:
        """Evaluates one position with this agent's network (batch of one)."""
        policies, values = self.network.predict(request[np.newaxis])
        return policies[0], float(values[0])

    @override
    def choose_move(self, node: MCTSNode) -> HanabiMove:
        """Returns the most visited root move (the first one on ties).

        In self-play, this agent's first `temperature_moves` moves of each game are
        instead drawn in proportion to root visits.
        """
        if node.is_terminal():
            raise RuntimeError(f"choose_move called on terminal node {node}")
        root = self.tree.root
        self._moves_this_game += 1
        if root is None or not root.move_uids:
            return super().choose_move(node)
        visits = root.visits
        self_play = self.self_play
        if (
            self_play is not None
            and self._moves_this_game <= self_play.temperature_moves
            and sum(visits) > 0
        ):
            best = self.rng.choices(range(len(visits)), visits)[0]
        else:
            # Unvisited children lose; the first of equally visited ones wins.
            best = max(range(len(visits)), key=lambda index: visits[index] or -inf)
        return self.environment.game.all_moves()[root.move_uids[best]]

    @override
    def root_visits(self) -> dict[int, float]:
        root = self.tree.root
        if root is None:
            return {}
        return dict(zip(root.move_uids, root.visits, strict=True))

    def _expand(
        self,
        path: list[tuple[TreeNode, int]],
        state: HanabiState,
        policy: npt.NDArray[np.float32],
    ) -> None:
        """Expands the leaf at the end of `path` with every legal move of `state`.

        Args:
            path: Path to the leaf, as returned by `SearchTree.select`.
            state: The leaf's state in this determinization (not terminal).
            policy: The network's move probabilities at the leaf, shape (num_actions,).
        """
        uids = state.legal_move_uids()
        if not uids.size:
            return
        priors = policy[uids]  # (num_moves,)
        total = priors.sum()
        priors = priors / total if total > 0 else np.full(len(uids), 1 / len(uids))
        if self.self_play is not None and not path:
            priors = self._add_root_noise(priors, self.self_play, self.rng)
        positive = priors > 0
        if not positive.all():
            uids, priors = uids[positive], priors[positive]
        self.tree.expand(path, uids.tolist(), priors.tolist())

    @staticmethod
    def _add_root_noise(
        priors: npt.NDArray[np.float32], self_play: SelfPlayConfig, rng: random.Random
    ) -> npt.NDArray[np.float32]:
        """Mixes Dirichlet noise into the root priors, shape (num_candidates,)."""
        gammas = np.array(
            [rng.gammavariate(self_play.dirichlet_alpha, 1.0) for _ in range(len(priors))],
            dtype=np.float32,
        )
        noise = gammas / gammas.sum()
        fraction = self_play.dirichlet_fraction
        mixed: npt.NDArray[np.float32] = (1 - fraction) * priors + fraction * noise
        return mixed
