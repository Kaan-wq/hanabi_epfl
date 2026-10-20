"""AlphaZero-style agent: information-set MCTS guided by a policy/value network."""

import random
from collections.abc import Generator
from math import sqrt
from typing import override

import numpy as np
import numpy.typing as npt

from hanabi_search.agents.alphazero.node import AlphaZeroNode
from hanabi_search.agents.base import Evaluation, EvaluationRequest
from hanabi_search.agents.mcts.agent import MCTSAgent
from hanabi_search.agents.mcts.node import MCTSNode
from hanabi_search.config import (
    AlphaZeroAgentConfig,
    GameConfig,
    MCTSAgentConfig,
    SelfPlayConfig,
)
from hanabi_search.env import HanabiMove, ObservationDict
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

    node_type = AlphaZeroNode

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
        self._moves_this_game = 0

    @override
    def end_episode(self, final_score: int) -> None:
        super().end_episode(final_score)
        self._moves_this_game = 0

    @override
    def select(self, node: MCTSNode) -> list[MCTSNode]:
        """Descends by PUCT until reaching an unexpanded node.

        Unlike plain MCTS, unvisited children are not tried first: the prior decides.
        """
        path = [node]
        while self.children.get(node):
            node = self.uct_select(node)
            path.append(node)
        return path

    @override
    def search_steps(
        self, node: MCTSNode, observation: ObservationDict
    ) -> Generator[EvaluationRequest, Evaluation]:
        """One search iteration: yields the leaf for the network, then expands it.

        Finished games are valued by their score and impossible paths by the reward
        reached; leaves at the depth limit get the network's value but no children.
        """
        descent = self.descend(node, observation)
        leaf = descent.path[-1]
        environment = self.environment
        if descent.impossible or leaf.is_terminal():
            reward = float(environment.reward())
        else:
            leaf_observation = descent.observation or environment.observe(
                # Chance (-1) wraps to the last player, as in `descend`.
                leaf.state.cur_player() % environment.num_players
            )
            encoded = environment.vectorized_observation(leaf_observation["pyhanabi"])
            policy, value = yield np.asarray(encoded, dtype=np.float32)
            reward = (value + 1) / 2 * self.game.max_score
            if descent.observation is not None:
                self.add_children(leaf, policy)
        self.backpropagate(descent.path, reward)

    @override
    def evaluate(self, request: EvaluationRequest) -> Evaluation:
        """Evaluates one position with this agent's network (batch of one)."""
        policies, values = self.network.predict(request[np.newaxis])
        return policies[0], float(values[0])

    def add_children(self, node: MCTSNode, policy: npt.NDArray[np.float32]) -> None:
        """Adds every legal move of `node` as a child, with its prior from `policy`.

        Args:
            node: The leaf being expanded (its focused state is not terminal).
            policy: The network's move probabilities at `node`, shape (num_actions,).
        """
        if node in self.children:
            return
        moves = node.state.legal_moves()
        if not moves:
            return
        uids = [self.environment.game.get_move_uid(move) for move in moves]

        priors = policy[uids]  # (num_candidates,)
        total = priors.sum()
        priors = priors / total if total > 0 else np.full(len(uids), 1 / len(uids))
        if self.self_play is not None and node == self.root_node:
            priors = self._add_root_noise(priors, self.self_play, self.rng)

        children: set[MCTSNode] = set()
        for move, prior in zip(moves, priors.tolist(), strict=True):
            if prior > 0:
                child = AlphaZeroNode((*node.moves, move), self.rules)
                child.prior = prior
                children.add(child)
        self.children[node] = children

    @override
    def uct_select(self, node: MCTSNode) -> MCTSNode:
        """Returns the child maximizing PUCT: `Q + c * P * sqrt(N) / (1 + n)`.

        `Q` is the child's mean reward as a fraction of the maximum score, so it is on
        the same [0, 1] scale as the exploration term. Unvisited children take the
        parent's mean reward as their `Q`.
        """
        max_score = self.game.max_score
        parent_visits = self.visits[node]
        parent_mean = self.total_reward[node] / parent_visits if parent_visits > 0 else 0.0
        scale = self.exploration_weight * sqrt(parent_visits)

        def puct(child: MCTSNode) -> float:
            assert isinstance(child, AlphaZeroNode)
            visits = self.visits[child]
            mean = self.total_reward[child] / visits if visits > 0 else parent_mean
            return mean / max_score + scale * child.prior / (1 + visits)

        return max(self.children[node], key=puct)

    @override
    def backpropagate(self, path: list[MCTSNode], reward: float) -> None:
        for node in reversed(path):
            self.visits[node] += 1
            self.total_reward[node] += reward

    @override
    def choose_move(self, node: MCTSNode) -> HanabiMove:
        """Returns the most visited root move.

        In self-play, this agent's first `temperature_moves` moves of each game are
        instead drawn in proportion to root visits.
        """
        if node.is_terminal():
            raise RuntimeError(f"choose_move called on terminal node {node}")
        children = self.children.get(node)
        self._moves_this_game += 1
        if not children:
            return super().choose_move(node)
        if self.self_play is not None and self._moves_this_game <= self.self_play.temperature_moves:
            # Sorted by move uid: set order must not affect which move is drawn.
            game = self.environment.game
            ordered = sorted(children, key=lambda child: game.get_move_uid(child.initial_move()))
            weights = [self.visits[child] for child in ordered]
            if sum(weights) > 0:
                return self.rng.choices(ordered, weights)[0].initial_move()

        def visit_count(child: MCTSNode) -> float:
            visits = self.visits[child]
            return float("-inf") if visits == 0 else visits

        return max(children, key=visit_count).initial_move()

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
