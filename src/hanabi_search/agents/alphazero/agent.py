"""AlphaZero-style agent: information-set MCTS guided by a policy/value network."""

from math import log, sqrt
from typing import override

import torch

from hanabi_search.agents.alphazero.node import AlphaZeroNode
from hanabi_search.agents.mcts.agent import MCTSAgent
from hanabi_search.agents.mcts.node import MCTSNode
from hanabi_search.config import AlphaZeroAgentConfig, GameConfig, MCTSAgentConfig
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
        use_rules=config.use_rules,
        determinization=config.determinization,
        num_workers=1,
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
        network: Policy/value network; shared with the trainer, if any.
        device: Device the network lives on.
        transition_sink: Receives this agent's labelled decisions after each game.
    """

    node_type = AlphaZeroNode

    def __init__(
        self,
        player_id: int,
        game: GameConfig,
        config: AlphaZeroAgentConfig,
        network: AlphaZeroNetwork,
        device: torch.device,
        transition_sink: TransitionSink | None = None,
    ) -> None:
        super().__init__(player_id, game, as_search_config(config), transition_sink)
        self.alphazero_config = config
        self.network = network
        self.device = device
        self.num_actions = self.environment.num_moves()

    @override
    def expand(self, node: MCTSNode, observation: ObservationDict) -> None:
        """Evaluates `node` with the network and adds children with their priors."""
        assert isinstance(node, AlphaZeroNode)
        if node in self.children:
            return
        encoded = self.environment.vectorized_observation(observation["pyhanabi"])
        # (1, observation_size)
        inputs = torch.tensor(encoded, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            logits, value = self.network(inputs)
            policy = torch.softmax(logits, dim=1).squeeze(0)  # (num_actions,)
        node.value = (value.item() + 1) / 2 * self.game.max_score

        candidates = node.find_children(observation)
        if not candidates:
            return
        converted = [self._as_move(candidate) for candidate in candidates]
        # Rule proposals are deduplicated through a set (as in the original implementation,
        # whose iteration order determines tie-breaking); legal moves keep engine order.
        moves = converted if isinstance(candidates[0], HanabiMove) else list(set(converted))
        uids = [self.environment.game.get_move_uid(move) for move in moves]

        mask = torch.zeros(self.num_actions, dtype=torch.float32, device=self.device)
        mask[uids] = 1.0
        policy *= mask
        total = policy.sum()
        if total > 0:
            policy /= total
        elif mask.sum() > 0:
            policy = mask / mask.sum()
        else:
            return

        children: set[MCTSNode] = set()
        for move, uid in zip(moves, uids, strict=True):
            prior = policy[uid].item()
            if prior > 0:
                child = AlphaZeroNode((*node.moves, move), self.rules)
                child.prior = prior
                children.add(child)
        self.children[node] = children

    @override
    def uct_select(self, node: MCTSNode) -> MCTSNode:
        """Returns the child maximizing PUCT: mean value plus a prior-weighted bonus."""
        log_visits = log(self.visits[node] + 1)

        def puct(child: MCTSNode) -> float:
            assert isinstance(child, AlphaZeroNode)
            visits = self.visits[child]
            mean = self.total_reward[child] / visits if visits > 0 else 0
            bonus = self.exploration_weight * child.prior * sqrt(log_visits) / (1 + visits)
            return mean + bonus

        return max(self.children[node], key=puct)

    @override
    def simulate(self, node: MCTSNode) -> float:
        """Returns the network's value of `node` (no rollout)."""
        assert isinstance(node, AlphaZeroNode)
        if node.value is None:
            # Not expanded (depth limit reached): fall back to the current reward.
            return float(self.environment.reward())
        return node.value

    @override
    def backpropagate(self, path: list[MCTSNode], reward: float) -> None:
        for node in reversed(path):
            self.visits[node] += 1
            self.total_reward[node] += reward

    @override
    def choose_move(self, node: MCTSNode) -> HanabiMove:
        """Returns the most visited root move."""
        if node.is_terminal():
            raise RuntimeError(f"choose_move called on terminal node {node}")
        children = self.children.get(node)
        if not children:
            return super().choose_move(node)

        def visit_count(child: MCTSNode) -> float:
            visits = self.visits[child]
            return float("-inf") if visits == 0 else visits

        return max(children, key=visit_count).initial_move()
