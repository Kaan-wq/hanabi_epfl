"""Re-determinizing information-set MCTS agent."""

import random
from collections import defaultdict
from collections.abc import Sequence
from math import log, sqrt
from typing import override

import numpy as np

from hanabi_search.agents.base import Agent
from hanabi_search.agents.mcts.node import MCTSNode
from hanabi_search.agents.rule_based import VanDenBerghAgent, ruleset
from hanabi_search.agents.rule_based.ruleset import Rule
from hanabi_search.config import GameConfig, MCTSAgentConfig
from hanabi_search.env import DeterminizedEnv, HanabiMove, HanabiState, MoveDict, ObservationDict
from hanabi_search.exceptions import HanabiSearchError
from hanabi_search.training.transition import (
    PendingDecision,
    TransitionSink,
    value_target,
    visit_policy,
)


def default_expansion_rules() -> list[Rule]:
    """Rules whose proposals MCTS expands when `use_rules` is set."""
    return [
        ruleset.tell_most_information_factory(True),
        ruleset.tell_anyone_useful_card,
        ruleset.tell_dispensable_factory(),
        ruleset.complete_tell_useful,
        ruleset.complete_tell_dispensable,
        ruleset.complete_tell_unplayable,
        ruleset.play_probably_safe_factory(0.7, False),
        ruleset.play_probably_safe_late_factory(0.4, 5),
        ruleset.discard_most_confident,
    ]


class MCTSAgent(Agent):
    """MCTS over information sets, re-determinizing hidden hands on every rollout.

    Each rollout resamples this agent's own hand from what it knows, replays the
    tree path, expands one node, and runs `max_simulation_steps` moves of
    Van den Bergh rule-based play to estimate the outcome.

    Args:
        player_id: Absolute seat of this agent.
        game: The game being played.
        config: Search hyperparameters.
        transition_sink: Receives this agent's labelled decisions after each game
            when `config.collect_data` is set.
    """

    node_type: type[MCTSNode] = MCTSNode

    def __init__(
        self,
        player_id: int,
        game: GameConfig,
        config: MCTSAgentConfig,
        transition_sink: TransitionSink | None = None,
    ) -> None:
        super().__init__(player_id, game)
        self.config = config
        self.rollouts = config.rollouts
        self.max_depth = config.max_depth
        self.exploration_weight = config.exploration_weight
        self.rules: Sequence[Rule] | None = default_expansion_rules() if config.use_rules else None
        self.environment = DeterminizedEnv(
            game.to_params(), player_id, config.determinization, config.rollout_reward
        )
        self.rollout_agents = [VanDenBerghAgent(seat, game) for seat in range(game.players)]
        self.children: dict[MCTSNode, set[MCTSNode]] = {}
        self.visits: defaultdict[MCTSNode, int] = defaultdict(int)
        self.total_reward: defaultdict[MCTSNode, float] = defaultdict(float)
        self.root_node = self.node_type((), self.rules)
        self.root_state: HanabiState | None = None
        self._transition_sink = transition_sink
        self._pending: list[PendingDecision] = []

    @property
    def collects_data(self) -> bool:
        """Whether decisions are recorded as training data."""
        return self.config.collect_data and self._transition_sink is not None

    @override
    def act(self, observation: ObservationDict, state: HanabiState) -> HanabiMove | None:
        if observation["current_player_offset"] != 0:
            return None
        self.reset(state)
        for _ in range(self.rollouts):
            self.run_rollout(observation)
        self.root_node.focused_state = self._root_state.copy()
        best = self.choose_move(self.root_node)
        if self.collects_data:
            self._pending.append(self._decision(observation, self.root_visits()))
        return best

    @override
    def end_episode(self, final_score: int) -> None:
        if self._transition_sink is not None and self._pending:
            value = value_target(final_score, self.game.max_score)
            self._transition_sink([decision.labelled(value) for decision in self._pending])
        self._pending.clear()

    def reset(self, state: HanabiState) -> None:
        """Starts a new search from `state`."""
        self.player_id = state.cur_player()
        self.root_state = state.copy()
        self.root_node = self.node_type((), self.rules)
        self.children.clear()
        self.visits.clear()
        self.total_reward.clear()
        self.visits[self.root_node] = 0
        self.total_reward[self.root_node] = 0

    def run_rollout(self, observation: ObservationDict) -> None:
        """Re-determinizes this agent's hand and runs one search iteration."""
        self.environment.state = self._root_state.copy()
        self.environment.replace_hand(self.player_id)
        self.root_node.focused_state = self.environment.state
        self.search(self.root_node, observation)

    def search(self, node: MCTSNode, observation: ObservationDict) -> float:
        """Selects a leaf, replays its moves, expands it, evaluates it and backpropagates.

        Returns:
            The reward backpropagated along the path.
        """
        path = self.select(node)
        leaf = path[-1]
        environment = self.environment
        state = environment.state
        depth = 0
        for move in leaf.moves:
            if move not in state.legal_moves():
                # The path is impossible under this determinization.
                reward = float(environment.reward())
                self.backpropagate(path, reward)
                return reward
            result = environment.step(move)
            state = environment.state
            # Index -1 (chance) wraps to the last player, as in the original implementation.
            observation = result.observations["player_observations"][state.cur_player()]
            depth += 1
            if depth > self.max_depth:
                break
        leaf.focused_state = state
        if depth < self.max_depth:
            self.expand(leaf, observation)
        reward = self.simulate(leaf)
        self.backpropagate(path, reward)
        return reward

    def select(self, node: MCTSNode) -> list[MCTSNode]:
        """Descends by UCT until reaching an unexpanded node or an unvisited child."""
        path = []
        while True:
            path.append(node)
            children = self.children.get(node)
            if not children:
                return path
            unexplored = [child for child in children if self.visits[child] == 0]
            if unexplored:
                path.append(unexplored[0])
                return path
            node = self.uct_select(node)

    def expand(self, node: MCTSNode, observation: ObservationDict) -> None:
        """Adds a child for every candidate move of `node`."""
        if node in self.children:
            return
        candidates = node.find_children(observation)
        moves = {self._as_move(candidate) for candidate in candidates}
        self.children[node] = {self.node_type((*node.moves, move), self.rules) for move in moves}

    def simulate(self, node: MCTSNode) -> float:
        """Plays up to `max_simulation_steps` rule-based moves from `node` and returns the reward.

        Note: the rollout continues in `node.focused_state` itself, not a copy.
        """
        environment = self.environment
        environment.state = node.state
        observations = environment.observe_all()
        done = node.is_terminal()
        reward = environment.reward()
        steps = 0
        while not done and steps < self.config.max_simulation_steps:
            player_observations = observations["player_observations"]
            observation = player_observations[player_observations[0]["current_player"]]
            action = self.rollout_agents[observation["current_player"]].choose(observation)
            if action is None:
                raise HanabiSearchError("rollout agent declined to act on its own turn")
            result = environment.step(action)
            observations, reward, done = result.observations, result.reward, result.done
            steps += 1
        return float(reward)

    def backpropagate(self, path: list[MCTSNode], reward: float) -> None:
        """Adds one visit and `reward` to every node on `path`."""
        for node in path:
            self.visits[node] += 1
            self.total_reward[node] += reward

    def uct_select(self, node: MCTSNode) -> MCTSNode:
        """Returns the child maximizing UCT: mean reward plus an exploration bonus."""
        log_visits = log(self.visits[node])

        def uct(child: MCTSNode) -> float:
            visits = self.visits[child]
            return self.total_reward[child] / visits + self.exploration_weight * sqrt(
                log_visits / visits
            )

        return max(self.children[node], key=uct)

    def choose_move(self, node: MCTSNode) -> HanabiMove:
        """Returns the root move with the best mean reward (children visited at most once lose).

        Raises:
            RuntimeError: If `node` is terminal.
        """
        if node.is_terminal():
            raise RuntimeError(f"choose_move called on terminal node {node}")
        children = self.children.get(node)
        if not children:
            return random.choice(node.state.legal_moves())

        def mean_reward(child: MCTSNode) -> float:
            visits = self.visits[child]
            return float("-inf") if visits <= 1 else self.total_reward[child] / visits

        return max(children, key=mean_reward).initial_move()

    def root_visits(self) -> dict[int, float]:
        """Returns visit counts of the root's children, keyed by move uid."""
        game = self.environment.game
        return {
            game.get_move_uid(child.initial_move()): self.visits[child]
            for child in self.children.get(self.root_node, ())
        }

    @property
    def _root_state(self) -> HanabiState:
        assert self.root_state is not None, "search not started"
        return self.root_state

    def _as_move(self, candidate: MoveDict | HanabiMove) -> HanabiMove:
        if isinstance(candidate, HanabiMove):
            return candidate
        return self.environment.build_move(candidate)

    def _decision(self, observation: ObservationDict, visits: dict[int, float]) -> PendingDecision:
        encoded = self.environment.vectorized_observation(observation["pyhanabi"])
        return PendingDecision(
            observation=np.asarray(encoded, dtype=np.float32),
            policy=visit_policy(visits, self.environment.num_moves()),
        )
