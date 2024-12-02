from math import sqrt, log
from collections import defaultdict
import ray
from pyhanabi import HanabiState, HanabiMove
from agents.mcts import mcts_env
from agents.mcts.mcts_node import MCTS_Node
from agents.rule_based.rule_based_agents import VanDenBerghAgent
from agents.rule_based.ruleset import Ruleset
from rl_env import Agent
import numpy as np


class MCTS_Agent(Agent):
    """Agent based on Redeterminizing Information Set Monte Carlo Tree Search"""

    def __init__(self, config):
        """Initialize the agent."""

        self.children = dict()
        self.Q = defaultdict(int)
        self.N = defaultdict(int)
        self.root_node = None
        self.root_state = None
        self.player_id = config["player_id"]

        self.max_rollout_num = config.get("max_rollout_num", 1000)
        self.max_simulation_steps = config.get("max_simulation_steps", 3)
        self.max_depth = config.get("max_depth", 60)
        self.exploration_weight = config.get("exploration_weight", 2.5)

        self.rules = config.get("rules", [
            Ruleset.tell_most_information_factory(True),
            Ruleset.tell_anyone_useful_card,
            Ruleset.tell_dispensable_factory(8),
            Ruleset.complete_tell_useful,
            Ruleset.complete_tell_dispensable,
            Ruleset.complete_tell_unplayable,
            Ruleset.play_probably_safe_factory(0.7, False),
            Ruleset.play_probably_safe_late_factory(0.4, 5),
            Ruleset.discard_most_confident,
        ])

        self.rules = None

        self.agents = [VanDenBerghAgent(config) for _ in range(config["players"])]
        self.determine_type = mcts_env.DetermineType.RESTORE
        self.score_type = mcts_env.ScoreType.SCORE
        self.mcts_type = config["mcts_types"][config["player_id"]]

        self.environment = mcts_env.make(
            "Hanabi-Full",
            num_players=config["players"],
            mcts_player=config["player_id"],
            determine_type=self.determine_type,
            score_type=self.score_type,
        )
        self.max_information_tokens = config.get("information_tokens", 8)

        self.num_actions = config['num_actions']
        self.training_data = []
        self.collect_data = config.get('collect_data', False)

    def act(self, observation, state):
        if observation["current_player_offset"] != 0:
            return None

        self.reset(state)
        rollout = 0

        while rollout < self.max_rollout_num:
            self.environment.state = self.root_state.copy()
            self.environment.replace_hand(self.player_id)
            self.root_node.focused_state = self.environment.state

            path, reward = self.mcts_search(self.root_node, observation)
            rollout += 1

        self.root_node.focused_state = self.root_state.copy()
        best_node = self.mcts_choose(self.root_node)

        # Collect training data
        if self.collect_data:
            self.record_training_data(observation, self.root_node)

        return best_node.initial_move()

    def mcts_search(self, node, observation):
        path = self.mcts_select(node)
        leaf = path[-1]
        depth = 0

        environment = self.environment
        state = environment.state
        max_depth = self.max_depth

        for move in leaf.moves:
            legal_moves = state.legal_moves()
            if move not in legal_moves:
                reward = self.environment.reward()
                self.mcts_backpropagate(path, reward)
                return path, reward

            observations, reward, done, _ = environment.step(move)
            state = environment.state
            current_player = state.cur_player()
            observation = observations["player_observations"][current_player]

            depth += 1
            if depth > max_depth:
                break

        leaf.focused_state = state

        if depth < max_depth:
            self.mcts_expand(leaf, observation)

        reward = self.mcts_simulate(leaf)
        self.mcts_backpropagate(path, reward)

        return path, reward

    def mcts_choose(self, node):
        """'Choose the best successor of node. (Choose a move from the root node)"""

        if node.is_terminal():
            raise RuntimeError(f"choose called on terminal node {node}")
        if node not in self.children:
            return node.find_random_child()

        return max(
            self.children[node],
            key=lambda n: float("-inf") if self.N[n] <= 1 else self.Q[n] / self.N[n],
        )

    def mcts_select(self, node):
        """Find an unexplored descendent of `node`"""
        path = []
        while True:
            path.append(node)
            if node not in self.children or not self.children[node]:
                return path

            unexplored = [child for child in self.children[node] if self.N[child] == 0]
            if unexplored:
                n = unexplored[0]
                path.append(n)
                return path

            node = self.uct_select(node)

    def mcts_expand(self, node, observation):
        """Expand the `node` with all children"""
        if node in self.children:
            return

        moves = node.find_children(observation)
        if moves and isinstance(moves[0], dict):
            build_move = self.environment._build_move
            moves = {build_move(action) for action in moves}
        else:
            moves = set(moves)

        self.children[node] = set(
            MCTS_Node(node.moves + (move,), self.rules) for move in moves
        )

    def mcts_simulate(self, node):
        """Run a simulation from the given node."""
        environment = self.environment
        environment.state = node.focused_state
        observations = environment._make_observation_all_players()
        agents = self.agents

        done = node.is_terminal()
        reward = environment.reward()
        steps = 0

        while not done and steps < self.max_simulation_steps:
            player_observations = observations["player_observations"]
            current_agent = player_observations[0]["current_player"]
            observation = player_observations[current_agent]
            agent = agents[observation["current_player"]]
            current_player_action = agent.act(observation)

            observations, reward, done, _ = environment.step(current_player_action)
            steps += 1

            if done:
                break

        return reward

    def mcts_backpropagate(self, path, reward):
        """Backpropagate the result of a simulation through the tree"""
        for node in path:
            self.N[node] += 1
            self.Q[node] += reward

    def uct_select(self, node):
        "Select a child of node, balancing exploration & exploitation"
        log_N_node = log(self.N[node])
        exploration_weight = self.exploration_weight

        def uct(child):
            Q_child = self.Q[child]
            N_child = self.N[child]
            return (Q_child / N_child) + exploration_weight * sqrt(
                log_N_node / N_child
            )

        return max(self.children[node], key=uct)

    def reset(self, state):
        """Reset the agent with a new state"""
        self.player_id = state.cur_player()
        self.root_state = state.copy()
        self.root_node = MCTS_Node((), self.rules)

        self.children.clear()
        self.Q.clear()
        self.N.clear()

        self.N[self.root_node] = 0
        self.Q[self.root_node] = 0

    def record_training_data(self, observation, node):
        """Record training data for the current state."""
        state_vector = self.environment.vectorized_observation(observation['pyhanabi'])

        # Get visit counts for child nodes
        visit_counts = np.zeros(self.num_actions)

        value_counts = np.zeros(self.num_actions)

        for child in self.children[node]:
            move = child.initial_move()
            action_idx = self.environment.game.get_move_uid(move)
            visit_counts[action_idx] = self.N[child]
            value_counts[action_idx] = self.Q[child] / self.N[child] if self.N[child] > 0 else 0

        # Normalize visit counts to get policy targets
        sum_counts = np.sum(visit_counts)
        if sum_counts > 0:
            policy_targets = visit_counts / sum_counts
        else:
            policy_targets = np.ones_like(visit_counts) / len(visit_counts)

        # Normalize value counts to get value target
        sum_values = np.sum(value_counts)
        if sum_values > 0:
            value_targets = value_counts / sum_values
        else:
            value_targets = np.ones_like(value_counts) / len(value_counts)

        self.training_data.append((state_vector, policy_targets, value_targets))


class PMCTS_Agent(MCTS_Agent):
    def __init__(self, config):
        super().__init__(config)
        if not ray.is_initialized():
            ray.init(include_dashboard=False)

        num_workers = 8
        worker_max_rollout_num = self.max_rollout_num // num_workers

        self.workers = [
            MCTS_Worker.remote(config, worker_max_rollout_num)
            for _ in range(num_workers)
        ]

    def act(self, observation, state):
        if observation["current_player_offset"] != 0:
            return None

        self.reset(state)

        # Serialize the state to JSON
        state_json = state.to_json()

        # Serialize the observation
        observation_copy = observation.copy()
        current_player = observation_copy["current_player"]
        observation_copy["pyhanabi"] = current_player

        # Use the workers to perform MCTS search
        worker_futures = [
            worker.perform_mcts_search.remote(observation_copy, state_json)
            for worker in self.workers
        ]

        # Get results from workers
        worker_results = ray.get(worker_futures)

        # Merge results
        merged_root_children_stats = {}
        for root_children_stats in worker_results:
            for move_json, stats in root_children_stats.items():
                if move_json not in merged_root_children_stats:
                    merged_root_children_stats[move_json] = {'Q': 0, 'N': 0}
                merged_root_children_stats[move_json]['Q'] += stats['Q']
                merged_root_children_stats[move_json]['N'] += stats['N']

        self.root_node.focused_state = self.root_state.copy()
        best_move = self.mcts_choose(self.root_node, merged_root_children_stats)

        # Collect training data
        if self.collect_data:
            self.record_training_data(observation, merged_root_children_stats)

        return best_move
    
    def mcts_choose(self, node, merged_root_children_stats):
        """Choose the best successor of the root node. (Redefinition for parallelization)"""

        if node.is_terminal():
            raise RuntimeError(f"choose called on terminal node {node}")
        if not merged_root_children_stats:
            return node.find_random_child()

        best_move_json = None
        best_value = float('-inf')
        for move_json, stats in merged_root_children_stats.items():
            N = stats['N']
            Q = stats['Q']
            if N <= 1:
                value = float('-inf')
            else:
                value = Q / N
            if value > best_value:
                best_value = value
                best_move_json = move_json

        return HanabiMove.from_json(best_move_json)
    
    def record_training_data(self, observation, merged_root_children_stats):
        """Record training data for the current state."""
        state_vector = self.environment.vectorized_observation(observation['pyhanabi'])

        visit_counts = np.zeros(self.num_actions, dtype=np.float64)
        value_counts = np.zeros(self.num_actions, dtype=np.float64)

        move_jsons = list(merged_root_children_stats.keys())
        Ns = np.array([merged_root_children_stats[move_json]['N'] for move_json in move_jsons], dtype=np.float64)
        Qs = np.array([merged_root_children_stats[move_json]['Q'] for move_json in move_jsons], dtype=np.float64)

        moves = [HanabiMove.from_json(move_json) for move_json in move_jsons]
        action_indices = [self.environment.game.get_move_uid(move) for move in moves]
        visit_counts[action_indices] = Ns

        with np.errstate(divide='ignore', invalid='ignore'):
            value_counts[action_indices] = np.divide(Qs, Ns, where=Ns != 0)

        # Normalize visit counts to get policy targets
        sum_counts = np.sum(visit_counts)
        if sum_counts > 0:
            policy_targets = visit_counts / sum_counts
        else:
            policy_targets = np.ones_like(visit_counts) / len(visit_counts)

        # Normalize value counts to get value target
        sum_values = np.sum(value_counts)
        if sum_values > 0:
            value_targets = value_counts / sum_values
        else:
            value_targets = np.ones_like(value_counts) / len(value_counts)

        # Record the training data
        self.training_data.append((state_vector, policy_targets, value_targets))



@ray.remote(num_cpus=0.5)
class MCTS_Worker:
    def __init__(self, config, max_rollout_num):
        self.agent = MCTS_Agent(config)
        self.max_rollout_num = max_rollout_num

    def perform_mcts_search(self, observation, state_json):
        # Reconstruct the state and observation
        state = HanabiState.from_json(state_json)
        current_player = observation['pyhanabi']
        observation['pyhanabi'] = state.observation(current_player)

        self.agent.reset(state)
        rollout = 0

        while rollout < self.max_rollout_num:
            self.agent.environment.state = self.agent.root_state.copy()
            self.agent.environment.replace_hand(self.agent.player_id)
            self.agent.root_node.focused_state = self.agent.environment.state

            path, reward = self.agent.mcts_search(self.agent.root_node, observation)
            rollout += 1

        root_children = self.agent.children[self.agent.root_node]
        root_children_stats = {}
        for child in root_children:
            move_json = child.initial_move().to_json()
            Q_value = self.agent.Q[child]
            N_value = self.agent.N[child]
            root_children_stats[move_json] = {'Q': Q_value, 'N': N_value}

        return root_children_stats
