from math import log, sqrt

import numpy as np
import ray
import tensorflow as tf
from agents.αzero.αzero_node import AlphaZeroNode
from pyhanabi import HanabiState, HanabiMove
from collections import defaultdict

from ..mcts.mcts_agent import MCTS_Agent


class AlphaZero_Agent(MCTS_Agent):
    """Agent based on AlphaZero."""

    def __init__(self, config):
        super().__init__(config)

        self.num_actions = config['num_actions']
        self.network = config['network']
        self.optimizer = config['optimizer']
        self.loss_fn = config['loss_fn']

        self.training_data = []

        self.max_rollout_num = 100
        self.max_simulation_steps = 0
        self.max_depth = 60
        self.exploration_weight = 2.5

    def act(self, observation, state):
        """Act method returns the action based on the observation using AlphaZero."""
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
        self.record_training_data(observation, self.root_node)

        return best_node.initial_move()
    
    def mcts_choose(self, node):
        """Choose the best successor of the root node."""
        if node.is_terminal():
            raise RuntimeError(f"choose called on terminal node {node}")
        if not self.children[node]:
            return node.find_random_child()

        return max(
            self.children[node],
            key=lambda n: float('-inf') if self.N[n] == 0 else self.N[n]
        )

    
    def mcts_expand(self, node, observation, from_rules=True):
        """Expand the `node` with all possible children with their policy and value."""
        if node in self.children:
            return
        
        obs_vector = self.environment.vectorized_observation(observation['pyhanabi'])

        policy_logits, value = self.network(tf.expand_dims(obs_vector, axis=0))
        policy = tf.nn.softmax(policy_logits)
        node.value = value.numpy()[0][0]

        moves = self.environment.state.legal_moves() if not from_rules else node.find_children(observation)
        if moves and isinstance(moves[0], dict):
            build_move = self.environment._build_move
            moves = {build_move(action) for action in moves}
        else:
            moves = set(moves)
        moves_uids = [self.environment.game.get_move_uid(move) for move in moves]

        mask_moves = np.full(self.num_actions, 0)
        if moves_uids:
            mask_moves[moves_uids] = 1

        policy = policy * mask_moves

        policy_sum = tf.reduce_sum(policy)
        if policy_sum > 0:
            policy /= policy_sum

        self.children[node] = set()
        for move in moves:
            child_node = AlphaZeroNode(node.moves + (move,), self.rules)
            action_idx = self.environment.game.get_move_uid(move)
            child_node.P = policy[0][action_idx]
            self.children[node].add(child_node)

    def uct_select(self, node):
        """Select a child of node, balancing exploration and exploitation using prior probabilities."""
        log_N_node = log(self.N[node] + 1)

        def puct(child):
            N = self.N[child]
            Q = self.Q[child] / N if N > 0 else 0
            P = child.P
            U = self.exploration_weight * P * sqrt(log_N_node) / (1 + N)
            return Q + U

        return max(self.children[node], key=puct)
    
    def mcts_simulate(self, node):
        """Return the value estimate for the given node."""
        return node.value
    
    def reset(self, state):
        """Reset the agent with a new state"""

        self.player_id = state.cur_player()
        self.root_state = state.copy()
        self.root_node = AlphaZeroNode((), self.rules)

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
        for child in self.children[node]:
            move = child.initial_move()
            action_idx = self.environment.game.get_move_uid(move)
            visit_counts[action_idx] = self.N[child]

        # Normalize visit counts to get policy targets
        sum_counts = np.sum(visit_counts)
        if sum_counts > 0:
            policy_targets = visit_counts / sum_counts
        else:
            policy_targets = np.ones_like(visit_counts) / len(visit_counts)

        self.training_data.append((state_vector, policy_targets, None))


class AlphaZeroP_Agent(AlphaZero_Agent):
    def __init__(self, config):
        super().__init__(config)
        if not ray.is_initialized():
            ray.init(include_dashboard=False)

        num_workers = 8
        worker_max_rollout_num = self.max_rollout_num // num_workers

        self.workers = [
            AlphaZero_Worker.remote(config, worker_max_rollout_num)
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
        merged_root_children_stats = defaultdict(int)
        for root_children_stats in worker_results:
            for move_json, N_value in root_children_stats.items():
                if move_json not in merged_root_children_stats:
                    merged_root_children_stats[move_json] = 0
                merged_root_children_stats[move_json] += N_value

        self.root_node.focused_state = self.root_state.copy()
        best_move = self.mcts_choose(self.root_node, merged_root_children_stats)

        return best_move
    
    def mcts_choose(self, node, merged_root_children_stats):
        """Choose the best successor of the root node. (Redefinition for parallelization)"""
        if node.is_terminal():
            raise RuntimeError(f"choose called on terminal node {node}")
        if not merged_root_children_stats:
            return node.find_random_child()

        best_move_json = None
        best_value = float('-inf')
        for move_json, N_value in merged_root_children_stats.items():
            if N_value <= 0:
                value = float('-inf')
            else:
                value = N_value
            if value > best_value:
                best_value = value
                best_move_json = move_json

        return HanabiMove.from_json(best_move_json)
    
    def record_training_data(self, observation, node):
        """Record training data for the current state. (Redefinition for parallelization)"""
        state_vector = self.environment.vectorized_observation(observation['pyhanabi'])

        # Get visit counts for child nodes
        visit_counts = np.zeros(self.num_actions)
        for child in self.children[node]:
            move = child.initial_move()
            action_idx = self.environment.game.get_move_uid(move)
            visit_counts[action_idx] = self.N[child]

        # Normalize visit counts to get policy targets
        sum_counts = np.sum(visit_counts)
        if sum_counts > 0:
            policy_targets = visit_counts / sum_counts
        else:
            policy_targets = np.ones_like(visit_counts) / len(visit_counts)

        self.training_data.append((state_vector, policy_targets, None))


@ray.remote(num_cpus=1)
class AlphaZero_Worker:
    def __init__(self, config, max_rollout_num):
        self.agent = AlphaZero_Agent(config)
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
        root_children_stats = defaultdict(int)
        for child in root_children:
            move_json = child.initial_move().to_json()
            N_value = self.agent.N[child]
            root_children_stats[move_json] = N_value

        return root_children_stats