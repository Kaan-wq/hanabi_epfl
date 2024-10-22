from collections import defaultdict
from math import log, sqrt

import os
import numpy as np
import ray
import tensorflow as tf
from agents.alphazero.alphazero_network import AlphaZeroNetwork
from agents.alphazero.alphazero_node import AlphaZeroNode
from pyhanabi import HanabiMove, HanabiState

from ..mcts.mcts_agent import MCTS_Agent
import logging
import datetime


class AlphaZero_Agent(MCTS_Agent):
    """Agent based on AlphaZero."""

    def __init__(self, config):
        super().__init__(config)
        agent_name = config['agent_name']

        self.num_actions = config['num_actions']
        self.obs_shape = config['obs_shape']
        self.network = config['network']

        self.training_data = []

        self.max_rollout_num = 10
        self.max_simulation_steps = 0
        self.max_depth = 60
        self.exploration_weight = 2.5

        # Dirichlet noise parameters
        self.dirichlet_epsilon = 0.25
        self.dirichlet_alpha = 0.03

        # Setup logger
        self.logger = logging.getLogger(agent_name)
        self.logger.setLevel(logging.DEBUG)
        log_directory = "agent_logs"
        os.makedirs(log_directory, exist_ok=True)
        log_filename = os.path.join(log_directory, f"{agent_name}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
        fh = logging.FileHandler(log_filename)
        fh.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        if not self.logger.handlers:
            self.logger.addHandler(fh)

        self.logger.info("AlphaZero_Agent initialized.")

    def act(self, observation, state):
        """Act method returns the action based on the observation using AlphaZero."""
        if observation["current_player_offset"] != 0:
            self.logger.debug("Not the current player's turn.")
            return None

        self.reset(state)
        rollout = 0

        self.logger.info("=== Starting MCTS Search ===")
        self.logger.info(f"Initial Observation: {observation}")
        self.logger.info(f"Max Rollouts: {self.max_rollout_num}")

        while rollout < self.max_rollout_num:
            self.logger.debug(f"--- Rollout #{rollout + 1} ---")
            self.environment.state = self.root_state.copy()
            self.environment.replace_hand(self.player_id)
            self.root_node.focused_state = self.environment.state

            path, reward = self.mcts_search(self.root_node, observation)
            rollout += 1
            self.logger.debug(f"Rollout #{rollout} completed with reward={reward}")

        self.logger.info("=== MCTS Search Completed ===")
        self.root_node.focused_state = self.root_state.copy()
        best_node = self.mcts_choose(self.root_node)

        # Log actual score of the game
        self.logger.info(f"Actual Game Score: {self.environment.reward()}")

        # Collect training data
        self.record_training_data(observation, self.root_node)

        # Log the chosen action
        chosen_action = best_node.initial_move()
        self.logger.info(f"Chosen Action: {chosen_action} | Visit Count: {self.N[best_node]}")

        return chosen_action

    def mcts_search(self, node, observation):
        """Perform MCTS search from the given node."""
        path = self.mcts_select(node)
        leaf = path[-1]
        depth = 0

        environment = self.environment
        state = environment.state
        max_depth = self.max_depth

        self.logger.debug(f"Traversing Path: {[str(n) for n in path]}")

        for move in leaf.moves:
            legal_moves = state.legal_moves()
            if move not in legal_moves:
                reward = self.environment.reward()
                self.logger.debug(f"Illegal move encountered: {move}")
                self.mcts_backpropagate(path, reward)
                return path, reward

            observations, reward, done, _ = environment.step(move)
            state = environment.state
            current_player = state.cur_player()
            observation = observations["player_observations"][current_player]

            depth += 1
            self.logger.debug(f"Move: {move} | Reward: {reward} | Depth: {depth}")

            if depth > max_depth:
                self.logger.debug(f"Max depth reached.")
                break

        leaf.focused_state = state

        if depth < max_depth:
            self.mcts_expand(leaf, observation)

        reward = self.mcts_simulate(leaf)
        self.mcts_backpropagate(path, reward)

        self.logger.debug(f"MCTS Search completed with final observation={observation}")
        self.logger.debug(f"MCTS Search completed with reward={reward}")

        return path, reward

    def mcts_choose(self, node):
        """Choose the best successor of the root node."""
        if node.is_terminal():
            raise RuntimeError(f"choose called on terminal node {node}")
        if not self.children[node]:
            return node.find_random_child()

        chosen_node = max(
            self.children[node],
            key=lambda n: float('-inf') if self.N[n] == 0 else self.N[n]
        )
        self.logger.debug(f"Best node chosen by MCTS: {chosen_node} | Visit Count: {self.N[chosen_node]}")
        return chosen_node

    def mcts_select(self, node):
        """Find an unexplored descendant of `node`."""
        path = []
        while True:
            path.append(node)
            if node not in self.children or not self.children[node]:
                self.logger.debug(f"Node {node} is a leaf node.")
                return path

            unexplored = [child for child in self.children[node] if self.N[child] == 0]
            if unexplored:
                n = unexplored[0]
                path.append(n)
                self.logger.debug(f"Expanding unexplored child: {n}")
                return path

            node = self.uct_select(node)
            self.logger.debug(f"Selected child for exploration: {node}")

    def mcts_expand(self, node, observation, from_rules=True):
        """Expand the `node` with all possible children with their policy and value."""
        if node in self.children:
            self.logger.debug(f"Node {node} already expanded.")
            return

        obs_vector = self.environment.vectorized_observation(observation['pyhanabi'])
        obs_vector = tf.cast(tf.reshape(obs_vector, [1, 1, -1, 1]), tf.float32)

        policy_logits, value = self.network(obs_vector)
        policy = tf.nn.softmax(policy_logits)
        node.value = value.numpy()[0][0]

        # Log network outputs
        self.logger.debug(f"Network Value Output: {node.value}")
        self.logger.debug(f"Network Policy Logits: {policy_logits.numpy()}")
        self.logger.debug(f"Network Policy after Softmax: {policy.numpy()}")

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

        # Log normalized policy
        self.logger.debug(f"Normalized Policy: {policy.numpy()}")

        # Add Dirichlet noise to the prior probabilities at the root node
        if node == self.root_node:
            epsilon = self.dirichlet_epsilon
            alpha = self.dirichlet_alpha
            legal_moves = np.flatnonzero(mask_moves)
            dirichlet_noise = np.random.dirichlet([alpha] * len(legal_moves))

            # Adjust the policy
            policy_values = policy.numpy()[0][legal_moves]
            policy_values = (1 - epsilon) * policy_values + epsilon * dirichlet_noise

            # Normalize the adjusted policy
            policy_values /= np.sum(policy_values)

            # Assign back to policy
            policy_numpy = policy.numpy()[0]
            policy_numpy[legal_moves] = policy_values
            policy = tf.convert_to_tensor([policy_numpy], dtype=tf.float32)

            # Log Dirichlet noise addition
            self.logger.debug(f"Dirichlet Noise Added: {policy_numpy}")

        self.children[node] = set()
        for move in moves:
            child_node = AlphaZeroNode(node.moves + (move,), self.rules)
            action_idx = self.environment.game.get_move_uid(move)
            child_node.P = policy[0][action_idx]
            self.children[node].add(child_node)
            self.logger.debug(f"Added Child Node: {child_node} | P={child_node.P}")

    def uct_select(self, node):
        """Select a child of node, balancing exploration and exploitation using prior probabilities."""
        log_N_node = log(self.N[node] + 1)

        puct_values = {}
        for child in self.children[node]:
            N = self.N[child]
            Q = self.Q[child] / N if N > 0 else 0
            P = child.P
            U = self.exploration_weight * P * sqrt(log_N_node) / (1 + N)
            puct = Q + U
            puct_values[child] = puct
            self.logger.debug(f"PUCT Calculation | Child: {child} | Q: {Q} | U: {U} | PUCT: {puct}")

        # Select the child with the highest PUCT value
        selected_child = max(self.children[node], key=lambda c: puct_values[c])
        self.logger.debug(f"Selected Child: {selected_child} | PUCT: {puct_values[selected_child]}")

        return selected_child

    def mcts_simulate(self, node):
        """Return the value estimate for the given node."""
        self.logger.debug(f"Simulating Node: {node} | Value: {node.value}")
        return node.value

    def mcts_backpropagate(self, path, reward):
        """Backpropagate the result of a simulation through the tree"""
        self.logger.debug(f"Backpropagating reward: {reward}")
        for node in path:
            self.N[node] += 1
            self.Q[node] += reward
            #self.logger.debug(f"Node: {node} | N: {self.N[node]} | Q: {self.Q[node]}")

    def reset(self, state):
        """Reset the agent with a new state."""
        self.player_id = state.cur_player()
        self.root_state = state.copy()
        self.root_node = AlphaZeroNode((), self.rules)

        self.children.clear()
        self.Q.clear()
        self.N.clear()

        self.N[self.root_node] = 0
        self.Q[self.root_node] = 0
        self.logger.info("Agent state has been reset.")

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

        # Log training data
        self.logger.debug("=== Recording Training Data ===")
        self.logger.debug(f"State Vector Shape: {len(state_vector)}")
        self.logger.debug(f"Policy Targets: {policy_targets}")
        self.logger.debug("=== End of Training Data ===")


class AlphaZeroP_Agent(AlphaZero_Agent):
    def __init__(self, config):
        super().__init__(config)
        if not ray.is_initialized():
            ray.init(include_dashboard=False)

        num_workers = 8
        worker_max_rollout_num = self.max_rollout_num // num_workers
        config['max_rollout_num'] = worker_max_rollout_num

        network_weights_id = ray.put(self.network.get_weights())
        config['network'] = None

        self.workers = [
            AlphaZero_Worker.remote(config, network_weights_id)
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
        results = defaultdict(int)
        for root_children_stats in worker_results:
            for move_json, N_value in root_children_stats.items():
                if move_json not in results:
                    results[move_json] = 0
                results[move_json] += N_value

        self.root_node.focused_state = self.root_state.copy()
        best_move = self.mcts_choose(self.root_node, results)

        # Collect training data
        self.record_training_data(observation, results)

        return best_move
    
    def mcts_choose(self, node, results):
        """Choose the best successor of the root node. (Redefinition for parallelization)"""
        if node.is_terminal():
            raise RuntimeError(f"choose called on terminal node {node}")
        if not results:
            return node.find_random_child()

        best_move_json = None
        best_value = float('-inf')
        for move_json, N_value in results.items():
            if N_value <= 0:
                value = float('-inf')
            else:
                value = N_value
            if value > best_value:
                best_value = value
                best_move_json = move_json

        return HanabiMove.from_json(best_move_json)
    
    def record_training_data(self, observation, results):
        """Record training data for the current state. (Redefinition for parallelization)"""
        state_vector = self.environment.vectorized_observation(observation['pyhanabi'])

        # Get visit counts for child nodes
        visit_counts = np.zeros(self.num_actions)
        for move_json, N_value in results.items():
            move = HanabiMove.from_json(move_json)
            action_idx = self.environment.game.get_move_uid(move)
            visit_counts[action_idx] = N_value

        # Normalize visit counts to get policy targets
        sum_counts = np.sum(visit_counts)
        if sum_counts > 0:
            policy_targets = visit_counts / sum_counts
        else:
            policy_targets = np.ones_like(visit_counts) / len(visit_counts)

        self.training_data.append((state_vector, policy_targets, None))


@ray.remote(num_cpus=1)
class AlphaZero_Worker:
    def __init__(self, config, network_weights_id):
        self.agent = AlphaZero_Agent(config)
        self.agent.network = AlphaZeroNetwork(self.agent.num_actions, self.agent.obs_shape)
        self.agent.network.set_weights(network_weights_id)

    def perform_mcts_search(self, observation, state_json):
        state = HanabiState.from_json(state_json)
        current_player = observation['pyhanabi']
        observation['pyhanabi'] = state.observation(current_player)

        self.agent.reset(state)
        rollout = 0

        while rollout < self.agent.max_rollout_num:
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