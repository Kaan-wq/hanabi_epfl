from collections import defaultdict
from math import log, sqrt

import numpy as np
import ray
import torch
import torch.nn.functional as F
from agents.alphazero.alphazero_utils import extract_tensors, replace_tensors
from agents.alphazero.alphazero_node import AlphaZeroNode
from pyhanabi import HanabiMove, HanabiState

from ..mcts.mcts_agent import MCTS_Agent


class AlphaZero_Agent(MCTS_Agent):
    """Agent based on AlphaZero."""

    def __init__(self, config):
        super().__init__(config)

        self.num_actions = config['num_actions']
        self.network = config['network']
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.training_data = []

        self.max_rollout_num = 400
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

        chosen_action = best_node.initial_move()

        return chosen_action

    def mcts_search(self, node, observation):
        """Perform MCTS search from the given node."""
        path = self.mcts_select(node)
        leaf = path[-1]
        depth = 0

        environment = self.environment
        state = environment.state
        max_depth = self.max_depth

        for move in leaf.moves:
            legal_moves = state.legal_moves()
            if move not in legal_moves:
                reward = self.environment.reward() # TODO: Check if this is correct
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
    
    #TODO: Replace name to mcts_choose if does not work
    def alpha_choose(self, node):
        """Choose the best successor of the root node."""
        if node.is_terminal():
            raise RuntimeError(f"choose called on terminal node {node}")
        if not self.children[node]:
            return node.find_random_child()

        chosen_node = max(
            self.children[node],
            key=lambda n: float('-inf') if self.N[n] == 0 else self.N[n]
        )
        return chosen_node

    def mcts_select(self, node):
        """Find an unexplored descendant of `node`."""
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

    def mcts_expand(self, node, observation, from_rules=True):
        """Expand the `node` with all possible children, assigning their policy and value."""
        if node in self.children:
            return

        obs_vector = self.environment.vectorized_observation(observation['pyhanabi'])
        obs_tensor = torch.tensor(obs_vector, dtype=torch.float32, device=self.device).unsqueeze(0)

        with torch.no_grad():
            # TODO: add value head
            policy_logits = self.network(obs_tensor)
            policy = F.softmax(policy_logits, dim=1).squeeze(0)

        if node == self.root_node:
            self.root_policy = policy

        # Determine legal moves
        moves = node.find_children(observation) if from_rules else self.environment.state.legal_moves()
        if not moves:
            return  # No moves to expand

        # Convert moves if they are in dictionary format
        if isinstance(next(iter(moves)), dict):
            build_move = self.environment._build_move
            moves = {build_move(action) for action in moves}

        # Map moves to their unique IDs
        moves_uids = [self.environment.game.get_move_uid(move) for move in moves]

        # Create a mask for valid moves directly on the device
        mask = torch.zeros(self.num_actions, dtype=torch.float32, device=self.device)
        mask[moves_uids] = 1.0

        # Apply the mask to the policy and normalize
        policy *= mask
        policy_sum = policy.sum()
        if policy_sum > 0:
            policy /= policy_sum
        else:
            # Assign uniform probabilities if policy sums to zero
            num_legal_moves = mask.sum()
            if num_legal_moves > 0:
                policy = mask / num_legal_moves
            else:
                return  # No legal moves available

        # Expand the node by adding all valid child nodes
        self.children[node] = set()
        for move in moves:
            action_idx = self.environment.game.get_move_uid(move)
            prior_prob = policy[action_idx].item()
            if prior_prob > 0:
                child_node = AlphaZeroNode(node.moves + (move,), self.rules)
                child_node.P = prior_prob
                self.children[node].add(child_node)

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

        # Select the child with the highest PUCT value
        selected_child = max(self.children[node], key=lambda c: puct_values[c])

        return selected_child
    
    #TODO: Replace name to mcts_simulate if does not work
    def alpha_simulate(self, node):
        """Return the value estimate for the given node."""
        return node.value

    def mcts_backpropagate(self, path, reward):
        """Backpropagate the result of a simulation through the tree"""
        for node in reversed(path):
            self.N[node] += 1
            self.Q[node] += reward

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

        self.training_data.append((state_vector, policy_targets, None, self.root_policy))


class AlphaZeroP_Agent(AlphaZero_Agent):
    def __init__(self, config):
        super().__init__(config)
        if not ray.is_initialized():
            ray.init(include_dashboard=False)

        num_workers = 2
        worker_max_rollout_num = self.max_rollout_num // num_workers
        config['max_rollout_num'] = worker_max_rollout_num

        # Share network weights
        network_ref = ray.put(extract_tensors(self.network))
        config['network'] = network_ref

        self.workers = [
            AlphaZero_Worker.remote(config)
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
            if N_value > best_value:
                best_value = N_value
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


@ray.remote(num_cpus=4)
class AlphaZero_Worker:
    def __init__(self, config):
        self.agent = AlphaZero_Agent(config)
        model_skeleton, model_weights = ray.get(config['network'])
        replace_tensors(model_skeleton, model_weights)
        self.agent.network = model_skeleton
        self.agent.network.eval()  # Set to evaluation mode

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
