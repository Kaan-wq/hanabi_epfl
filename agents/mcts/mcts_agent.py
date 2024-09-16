import logging
import math
import random
import time

import pyhanabi
from rl_env import Agent

from hanabi_learning_environment.agents.mcts.mcts_node import MCTSNode


class MCTSAgent(Agent):
    """Agent that uses simple Monte Carlo tree search to plan actions."""

    logging.basicConfig(level=logging.INFO)

    def __init__(self, config, mcts_simulations=1000, mcts_c=2.0):
        """Initialize the agent."""
        self.config = config
        self.mcts_simulations = mcts_simulations
        self.mcts_c = mcts_c

        self.nodes = dict()
        self.root_node = None
        self.root_state = None

        self.player_id = config['player_id']

        self.max_time_limit = 10000
        self.max_rollouts = 100

        self.max_sim_steps = 3
        self.exploration_weight = 1.0
        self.max_depth = 100

        # self.env

    def act(self, observation):
        """Act by running Monte Carlo tree search."""

        if observation['current_player_offset'] != 0:
            logging.info('Agent %d skips' % self.player_id)
            return None

        rollout_count = 0
        start_time = time.time()
        elapsed_time = 0

        root = MCTSNode(moves=())

        while rollout_count < self.max_rollouts and elapsed_time < self.max_time_limit:

            # Re-determinize the root hand

            # Run a rollout of MCTS
            path, reward = self.mcts_search(root)

            rollout_count += 1
            elapsed_time = (time.time() - start_time) * 1000

        return None
    
    def mcts_search(self, root_node):
        """Run Monte Carlo tree search from the root."""

        # Get the path to an unexplored or terminal node
        path = self.mcts_select(root_node)
        leaf = path[-1]

        depth = 0
        for move in leaf.moves():
            observations, reward, done, unused_info = self.environment.step(move)
            observation = observations['player_observations'][self.environment.state.cur_player()]
            depth += 1

            if depth > self.max_depth:
                break

        if depth < self.max_depth:
            self.mcts_expand(leaf, observation)

        reward = self.mcts_simulate(leaf)
        self.mcts_backpropagate(path, reward)

        return path, reward

    def mcts_choose(self, node):
        """Choose the best move from the root."""

        return node.best_child(c_param=self.mcts_c)

    def mcts_select(self, node, observation):
        """Select an unexplored child node."""
        path = []

        while True:
            path.append(node)

            # If the node is terminal or unexplored, return the path
            if node not in self.nodes or node.is_terminal():
                # node not in self.nodes SHOULD BE IMPOSSIBLE
                return path
            
            # Take one of the unexplored children
            unexplored = node.children() - self.nodes # TODO Won't work cause of diff between nodes and legal moves
            if unexplored:
                n = unexplored.pop()
                path.append(n)
                node.update_explored(n)
                return path
            
            # Otherwise, select the best child and continue
            node = node.best_child(c_param=self.mcts_c)

    def mcts_expand(self, node, observation):
        """Expand the node."""
        assert node not in self.nodes

        moves = node.children_moves(observation)


        pass

    def mcts_simulate(self, node):
        """Simulate from the node until the end."""
        pass

    def mcts_backpropagate(self, path, reward):
        """Backpropagate the reward."""

        for node in path:
            node.update_visit()
            node.update_reward(reward)

    def mcts_reset(self):
        """Reset the MCTS tree."""
        pass
