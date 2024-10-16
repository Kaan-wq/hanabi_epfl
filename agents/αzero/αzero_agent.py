from ..mcts.mcts_agent import MCTS_Agent
from agents.αzero.αzero_network import AlphaZeroNetwork
from agents.αzero.αzero_node import AlphaZeroNode
import tensorflow as tf
import numpy as np
from math import log, sqrt
from pyhanabi import HanabiGame

class AlphaZero_Agent(MCTS_Agent):
    """
    Agent that uses MCTS to plan and execute actions.
    """

    def __init__(self, config):
        super().__init__(config)

        self.observation_size = self.environment.vectorized_observation_shape()
        self.num_actions = self.environment.num_moves()
        self.network = AlphaZeroNetwork(self.observation_size, self.num_actions)

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        self.loss_fn = self._create_loss_function()

        self.max_rollout_num = 100
        self.max_simulation_steps = 0
        self.max_depth = 60
        self.exploration_weight = 2.5

    def act(self, observation, state):
        """Act method returns the action based on the observation using MCTS."""
        return super().act(observation, state)
    
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

        #print("\n\n============================================ AlphaZero Expansion ============================================\n\n")
        #print(f"\nPolicy: {policy}\nValue: {value}")
        #print("\nMasked Policy", policy)
        #print("\nNormalized Policy", policy)
        #print("\nMove UIDs", sorted(moves_uids))
        #print("\nMove Rules UIDs", sorted(moves_rules_uids))  
        #for move in moves_rules:
        #    action_idx = self.environment.game.get_move_uid(move)
        #    print(f"\nMove: {move}\nAction Index: {action_idx}\nPolicy: {policy[0][action_idx]}") 
        #print("\n\n=============================================================================================================\n\n")

    def uct_select(self, node):
        """Select a child of node, balancing exploration and exploitation using prior probabilities."""
        log_N_node = log(self.N[node] + 1)

        def puct(child):
            Q = self.Q[child] / self.N[child] if self.N[child] > 0 else 0
            P = child.P
            N = self.N[child]
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

    def _create_loss_function(self):
        """Create the loss function for training."""
        def loss_fn(policy_targets, value_targets, policy_predictions, value_predictions):
            policy_loss = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(labels=policy_targets, logits=policy_predictions)
            )
            value_loss = tf.reduce_mean(tf.square(value_targets - value_predictions))
            return policy_loss + value_loss
        return loss_fn