import random

class MCTSNode():
    """
    Node for MCTS. Represents an information set in the game.
    """

    def __init__(self, moves):
        self.moves = moves
        self.state = None
        self._visits = 0
        self._rewards = 0

    def children(self, observation):
        if self.is_terminal():
            return []
        
        return observation['legal_moves']
    
    def random_child(self):
        return random.choice(self.children())
    
    def update_reward(self, reward):
        self._rewards += reward

    def update_visit(self):
        self._visits += 1

    def moves(self):
        return self.moves

    def is_terminal(self):
        return self.state.is_terminal()
    
    def __str__(self):
        return f"{self.moves}"

    def __repr__(self):
        return f"{self.moves}"

    def __hash__(self):
        return hash(self.moves)

    def __eq__(self, other):
        return self.moves == other.moves
    
    