import random

class MCTS_Node():
    """
    Node for MCTS. Represents an information set in the game.
    """

    def __init__(self, moves, rules):
        self.moves = moves
        self.focused_state = None
        self.rules = rules

    def find_children(self, observation):
        assert self.focused_state is not None

        if self.is_terminal():
            return []
        
        if self.rules is not None:
            actions_by_rules = [rule(observation) for rule in self.rules]
            children = [action for action in actions_by_rules if action is not None]
        else:
            children = self.focused_state.legal_moves()
        
        return children
    
    def find_random_child(self):
        return random.choice(self.children())
    
    def initial_move(self):
        return self.moves[0]

    def is_terminal(self):
        return self.focused_state.is_terminal()
    
    def __str__(self):
        return f"{self.moves}"

    def __repr__(self):
        return f"{self.moves}"

    def __hash__(self):
        return hash(self.moves)

    def __eq__(self, other):
        return self.moves == other.moves