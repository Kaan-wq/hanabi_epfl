import random
import ujson as json
import pyhanabi

class MCTS_Node:
    """
    Node for MCTS. Represents an information set in the game.
    """
    
    __slots__ = ['moves', '_hash','focused_state', 'rules']

    def __init__(self, moves, rules):
        self.moves = moves
        self._hash = hash(self.moves)
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
        return random.choice(self.find_children())

    def initial_move(self):
        return self.moves[0]

    def is_terminal(self):
        return self.focused_state.is_terminal()

    def __str__(self):
        return f"{self.moves}"

    def __repr__(self):
        return f"{self.moves}"

    def __hash__(self):
        return self._hash

    def __eq__(self, other):
        if not isinstance(other, MCTS_Node):
            return NotImplemented
        return self.moves == other.moves

    def to_json(self):
        ser_moves = [move.to_json() for move in self.moves]
        return json.dumps({'moves': ser_moves})
    
    @classmethod
    def from_json(cls, json_node_str):
        json_node = json.loads(json_node_str)
        ser_moves = json_node['moves']
        moves = tuple(pyhanabi.HanabiMove.from_json(move_json) for move_json in ser_moves)
        return cls(moves, rules=None)
