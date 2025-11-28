import random
import ujson as json
import pyhanabi

class MCTS_Node:
    """
    Node for MCTS. Represents an information set in the game.
    """
    
    __slots__ = ['moves', '_hash','focused_state', 'rules']

    moves: list[pyhanabi.HanabiMove]
    hash: int
    focused_state: pyhanabi.HanabiState
    rules: list[pyhanabi.HanabiMove]

    def __init__(self, moves, rules) -> None: ...

    def find_children(self, observation) -> list[pyhanabi.HanabiMove]: ...

    def find_random_child(self) -> pyhanabi.HanabiMove: ...

    def initial_move(self) -> pyhanabi.HanabiMove: ...

    def is_terminal(self) -> bool: ...

    def __str__(self) -> str: ...

    def __repr__(self) -> str: ...

    def __hash__(self) -> int: ...

    def __eq__(self, other) -> bool: ...

    def to_json(self) -> str: ...
    
    @classmethod
    def from_json(cls, json_node_str) -> MCTS_Node: ...