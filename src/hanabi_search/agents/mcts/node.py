"""Search-tree node for information-set MCTS."""

from collections.abc import Sequence

from hanabi_search.agents.rule_based.ruleset import Rule
from hanabi_search.env import HanabiMove, HanabiState, MoveDict, ObservationDict


class MCTSNode:
    """A node identified by the moves leading to it from the root.

    Nodes are compared and hashed by their move sequence, so the same sequence
    reached under different determinizations shares statistics.

    Args:
        moves: Moves from the root to this node.
        rules: If given, children are the moves these rules propose instead of all
            legal moves.
    """

    __slots__ = ("_hash", "focused_state", "moves", "rules")

    def __init__(self, moves: tuple[HanabiMove, ...], rules: Sequence[Rule] | None) -> None:
        self.moves = moves
        self._hash = hash(moves)
        self.rules = rules
        # State of the determinization currently being searched through this node.
        self.focused_state: HanabiState | None = None

    @property
    def state(self) -> HanabiState:
        """The focused state; set before the node is expanded or evaluated."""
        assert self.focused_state is not None, "node has no focused state"
        return self.focused_state

    def find_children(self, observation: ObservationDict) -> list[MoveDict] | list[HanabiMove]:
        """Returns candidate moves: those proposed by the rules, or all legal moves."""
        if self.is_terminal():
            return []
        if self.rules is not None:
            return [action for rule in self.rules if (action := rule(observation)) is not None]
        return self.state.legal_moves()

    def initial_move(self) -> HanabiMove:
        """Returns the first move from the root, i.e. the move this subtree evaluates."""
        return self.moves[0]

    def is_terminal(self) -> bool:
        """Returns whether the focused state is a finished game."""
        return self.state.is_terminal()

    def __str__(self) -> str:
        return str(self.moves)

    def __repr__(self) -> str:
        return str(self.moves)

    def __hash__(self) -> int:
        return self._hash

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, MCTSNode):
            return NotImplemented
        return self.moves == other.moves
