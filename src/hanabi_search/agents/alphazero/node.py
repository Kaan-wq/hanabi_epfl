"""Search-tree node for AlphaZero: an MCTS node with the network's prior."""

from collections.abc import Sequence

from hanabi_search.agents.mcts.node import MCTSNode
from hanabi_search.agents.rule_based.ruleset import Rule
from hanabi_search.env import HanabiMove


class AlphaZeroNode(MCTSNode):
    """`MCTSNode` with the network's prior for reaching it.

    Args:
        moves: Moves from the root to this node.
        rules: See `MCTSNode`.
    """

    __slots__ = ("prior",)

    def __init__(self, moves: tuple[HanabiMove, ...], rules: Sequence[Rule] | None) -> None:
        super().__init__(moves, rules)
        self.prior = 0.0
