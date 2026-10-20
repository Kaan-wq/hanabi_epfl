"""AlphaZero search tree: per expanded node, its children's statistics in flat lists.

A node is identified by the moves leading to it from the root, so the same move
sequence reached under different determinizations shares statistics. A child's
visit count and value sum live in its parent's arrays; the root's are scalars.
"""

from dataclasses import dataclass
from math import inf, sqrt


@dataclass(slots=True)
class TreeNode:
    """An expanded node and the statistics of its children, in legal-move order.

    Python lists, not NumPy arrays: nodes have a few dozen children at most, where
    a Python loop beats NumPy's per-call overhead.

    Attributes:
        move_uids: Id of the move leading to each child, shape (num_children,).
        priors: The network's prior of each child, shape (num_children,).
        visits: Visit count of each child, shape (num_children,).
        totals: Sum of the rewards backpropagated through each child, in score
            units, shape (num_children,).
        children: Each child's node once it is expanded, else `None`.
    """

    move_uids: list[int]
    priors: list[float]
    visits: list[int]
    totals: list[float]
    children: list[TreeNode | None]

    @classmethod
    def with_priors(cls, move_uids: list[int], priors: list[float]) -> TreeNode:
        """Returns an unvisited node whose children follow the moves `move_uids`."""
        count = len(move_uids)
        return cls(move_uids, priors, [0] * count, [0.0] * count, [None] * count)

    def best_child(
        self, visits: float, total: float, exploration_weight: float, max_score: float
    ) -> int:
        """Returns the index of the child maximizing PUCT: `Q + c * P * sqrt(N) / (1 + n)`.

        `Q` is the child's mean reward as a fraction of `max_score`, on the same
        [0, 1] scale as the exploration term; unvisited children take this node's
        mean reward. Ties go to the first child.

        Args:
            visits: This node's visit count `N`.
            total: Sum of the rewards backpropagated through this node.
            exploration_weight: `c`.
            max_score: Score of a perfect game.
        """
        parent_mean = total / visits if visits > 0 else 0.0
        scale = exploration_weight * sqrt(visits)
        best_score, best_index = -inf, 0
        for index, (count, value, prior) in enumerate(
            zip(self.visits, self.totals, self.priors, strict=True)
        ):
            mean = value / count if count > 0 else parent_mean
            score = mean / max_score + scale * prior / (1 + count)
            if score > best_score:
                best_score, best_index = score, index
        return best_index


@dataclass(slots=True)
class SearchTree:
    """The tree of one search, from the searching player's decision.

    Args:
        exploration_weight: PUCT constant `c`.
        max_score: Score of a perfect game (scales mean rewards to [0, 1]).
    """

    exploration_weight: float
    max_score: float
    root: TreeNode | None = None
    root_visits: float = 0.0
    root_total: float = 0.0

    def select(self) -> tuple[list[tuple[TreeNode, int]], list[int]]:
        """Descends by PUCT from the root to a node that is not expanded yet.

        Returns:
            The path as (node, chosen child index) pairs, and the ids of the moves along it.
        """
        path: list[tuple[TreeNode, int]] = []
        move_uids: list[int] = []
        node = self.root
        visits, total = self.root_visits, self.root_total
        weight, max_score = self.exploration_weight, self.max_score
        while node is not None and node.move_uids:
            index = node.best_child(visits, total, weight, max_score)
            path.append((node, index))
            move_uids.append(node.move_uids[index])
            visits, total = node.visits[index], node.totals[index]
            node = node.children[index]
        return path, move_uids

    def is_expanded(self, path: list[tuple[TreeNode, int]]) -> bool:
        """Returns whether the node at the end of `path` is expanded."""
        if not path:
            return self.root is not None
        parent, index = path[-1]
        return parent.children[index] is not None

    def expand(
        self, path: list[tuple[TreeNode, int]], move_uids: list[int], priors: list[float]
    ) -> None:
        """Expands the node at the end of `path` with a child per move in `move_uids`."""
        node = TreeNode.with_priors(move_uids, priors)
        if not path:
            self.root = node
        else:
            parent, index = path[-1]
            parent.children[index] = node

    def backpropagate(self, path: list[tuple[TreeNode, int]], reward: float) -> None:
        """Adds one visit and `reward` to the root and every child along `path`."""
        self.root_visits += 1
        self.root_total += reward
        for node, index in path:
            node.visits[index] += 1
            node.totals[index] += reward
