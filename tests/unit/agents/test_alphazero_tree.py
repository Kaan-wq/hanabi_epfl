import random
from math import sqrt

import pytest

from hanabi_search.agents.alphazero.tree import SearchTree, TreeNode


def _node(stats: list[tuple[float, int, float]]) -> TreeNode:
    """A node with one child per (prior, visits, total) entry, move ids 0, 1, ...

    Priors must be decreasing and the visited children must come first, as search
    keeps them.
    """
    node = TreeNode.with_priors(list(range(len(stats))), [prior for prior, _, _ in stats])
    assert node.move_uids == list(range(len(stats)))
    node.visits[:] = [visits for _, visits, _ in stats]
    node.totals[:] = [total for _, _, total in stats]
    return node


def _puct(node: TreeNode, index: int, visits: float, total: float) -> float:
    count, prior = node.visits[index], node.priors[index]
    parent_mean = total / visits if visits else 0.0
    mean = node.totals[index] / count if count else parent_mean
    return mean / 25 + 2.5 * prior * sqrt(visits) / (1 + count)


def test_puct_uses_sqrt_parent_visits_and_normalized_values() -> None:
    # Q as a score fraction + 2.5 * P * sqrt(9) / (1 + n), parent mean 90 / 9 = 10:
    #   strong: 20/25 + 2.5*0.5*3/5 = 1.55
    #   perfect: 25/25 + 2.5*0.3*3/6 = 1.375
    #   unvisited: 10/25 + 2.5*0.2*3/1 = 1.9
    node = _node([(0.5, 4, 80.0), (0.3, 5, 125.0), (0.2, 0, 0.0)])
    assert node.best_child(9, 90.0, 2.5, 25) == 2
    node.visits[2] = 1  # unvisited -> 0 + 2.5*0.2*3/2 = 0.75 < strong (1.55).
    assert node.best_child(9, 90.0, 2.5, 25) == 0


def test_puct_ties_go_to_the_first_child() -> None:
    node = _node([(0.5, 0, 0.0), (0.5, 0, 0.0)])
    assert node.best_child(1, 10.0, 2.5, 25) == 0


def test_children_are_ordered_by_decreasing_prior_without_zero_priors() -> None:
    node = TreeNode.with_priors([7, 3, 5, 9], [0.2, 0.5, 0.0, 0.3])
    assert node.move_uids == [3, 9, 7]
    assert node.priors == [0.5, 0.3, 0.2]
    tied = TreeNode.with_priors([4, 2, 6], [0.25, 0.5, 0.25])
    assert tied.move_uids == [2, 4, 6]


def test_best_child_matches_a_full_puct_scan_during_search() -> None:
    rng = random.Random(0)
    tree = SearchTree(exploration_weight=2.5, max_score=25)
    tree.expand([], list(range(12)), [rng.random() for _ in range(12)])
    for _ in range(300):
        path, _ = tree.select()
        visits, total = tree.root_visits, tree.root_total
        for node, index in path:
            scores = [_puct(node, child, visits, total) for child in range(len(node.priors))]
            assert index == scores.index(max(scores))
            visits, total = node.visits[index], node.totals[index]
        if len(path) < 3 and not tree.is_expanded(path):
            tree.expand(path, list(range(8)), [rng.random() for _ in range(8)])
        tree.backpropagate(path, rng.uniform(0, 25))


def test_select_lets_the_prior_decide_over_unvisited_children() -> None:
    tree = SearchTree(exploration_weight=2.5, max_score=25)
    tree.root = _node([(0.99, 4, 80.0), (0.01, 0, 0.0)])
    tree.root_visits, tree.root_total = 4, 80.0
    path, move_uids = tree.select()
    assert [index for _, index in path] == [0]
    assert move_uids == [0]


def test_select_stops_at_unexpanded_child_and_expand_attaches_it() -> None:
    tree = SearchTree(exploration_weight=2.5, max_score=25)
    assert tree.select() == ([], [])
    assert not tree.is_expanded([])
    tree.expand([], [0, 1], [0.7, 0.3])
    path, _ = tree.select()
    assert not tree.is_expanded(path)
    tree.expand(path, [2, 3], [0.5, 0.5])
    assert tree.is_expanded(path)
    deeper, move_uids = tree.select()
    assert len(deeper) == 2 and move_uids[1] in (2, 3)


def test_backpropagate_updates_root_and_path() -> None:
    tree = SearchTree(exploration_weight=2.5, max_score=25)
    tree.expand([], [0, 1], [0.7, 0.3])
    path, _ = tree.select()
    tree.backpropagate(path, 12.0)
    root = tree.root
    assert root is not None
    assert (tree.root_visits, tree.root_total) == (1, 12.0)
    assert root.visits == [1, 0] and root.totals == pytest.approx([12.0, 0])
