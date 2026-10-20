import numpy as np
import pytest

from hanabi_search.agents.alphazero.tree import SearchTree, TreeNode
from hanabi_search.env import HanabiEnv, HanabiMove


def _node(env: HanabiEnv, stats: list[tuple[float, float, float]]) -> TreeNode:
    """A node with one child per (prior, visits, total) entry."""
    moves = env.state.legal_moves()[: len(stats)]
    node = TreeNode.with_priors(moves, np.array([prior for prior, _, _ in stats]))
    node.visits[:] = [visits for _, visits, _ in stats]
    node.totals[:] = [total for _, _, total in stats]
    return node


def test_puct_uses_sqrt_parent_visits_and_normalized_values(env: HanabiEnv) -> None:
    # Q as a score fraction + 2.5 * P * sqrt(9) / (1 + n), parent mean 90 / 9 = 10:
    #   strong: 20/25 + 2.5*0.5*3/5 = 1.55
    #   unvisited: 10/25 + 2.5*0.4*3/1 = 3.4
    #   perfect: 25/25 + 2.5*0.1*3/6 = 1.125
    node = _node(env, [(0.5, 4, 80.0), (0.4, 0, 0.0), (0.1, 5, 125.0)])
    assert node.best_child(9, 90.0, 2.5, 25) == 1
    node.visits[1] = 1  # unvisited -> 0 + 2.5*0.4*3/2 = 1.5 < strong (1.55).
    assert node.best_child(9, 90.0, 2.5, 25) == 0


def test_puct_ties_go_to_the_first_child(env: HanabiEnv) -> None:
    node = _node(env, [(0.5, 0, 0.0), (0.5, 0, 0.0)])
    assert node.best_child(1, 10.0, 2.5, 25) == 0


def test_select_lets_the_prior_decide_over_unvisited_children(env: HanabiEnv) -> None:
    tree = SearchTree(exploration_weight=2.5, max_score=25)
    tree.root = _node(env, [(0.99, 4, 80.0), (0.01, 0, 0.0)])
    tree.root_visits, tree.root_total = 4, 80.0
    path, moves = tree.select()
    assert [index for _, index in path] == [0]
    assert moves == [tree.root.moves[0]]


def test_select_stops_at_unexpanded_child_and_expand_attaches_it(env: HanabiEnv) -> None:
    tree = SearchTree(exploration_weight=2.5, max_score=25)
    assert tree.select() == ([], [])
    assert not tree.is_expanded([])
    tree.expand([], env.state.legal_moves()[:2], np.array([0.7, 0.3]))
    path, _ = tree.select()
    assert not tree.is_expanded(path)
    child_moves: list[HanabiMove] = env.state.legal_moves()[2:4]
    tree.expand(path, child_moves, np.array([0.5, 0.5]))
    assert tree.is_expanded(path)
    deeper, moves = tree.select()
    assert len(deeper) == 2 and moves[1] in child_moves


def test_backpropagate_updates_root_and_path(env: HanabiEnv) -> None:
    tree = SearchTree(exploration_weight=2.5, max_score=25)
    tree.expand([], env.state.legal_moves()[:2], np.array([0.7, 0.3]))
    path, _ = tree.select()
    tree.backpropagate(path, 12.0)
    root = tree.root
    assert root is not None
    assert (tree.root_visits, tree.root_total) == (1, 12.0)
    assert root.visits.tolist() == [1, 0] and root.totals.tolist() == pytest.approx([12.0, 0])
