import random

import numpy as np
import pytest
import torch

from hanabi_search.agents import AlphaZeroAgent
from hanabi_search.agents.base import run_steps
from hanabi_search.config import AlphaZeroAgentConfig, GameConfig, SelfPlayConfig
from hanabi_search.env import HanabiEnv, HanabiMove, ObservationDict
from hanabi_search.networks import AlphaZeroNetwork
from tests.support import current_observation

GAME = GameConfig(players=2)


def _agent(
    env: HanabiEnv,
    config: AlphaZeroAgentConfig | None = None,
    self_play: SelfPlayConfig | None = None,
) -> AlphaZeroAgent:
    torch.manual_seed(0)
    network = AlphaZeroNetwork(env.num_moves(), env.vectorized_observation_shape()[0], (32, 16))
    agent = AlphaZeroAgent(
        env.state.cur_player(),
        GAME,
        config or AlphaZeroAgentConfig(),
        network,
        self_play=self_play,
    )
    agent.begin_episode(random.Random(0))
    return agent


def _root_with_children(
    agent: AlphaZeroAgent, env: HanabiEnv, stats: list[tuple[float, int]]
) -> list[HanabiMove]:
    """Gives the root one child per (prior, visits) entry, in legal-move order."""
    agent.reset(env.state)
    agent.root_node.focused_state = env.state
    moves = env.state.legal_moves()[: len(stats)]
    agent.tree.expand(
        [], [env.game.get_move_uid(move) for move in moves], [prior for prior, _ in stats]
    )
    root = agent.tree.root
    assert root is not None
    root.visits[:] = [visits for _, visits in stats]
    return moves


def _search(agent: AlphaZeroAgent, observation: ObservationDict) -> int:
    """Runs one search iteration from a fresh determinization; returns how many
    evaluations it requested."""
    agent.redeterminize()
    requests = 0
    steps = agent.search_steps(agent.root_node, observation)
    try:
        request = next(steps)
        while True:
            requests += 1
            request = steps.send(agent.evaluate(request))
    except StopIteration:
        return requests


def test_first_search_expands_the_root_with_priors_summing_to_one(env: HanabiEnv) -> None:
    agent = _agent(env, AlphaZeroAgentConfig())
    agent.reset(env.state)
    assert _search(agent, current_observation(env)) == 1
    root = agent.tree.root
    assert root is not None
    assert root.move_uids == env.state.legal_move_uids().tolist()
    assert sum(root.priors) == pytest.approx(1.0)
    assert 0 <= agent.tree.root_total <= GAME.max_score


def test_terminal_leaf_is_valued_by_its_score(env: HanabiEnv) -> None:
    agent = _agent(env)
    while not env.state.is_terminal():
        env.step(env.state.legal_moves()[0])
    agent.reset(env.state)
    assert _search(agent, env.observe_all()["player_observations"][0]) == 0
    assert agent.tree.root_total == env.state.score()
    assert agent.tree.root is None


def test_leaf_at_depth_limit_is_valued_by_the_network_without_children(env: HanabiEnv) -> None:
    agent = _agent(env, AlphaZeroAgentConfig(max_depth=1))
    agent.reset(env.state)
    observation = current_observation(env)
    _search(agent, observation)  # Expands the root.
    assert _search(agent, observation) == 1
    root = agent.tree.root
    assert root is not None
    (leaf,) = np.flatnonzero(root.visits)
    assert root.children[leaf] is None
    # The network's (untrained) value, not the current score of 0.
    assert root.totals[leaf] > 0


def test_act_matches_driving_act_steps(env: HanabiEnv) -> None:
    observation = current_observation(env)
    moves = []
    for drive in (False, True):
        agent = _agent(env, AlphaZeroAgentConfig(rollouts=8))
        if drive:
            moves.append(run_steps(agent.act_steps(observation, env.state), agent.evaluate))
        else:
            moves.append(agent.act(observation, env.state))
    assert moves[0] == moves[1]
    assert isinstance(moves[0], HanabiMove) and env.state.move_is_legal(moves[0])


def _root_priors(env: HanabiEnv, self_play: SelfPlayConfig | None) -> dict[str, float]:
    agent = _agent(env, self_play=self_play)
    agent.reset(env.state)
    _search(agent, current_observation(env))
    root = agent.tree.root
    assert root is not None
    return dict(zip(map(str, root.move_uids), root.priors, strict=True))


def test_root_noise_only_in_self_play(env: HanabiEnv) -> None:
    greedy = _root_priors(env, None)
    assert _root_priors(env, SelfPlayConfig(dirichlet_fraction=0.0)) == pytest.approx(greedy)
    noisy = _root_priors(env, SelfPlayConfig(dirichlet_fraction=0.5))
    assert noisy.keys() == greedy.keys()
    assert sum(noisy.values()) == pytest.approx(1.0)
    assert noisy != pytest.approx(greedy)


def _chosen_moves(agent: AlphaZeroAgent, env: HanabiEnv, count: int) -> list[str]:
    _root_with_children(agent, env, [(0.3, 10), (0.3, 10), (0.4, 30)])
    return [str(agent.choose_move(agent.root_node)) for _ in range(count)]


def test_greedy_without_self_play(env: HanabiEnv) -> None:
    agent = _agent(env)
    most_visited = str(env.state.legal_moves()[2])
    assert set(_chosen_moves(agent, env, 50)) == {most_visited}


def test_temperature_draws_first_moves_by_visits_then_plays_greedily(env: HanabiEnv) -> None:
    agent = _agent(env, self_play=SelfPlayConfig(temperature_moves=600))
    most_visited = str(env.state.legal_moves()[2])
    drawn = _chosen_moves(agent, env, 600)
    assert drawn.count(most_visited) / len(drawn) == pytest.approx(0.6, abs=0.06)
    assert set(_chosen_moves(agent, env, 20)) == {most_visited}
    agent.end_episode(0)
    assert len(set(_chosen_moves(agent, env, 20))) > 1
