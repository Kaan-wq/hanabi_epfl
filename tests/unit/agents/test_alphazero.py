import random

import pytest
import torch

from hanabi_search.agents import AlphaZeroAgent
from hanabi_search.agents.alphazero.node import AlphaZeroNode
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
    agent: AlphaZeroAgent, env: HanabiEnv, stats: list[tuple[float, int, float]]
) -> list[AlphaZeroNode]:
    """Gives the root one child per (prior, visits, total_reward) entry."""
    agent.reset(env.state)
    root = agent.root_node
    children = []
    for move, (prior, visits, total) in zip(env.state.legal_moves(), stats, strict=False):
        child = AlphaZeroNode((move,), agent.rules)
        child.prior = prior
        agent.visits[child] = visits
        agent.total_reward[child] = total
        children.append(child)
    agent.children[root] = set(children)
    return children


def test_puct_uses_sqrt_parent_visits_and_normalized_values(env: HanabiEnv) -> None:
    agent = _agent(env, AlphaZeroAgentConfig(exploration_weight=2.5))
    # Q in score fraction + 2.5 * P * sqrt(9) / (1 + n):
    #   strong: 20/25 + 2.5*0.5*3/5 = 1.55
    #   unvisited: parent mean 10/25 + 2.5*0.4*3/1 = 3.4
    #   perfect: 25/25 + 2.5*0.1*3/6 = 1.125
    strong, unvisited, _perfect = _root_with_children(
        agent, env, [(0.5, 4, 80.0), (0.4, 0, 0.0), (0.1, 5, 125.0)]
    )
    agent.visits[agent.root_node] = 9
    agent.total_reward[agent.root_node] = 90.0
    assert agent.uct_select(agent.root_node) == unvisited
    agent.visits[unvisited] = 1
    agent.total_reward[unvisited] = 0.0
    # unvisited: 0 + 2.5*0.4*3/2 = 1.5 < strong (1.55).
    assert agent.uct_select(agent.root_node) == strong


def test_select_lets_the_prior_decide_over_unvisited_children(env: HanabiEnv) -> None:
    agent = _agent(env)
    strong, _unlikely = _root_with_children(agent, env, [(0.99, 4, 80.0), (0.01, 0, 0.0)])
    agent.visits[agent.root_node] = 4
    agent.total_reward[agent.root_node] = 80.0
    assert agent.select(agent.root_node) == [agent.root_node, strong]


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
    agent = _agent(env, AlphaZeroAgentConfig(use_rules=False))
    agent.reset(env.state)
    assert _search(agent, current_observation(env)) == 1
    children = agent.children[agent.root_node]
    assert len(children) == len(env.state.legal_moves())
    assert sum(child.prior for child in children if isinstance(child, AlphaZeroNode)) == (
        pytest.approx(1.0)
    )
    assert 0 <= agent.total_reward[agent.root_node] <= GAME.max_score


def test_terminal_leaf_is_valued_by_its_score(env: HanabiEnv) -> None:
    agent = _agent(env)
    while not env.state.is_terminal():
        env.step(env.state.legal_moves()[0])
    agent.reset(env.state)
    assert _search(agent, env.observe_all()["player_observations"][0]) == 0
    assert agent.total_reward[agent.root_node] == env.state.score()
    assert agent.root_node not in agent.children


def test_leaf_at_depth_limit_is_valued_by_the_network_without_children(env: HanabiEnv) -> None:
    agent = _agent(env, AlphaZeroAgentConfig(max_depth=1))
    agent.reset(env.state)
    observation = current_observation(env)
    _search(agent, observation)  # Expands the root.
    assert _search(agent, observation) == 1
    (leaf,) = [child for child in agent.children[agent.root_node] if agent.visits[child]]
    assert leaf not in agent.children
    # The network's (untrained) value, not the current score of 0.
    assert agent.total_reward[leaf] > 0


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
    agent = _agent(env, AlphaZeroAgentConfig(use_rules=False), self_play)
    agent.reset(env.state)
    _search(agent, current_observation(env))
    return {
        str(child.initial_move()): child.prior
        for child in agent.children[agent.root_node]
        if isinstance(child, AlphaZeroNode)
    }


def test_root_noise_only_in_self_play(env: HanabiEnv) -> None:
    greedy = _root_priors(env, None)
    assert _root_priors(env, SelfPlayConfig(dirichlet_fraction=0.0)) == pytest.approx(greedy)
    noisy = _root_priors(env, SelfPlayConfig(dirichlet_fraction=0.5))
    assert noisy.keys() == greedy.keys()
    assert sum(noisy.values()) == pytest.approx(1.0)
    assert noisy != pytest.approx(greedy)


def _chosen_moves(agent: AlphaZeroAgent, env: HanabiEnv, count: int) -> list[str]:
    _root_with_children(agent, env, [(0.3, 10, 0.0), (0.3, 10, 0.0), (0.4, 30, 0.0)])
    agent.root_node.focused_state = env.state
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
