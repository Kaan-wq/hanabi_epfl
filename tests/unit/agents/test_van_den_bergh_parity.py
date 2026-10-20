"""The engine's Van den Bergh policy against the Python rules it ports."""

import random

import pytest

from hanabi_search.agents.rule_based import VanDenBerghAgent
from hanabi_search.config import GameConfig
from hanabi_search.env import AgentObservationType, HanabiEnv, ObservationDict
from tests.support import PLAYER_COUNTS, SMALL_GAME, current_observation, make_env

GAMES = [
    GameConfig(players=players, observation_type=observation_type)
    for players in PLAYER_COUNTS
    for observation_type in AgentObservationType
] + [SMALL_GAME]


def python_rule_uid(env: HanabiEnv, agent: VanDenBerghAgent, observation: ObservationDict) -> int:
    """The uid of the first Python rule's move, or -1 if no rule applies."""
    for rule in agent.rules:
        action = rule(observation, agent.rng)
        if action is not None:
            return env.game.get_move_uid(env.to_move(action))
    return -1


@pytest.mark.parametrize("game", GAMES, ids=lambda g: f"{g.players}p-{g.observation_type.name}")
def test_engine_policy_matches_python_rules(game: GameConfig) -> None:
    rng = random.Random(0)
    checked = 0
    for seed in range(20):
        env = make_env(game, seed)
        agent = VanDenBerghAgent(0, game)
        agent.begin_episode(random.Random(seed))
        while not env.state.is_terminal():
            observation = current_observation(env)
            engine_uid = observation["pyhanabi"].van_den_bergh_move_uid()
            expected = python_rule_uid(env, agent, observation)
            assert (engine_uid if engine_uid is not None else -1) == expected, env.state
            checked += 1
            # Mix in random moves to reach states the policy alone avoids.
            if expected < 0 or rng.random() < 0.2:
                env.step(rng.choice(env.state.legal_moves()))
            else:
                env.step(expected)
    assert checked > 300


def test_rule_move_is_none_off_turn() -> None:
    env = make_env(GameConfig(players=3))
    observations = env.observe_all()
    waiting = (observations["current_player"] + 1) % 3
    observation = observations["player_observations"][waiting]
    assert observation["pyhanabi"].van_den_bergh_move_uid() is None


def test_agent_acts_with_the_engine_policy() -> None:
    game = GameConfig(players=2)
    env = make_env(game, seed=3)
    observation = current_observation(env)
    agent = VanDenBerghAgent(observation["current_player"], game)
    agent.begin_episode(random.Random(0))
    action = agent.act(observation, env.state)
    assert action is not None
    assert env.state.move_is_legal(env.to_move(action))
    assert agent.act(env.observe(1 - observation["current_player"]), env.state) is None
