import random

import numpy as np
import pytest

from hanabi_search.agents import RULE_BASED_AGENTS
from hanabi_search.agents.rule_based import ruleset
from hanabi_search.agents.rule_based.card_counting import (
    get_card_playability,
    get_max_fireworks,
    get_probability_notcritical,
    get_probability_useless,
    playable_cards,
    useless_cards,
)
from hanabi_search.config import GameConfig, RuleBasedPolicy
from hanabi_search.env import HanabiEnv
from hanabi_search.evaluation import play_games
from tests.support import PLAYER_COUNTS, SMALL_GAME, current_observation, make_env


@pytest.mark.parametrize("policy", sorted(RULE_BASED_AGENTS))
@pytest.mark.parametrize("players", PLAYER_COUNTS)
def test_agent_plays_legal_games(policy: RuleBasedPolicy, players: int) -> None:
    # Flawed, Inner and Piers regress the missing `tell_randomly` rule.
    game = GameConfig(players=players)
    agents = [RULE_BASED_AGENTS[policy](seat, game) for seat in range(players)]
    (result,) = play_games(game, agents, [players])
    assert 0 <= result.score <= game.max_score


@pytest.mark.parametrize("policy", sorted(RULE_BASED_AGENTS))
def test_agent_plays_reduced_game(policy: RuleBasedPolicy) -> None:
    agents = [RULE_BASED_AGENTS[policy](seat, SMALL_GAME) for seat in range(SMALL_GAME.players)]
    (result,) = play_games(SMALL_GAME, agents, [0])
    assert 0 <= result.score <= SMALL_GAME.max_score


def test_agent_returns_none_when_not_its_turn(env: HanabiEnv) -> None:
    observations = env.observe_all()
    waiting = 1 - observations["current_player"]
    agent = RULE_BASED_AGENTS["VanDenBergh"](waiting, GameConfig())
    assert agent.act(observations["player_observations"][waiting], env.state) is None


def test_rule_based_play_is_reproducible() -> None:
    game = GameConfig(players=3)
    first = play_games(game, [RULE_BASED_AGENTS["Inner"](s, game) for s in range(3)], [4])
    second = play_games(game, [RULE_BASED_AGENTS["Inner"](s, game) for s in range(3)], [4])
    assert first[0].moves == second[0].moves


def test_card_counts_at_start(env: HanabiEnv) -> None:
    observation = current_observation(env)
    assert get_max_fireworks(observation) == dict.fromkeys("RYGWB", 5)
    playable = playable_cards(observation)
    assert playable.shape == (5, 5)
    assert playable[:, 0].all() and not playable[:, 1:].any()
    assert not useless_cards(observation).any()


def test_probabilities_follow_unseen_copies(env: HanabiEnv) -> None:
    # Before any hint, P(playable) for every card is (unseen 1s) / (all unseen cards).
    observation = current_observation(env)
    unseen = observation["pyhanabi"].unseen_card_counts()
    expected = unseen[:, 0].sum() / unseen.sum()
    assert np.array_equal(get_card_playability(observation), np.full(5, expected))
    # Only 1s and 5s have no copy to spare (3 and 1 copies; 2-4 have 2).
    notcritical = get_probability_notcritical(observation)
    assert np.all((notcritical > 0) & (notcritical < 1))


def test_probabilities_follow_hints() -> None:
    env = make_env(GameConfig(players=2), seed=1)
    hinted = env.state.player_hands()[1 - env.state.cur_player()][0]
    env.step({"action_type": "REVEAL_RANK", "rank": hinted.rank(), "target_offset": 1})
    observation = current_observation(env)
    playability = get_card_playability(observation)
    assert playability[0] == (1.0 if hinted.rank() == 0 else 0.0)


def test_card_probabilities_are_probabilities() -> None:
    env = make_env(SMALL_GAME, seed=2)
    observation = current_observation(env)
    for probabilities in (get_card_playability(observation), get_probability_useless(observation)):
        assert probabilities.shape == (SMALL_GAME.hand_size,)
        assert np.all((probabilities >= 0) & (probabilities <= 1))
    assert set(get_max_fireworks(observation)) == {"R", "Y"}


def test_tell_randomly_needs_information_tokens(env: HanabiEnv) -> None:
    observation = current_observation(env)
    rng = random.Random(0)
    assert ruleset.tell_randomly(observation, rng) is not None
    observation["information_tokens"] = 0
    assert ruleset.tell_randomly(observation, rng) is None


def test_discard_rules_respect_full_tokens(env: HanabiEnv) -> None:
    observation = current_observation(env)
    assert observation["information_tokens"] == observation["max_information_tokens"]
    assert ruleset.discard_most_confident(observation, random.Random(0)) is None
    assert ruleset.discard_oldest_first(observation, random.Random(0)) is None
