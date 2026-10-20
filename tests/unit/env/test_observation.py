import numpy as np
import pytest

from hanabi_search.config import GameConfig
from hanabi_search.env import HanabiEnv, candidate_counts
from tests.support import PLAYER_COUNTS, SMALL_GAME, advance_randomly, make_env

GAMES = [GameConfig(players=p) for p in PLAYER_COUNTS] + [SMALL_GAME]


def _mid_game(game: GameConfig) -> HanabiEnv:
    env = make_env(game, seed=5)
    advance_randomly(env, 10, seed=5)
    return env


@pytest.mark.parametrize("game", GAMES)
def test_unseen_cards_are_the_deck_and_own_hand(game: GameConfig) -> None:
    env = _mid_game(game)
    state = env.state
    for player in range(game.players):
        # The deck plus the card at position 0, from the engine's own count.
        expected = candidate_counts(state, player, 0, use_knowledge=False)
        for card in state.player_hands()[player][1:]:
            expected[card.color(), card.rank()] += 1
        assert np.array_equal(state.observation(player).unseen_card_counts(), expected)


@pytest.mark.parametrize("game", GAMES)
def test_plausible_cards_follow_knowledge(game: GameConfig) -> None:
    env = _mid_game(game)
    observation = env.state.observation(env.state.cur_player())
    mask = observation.plausible_cards()
    hand_size = env.game.hand_size()
    assert mask.shape == (game.players, hand_size, game.colors, game.ranks)
    for offset, hand in enumerate(observation.card_knowledge()):
        for index, knowledge in enumerate(hand):
            for color in range(game.colors):
                for rank in range(game.ranks):
                    assert mask[offset, index, color, rank] == (
                        knowledge.color_plausible(color) and knowledge.rank_plausible(rank)
                    )
        assert not mask[offset, len(hand) :].any()


def test_true_own_cards_are_plausible(env: HanabiEnv) -> None:
    advance_randomly(env, 12)
    player = env.state.cur_player()
    mask = env.state.observation(player).plausible_cards()
    for index, card in enumerate(env.state.player_hands()[player]):
        assert mask[0, index, card.color(), card.rank()]


def test_card_arrays_are_read_only(env: HanabiEnv) -> None:
    observation = env.state.observation(0)
    with pytest.raises(ValueError, match="read-only"):
        observation.unseen_card_counts()[0, 0] = 9
    with pytest.raises(ValueError, match="read-only"):
        observation.plausible_cards()[0, 0, 0, 0] = False
