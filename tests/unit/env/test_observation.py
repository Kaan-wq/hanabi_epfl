import numpy as np
import pytest

from hanabi_search.config import GameConfig
from hanabi_search.env import HanabiEnv, candidate_counts
from tests.support import PLAYER_COUNTS, SMALL_GAME, advance_randomly, current_observation, make_env

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


@pytest.mark.parametrize("game", GAMES)
def test_arrays_match_per_card_wrappers(game: GameConfig) -> None:
    env = _mid_game(game)
    for player in range(game.players):
        observation = env.state.observation(player)
        hands = observation.hand_arrays()
        knowledge = observation.knowledge_arrays()
        for offset, (hand, hints) in enumerate(
            zip(observation.observed_hands(), observation.card_knowledge(), strict=True)
        ):
            size = hands.sizes[offset]
            assert size == len(hand) == len(hints)
            assert hands.colors[offset, :size].tolist() == [card.color() for card in hand]
            assert hands.ranks[offset, :size].tolist() == [card.rank() for card in hand]
            assert (hands.colors[offset, size:] == -1).all()
            expected_colors = [-1 if k.color() is None else k.color() for k in hints]
            expected_ranks = [-1 if k.rank() is None else k.rank() for k in hints]
            assert knowledge.hinted_colors[offset, :size].tolist() == expected_colors
            assert knowledge.hinted_ranks[offset, :size].tolist() == expected_ranks
        assert (hands.colors[0] == -1).all()  # Own cards stay hidden.
        colors, ranks = observation.discard_pile_arrays()
        assert list(zip(colors.tolist(), ranks.tolist(), strict=True)) == [
            (card.color(), card.rank()) for card in observation.discard_pile()
        ]
        uids = observation.legal_move_uids()
        assert uids == [env.game.get_move_uid(move) for move in observation.legal_moves()]
        assert (uids != []) == (observation.cur_player_offset() == 0)


def test_observation_dict_matches_wrappers(env: HanabiEnv) -> None:
    advance_randomly(env, 6)
    for player in range(env.num_players):
        observation = env.state.observation(player)
        result = env.observation_dict(observation)
        assert result["observed_hands"] == [
            [card.to_dict() for card in hand] for hand in observation.observed_hands()
        ]
        assert result["card_knowledge"] == [
            [k.to_dict() for k in hand] for hand in observation.card_knowledge()
        ]
        assert result["discard_pile"] == [card.to_dict() for card in observation.discard_pile()]
        assert result["legal_moves"] == [move.to_dict() for move in observation.legal_moves()]


def test_legal_move_dicts_are_copies(env: HanabiEnv) -> None:
    first = current_observation(env)["legal_moves"]
    first[0]["action_type"] = "CHANGED"
    assert current_observation(env)["legal_moves"][0]["action_type"] != "CHANGED"


def test_encoding_is_a_bit_array(env: HanabiEnv) -> None:
    encoded = env.vectorized_observation(env.state.observation(0))
    assert encoded.dtype == np.uint8
    assert encoded.shape == (env.vectorized_observation_shape()[0],)
    assert set(np.unique(encoded).tolist()) <= {0, 1}


def test_hand_and_knowledge_arrays_are_read_only(env: HanabiEnv) -> None:
    observation = env.state.observation(0)
    with pytest.raises(ValueError, match="read-only"):
        observation.hand_arrays().colors[0, 0] = 1
    with pytest.raises(ValueError, match="read-only"):
        observation.knowledge_arrays().hinted_ranks[0, 0] = 1
