import pytest

from hanabi_search.env import (
    CHANCE_PLAYER_ID,
    AgentObservationType,
    HanabiEndOfGameType,
    HanabiEnv,
    HanabiGame,
    HanabiMove,
    HanabiState,
)
from hanabi_search.exceptions import SerializationError
from tests.support import SMALL_GAME, advance_randomly, make_env


def test_game_parameters() -> None:
    game = HanabiGame({"players": 4, "colors": 3, "ranks": 4, "seed": 1, "observation_type": 0})
    assert (game.num_players(), game.num_colors(), game.num_ranks()) == (4, 3, 4)
    assert game.hand_size() == 4  # 4+ players hold 4 cards.
    assert game.max_score() == 12
    assert game.observation_type() is AgentObservationType.MINIMAL
    assert [game.num_cards(0, rank) for rank in range(4)] == [3, 2, 2, 1]


def test_game_json_round_trip() -> None:
    game = HanabiGame({"players": 3, "seed": 5})
    assert HanabiGame.from_json(game.to_json()).to_json() == game.to_json()


def test_game_from_invalid_json_raises() -> None:
    with pytest.raises(SerializationError):
        HanabiGame.from_json("not json")


def test_new_state_starts_with_deals(state: HanabiState) -> None:
    fresh = HanabiState(state.game)
    assert fresh.cur_player() == CHANCE_PLAYER_ID
    assert fresh.end_of_game_status() is HanabiEndOfGameType.NOT_FINISHED


def test_copied_state_is_independent_and_keeps_its_game(env: HanabiEnv) -> None:
    # Regression: copies used to point at a raw C++ game (StateParentGame), so their
    # observations reported the wrong number of fireworks and to_json() crashed.
    copy = env.state.copy()
    assert copy.game is env.state.game
    assert len(copy.observation(0).fireworks()) == 5
    assert HanabiState.from_json(copy.to_json()).to_json() == copy.to_json()

    copy.apply_move(copy.legal_moves()[0])
    assert str(copy) != str(env.state)


def test_state_json_round_trip_mid_game(env: HanabiEnv) -> None:
    advance_randomly(env, 15)
    restored = HanabiState.from_json(env.state.to_json())
    assert str(restored) == str(env.state)
    assert [str(m) for m in restored.legal_moves()] == [str(m) for m in env.state.legal_moves()]


@pytest.mark.parametrize("payload", ["[]", '{"HanabiGame": "{}"}', "garbage"])
def test_state_from_invalid_json_raises(payload: str) -> None:
    with pytest.raises(SerializationError):
        HanabiState.from_json(payload)


def test_fireworks_follow_number_of_colors() -> None:
    # Regression: fireworks were read for 5 colors whatever the game.
    env = make_env(SMALL_GAME)
    assert len(env.state.fireworks()) == SMALL_GAME.colors
    assert len(env.state.observation(0).fireworks()) == SMALL_GAME.colors


def test_terminal_state_has_no_legal_moves(env: HanabiEnv) -> None:
    while not env.state.is_terminal():
        env.step(env.state.legal_moves()[0])
    assert env.state.legal_moves() == []
    assert env.state.end_of_game_status() is not HanabiEndOfGameType.NOT_FINISHED


def test_return_and_deal_specific_swap_a_card(env: HanabiEnv) -> None:
    state = env.state
    player = state.cur_player()
    old_hand = state.player_hands()[player]
    state.apply_move(HanabiMove.get_return_move(0, player))
    assert state.cur_player() == CHANCE_PLAYER_ID
    returned = old_hand[0]
    state.apply_move(
        HanabiMove.get_deal_specific_move(0, player, returned.color(), returned.rank())
    )
    assert state.cur_player() == player
    assert state.player_hands()[player] == old_hand
