import json
from typing import Any, cast

import pytest

from hanabi_search.config import GameConfig
from hanabi_search.env import (
    CHANCE_PLAYER_ID,
    AgentObservationType,
    HanabiEndOfGameType,
    HanabiEnv,
    HanabiGame,
    HanabiMove,
    HanabiState,
    ObservationEncoder,
)
from hanabi_search.exceptions import (
    ConfigError,
    EngineCallError,
    IllegalMoveError,
    SerializationError,
)
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


@pytest.mark.parametrize(
    ("field", "value"), [("num_players", 9), ("hand_size", 0), ("rng_state", "garbage")]
)
def test_game_from_out_of_range_json_raises(field: str, value: object) -> None:
    # Regression: invalid parameters used to abort the process.
    payload = json.loads(HanabiGame({"players": 3, "seed": 5}).to_json())
    payload[field] = value
    with pytest.raises(SerializationError):
        HanabiGame.from_json(json.dumps(payload))


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


@pytest.mark.parametrize(
    ("field", "value"), [("hands", []), ("cur_player", 7), ("fireworks", [9]), ("deck", None)]
)
def test_state_from_out_of_range_json_raises(state: HanabiState, field: str, value: object) -> None:
    payload = json.loads(state.to_json())
    engine_state = json.loads(payload["HanabiState"])
    engine_state[field] = value
    payload["HanabiState"] = json.dumps(engine_state)
    with pytest.raises(SerializationError):
        HanabiState.from_json(json.dumps(payload))


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


def test_out_of_range_return_and_deal_specific_moves_are_illegal(state: HanabiState) -> None:
    # Regression: these read outside the engine's hand array.
    num_players = state.game.num_players()
    assert state.move_is_legal(HanabiMove.get_return_move(0, 0))
    assert not state.move_is_legal(HanabiMove.get_return_move(0, num_players))
    assert not state.move_is_legal(HanabiMove.get_return_move(-1, 0))

    state.apply_move(HanabiMove.get_return_move(0, 0))
    assert state.cur_player() == CHANCE_PLAYER_ID
    for player, index, color, rank in [(num_players, 0, 0, 0), (0, 9, 0, 0), (0, 0, 7, 0)]:
        move = HanabiMove.get_deal_specific_move(index, player, color, rank)
        assert not state.move_is_legal(move)


@pytest.mark.parametrize(
    "params",
    [
        {"players": 1},
        {"players": 6},
        {"colors": 0},
        {"hand_size": 30},
        {"players": 3, "hand_size": 9},  # fits the deck, but hands hold at most 8 cards
        {"observation_type": 5},
    ],
)
def test_game_with_invalid_parameters_raises(params: dict[str, int]) -> None:
    # Regression: invalid parameters used to abort the process.
    with pytest.raises(ConfigError):
        HanabiGame(params)


def test_get_move_out_of_range_raises() -> None:
    game = HanabiGame({"players": 2})
    assert game.get_move(game.max_moves() - 1).type() is not None
    with pytest.raises(EngineCallError):
        game.get_move(game.max_moves())


def test_observation_encoder_with_unknown_type_raises() -> None:
    with pytest.raises(EngineCallError):
        ObservationEncoder(HanabiGame({"players": 2}), cast(Any, 7))


def test_illegal_move_raises_and_leaves_state_unchanged(state: HanabiState) -> None:
    # Regression: an illegal move used to abort the process.
    before = state.to_json()
    with pytest.raises(IllegalMoveError):
        state.apply_move(HanabiMove.get_play_move(9))
    assert state.to_json() == before


def test_deal_outside_chance_turn_raises(state: HanabiState) -> None:
    assert state.cur_player() != CHANCE_PLAYER_ID
    with pytest.raises(EngineCallError):
        state.deal_random_card()


def test_remove_knowledge_out_of_range_raises(state: HanabiState) -> None:
    state.remove_knowledge(0, 0)
    with pytest.raises(EngineCallError):
        state.remove_knowledge(state.num_players(), 0)
    with pytest.raises(EngineCallError):
        state.remove_knowledge(0, 9)


def test_observation_and_knowledge_out_of_range_raise(state: HanabiState) -> None:
    with pytest.raises(EngineCallError):
        state.observation(state.num_players())
    knowledge = state.observation(0).card_knowledge()[0][0]
    assert knowledge.color_plausible(0)
    with pytest.raises(EngineCallError):
        knowledge.color_plausible(5)
    with pytest.raises(EngineCallError):
        knowledge.rank_plausible(-1)


@pytest.mark.parametrize("game", [GameConfig(players=p) for p in (2, 3, 5)] + [SMALL_GAME])
def test_legal_moves_are_the_legal_ones_of_all_moves(game: GameConfig) -> None:
    env = make_env(game, seed=4)
    for _ in range(12):
        if env.state.is_terminal():
            break
        legal = env.state.legal_moves()
        assert legal == [m for m in env.game.all_moves() if env.state.move_is_legal(m)]
        assert [env.game.get_move_uid(m) for m in legal] == sorted(
            env.game.get_move_uid(m) for m in legal
        )
        env.step(legal[-1])


def test_all_moves_are_indexed_by_id(env: HanabiEnv) -> None:
    moves = env.game.all_moves()
    assert len(moves) == env.game.max_moves()
    assert [env.game.get_move_uid(move) for move in moves] == list(range(len(moves)))
    assert env.game.all_moves() is moves


def test_no_legal_moves_while_a_card_must_be_dealt() -> None:
    state = HanabiState(HanabiGame({"players": 2}))
    assert state.cur_player() == CHANCE_PLAYER_ID
    assert state.legal_moves() == []


def test_hands_from_arrays_match_per_player(env: HanabiEnv) -> None:
    advance_randomly(env, 8)
    hands = env.state.player_hands()
    arrays = env.state.hand_arrays()
    for player, hand in enumerate(hands):
        assert env.state.player_hand(player) == hand
        assert arrays.sizes[player] == len(hand)
        assert arrays.colors[player, : len(hand)].tolist() == [card.color() for card in hand]
        assert (arrays.colors[player, len(hand) :] == -1).all()
        assert all(card.valid() for card in hand)


@pytest.mark.parametrize("player", [-1, 2])
def test_player_hand_out_of_range_raises(env: HanabiEnv, player: int) -> None:
    with pytest.raises(IndexError):
        env.state.player_hand(player)


def test_engine_errors_carry_the_engine_reason(state: HanabiState) -> None:
    with pytest.raises(ConfigError, match=r"hand_size must be in \[1, 8\]"):
        HanabiGame({"players": 3, "hand_size": 9})
    with pytest.raises(EngineCallError, match=r"player 5 is not in \[0, 2\)"):
        state.observation(5)
    with pytest.raises(EngineCallError, match="not a chance turn"):
        state.deal_random_card()
    payload = json.loads(state.to_json())
    engine_state = json.loads(payload["HanabiState"])
    engine_state["cur_player"] = 7
    payload["HanabiState"] = json.dumps(engine_state)
    with pytest.raises(SerializationError, match="invalid JSON: 'cur_player'"):
        HanabiState.from_json(json.dumps(payload))


def test_illegal_move_error_says_why_before_the_state(state: HanabiState) -> None:
    discard = HanabiMove.get_discard_move(0)  # Not legal while information tokens are full.
    with pytest.raises(IllegalMoveError, match=r"Illegal move .*: move .* is not legal") as error:
        state.apply_move(discard)
    assert "\nState:\n" in str(error.value)
