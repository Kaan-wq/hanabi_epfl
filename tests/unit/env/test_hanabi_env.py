import random

import pytest

from hanabi_search.config import GameConfig
from hanabi_search.env import HanabiEnv, HanabiMove
from hanabi_search.exceptions import IllegalMoveError
from tests.support import PLAYER_COUNTS, SMALL_GAME, current_observation, make_env


@pytest.mark.parametrize("players", PLAYER_COUNTS)
def test_random_game_runs_to_completion(players: int) -> None:
    env = HanabiEnv(GameConfig(players=players).to_params(seed=players))
    observations = env.reset()
    rng = random.Random(players)
    done = False
    steps = 0
    while not done:
        player = observations["current_player"]
        legal = observations["player_observations"][player]["legal_moves"]
        result = env.step(rng.choice(legal))
        observations, done = result.observations, result.done
        steps += 1
    assert env.state.is_terminal()
    assert 0 <= env.score() <= 25
    assert steps > 0


@pytest.mark.parametrize("game", [GameConfig(players=p) for p in PLAYER_COUNTS] + [SMALL_GAME])
def test_observation_shapes(game: GameConfig) -> None:
    env = make_env(game, seed=3)
    observations = env.observe_all()
    assert len(observations["player_observations"]) == game.players
    expected_hand = game.hand_size or (5 if game.players < 4 else 4)
    for observation in observations["player_observations"]:
        assert len(observation["observed_hands"]) == game.players
        assert all(len(hand) == expected_hand for hand in observation["observed_hands"])
        assert len(observation["fireworks"]) == game.colors
        assert observation["num_colors"] == game.colors
        assert len(observation["cards_per_rank"]) == game.ranks
        encoded = env.vectorized_observation(observation["pyhanabi"])
        assert len(encoded) == env.vectorized_observation_shape()[0]
        assert set(encoded) <= {0, 1}


def test_own_cards_are_hidden(env: HanabiEnv) -> None:
    for observation in env.observe_all()["player_observations"]:
        assert all(card["color"] is None for card in observation["observed_hands"][0])
        assert all(card["color"] is not None for card in observation["observed_hands"][1])


@pytest.mark.parametrize("game", [GameConfig(players=3), SMALL_GAME])
def test_legal_move_mask_matches_engine(game: GameConfig) -> None:
    env = make_env(game, seed=11)
    for observation in env.observe_all()["player_observations"]:
        uids = observation["legal_moves_as_int"]
        assert all(0 <= uid < env.num_moves() for uid in uids)
        if observation["current_player_offset"] == 0:
            engine_moves = {str(move) for move in env.state.legal_moves()}
            assert {str(env.game.get_move(uid)) for uid in uids} == engine_moves
        else:
            assert uids == []


def test_step_accepts_all_action_forms(env: HanabiEnv) -> None:
    observation = current_observation(env)
    env.step(observation["legal_moves"][0])
    env.step(current_observation(env)["legal_moves_as_int"][0])
    env.step(env.state.legal_moves()[0])


@pytest.mark.parametrize(
    "action",
    [
        {"action_type": "REVEAL_COLOR", "target_offset": 0, "color": "R"},  # cannot hint self
        {"action_type": "PLAY", "card_index": 9},
        {"action_type": "PLAY"},
        {"action_type": "DEAL", "color": "R", "rank": 0},
        {"action_type": "JUMP"},
    ],
)
def test_illegal_actions_raise(env: HanabiEnv, action: dict[str, object]) -> None:
    with pytest.raises(IllegalMoveError):
        env.build_move(action)  # type: ignore[arg-type]


def test_discard_with_full_tokens_is_illegal(env: HanabiEnv) -> None:
    assert env.state.information_tokens() == env.game.max_information_tokens()
    with pytest.raises(IllegalMoveError):
        env.build_move({"action_type": "DISCARD", "card_index": 0})


def test_same_seed_same_game() -> None:
    first, second = make_env(GameConfig(players=3), seed=5), make_env(GameConfig(players=3), seed=5)
    assert str(first.state) == str(second.state)


def test_to_move_passes_engine_moves_through(env: HanabiEnv) -> None:
    move = env.state.legal_moves()[0]
    assert env.to_move(move) is move
    assert isinstance(env.to_move(0), HanabiMove)
