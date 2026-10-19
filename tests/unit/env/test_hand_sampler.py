import random

import pytest

from hanabi_search.config import GameConfig
from hanabi_search.env import (
    DeterminizationType,
    DeterminizedEnv,
    HanabiEnv,
    HanabiMove,
    HanabiState,
    candidate_counts,
    resample_hand,
    restore_hand,
)
from hanabi_search.exceptions import ConfigError, EngineCallError
from tests.support import SMALL_GAME, advance_randomly, make_env


def _cards_in_game(state: HanabiState) -> int:
    return (
        state.deck_size()
        + sum(map(len, state.player_hands()))
        + len(state.discard_pile())
        + sum(state.fireworks())
    )


def _full_deck_size(env: HanabiEnv) -> int:
    game = env.game
    return sum(
        game.num_cards(color, rank)
        for color in range(game.num_colors())
        for rank in range(game.num_ranks())
    )


@pytest.mark.parametrize("game", [GameConfig(players=3), SMALL_GAME])
def test_resampled_hand_fits_knowledge_and_card_counts(game: GameConfig) -> None:
    env = make_env(game, seed=4)
    advance_randomly(env, 12)
    if env.state.is_terminal():
        pytest.skip("game ended early")
    player = env.state.cur_player()
    knowledge = env.state.observation(player).card_knowledge()[0]
    hand_size = len(env.state.player_hands()[player])

    resample_hand(env.state, player, seed=0)

    hand = env.state.player_hands()[player]
    assert len(hand) == hand_size
    for card, card_knowledge in zip(hand, knowledge, strict=True):
        assert card_knowledge.color_plausible(card.color())
        assert card_knowledge.rank_plausible(card.rank())
    assert _cards_in_game(env.state) == _full_deck_size(env)
    assert env.state.cur_player() == player


def test_same_seed_gives_same_hands(env: HanabiEnv) -> None:
    hands = []
    for _ in range(2):
        state = env.state.copy()
        resample_hand(state, 1, seed=7)
        hands.append(state.player_hands()[1])
    assert hands[0] == hands[1]


def test_resample_out_of_range_raises(env: HanabiEnv) -> None:
    before = env.state.to_json()
    with pytest.raises(EngineCallError):
        resample_hand(env.state, env.state.num_players(), seed=0)
    assert env.state.to_json() == before


@pytest.mark.parametrize("seed", [-1, 2**64])
def test_seed_out_of_range_raises(env: HanabiEnv, seed: int) -> None:
    with pytest.raises(ConfigError):
        resample_hand(env.state, 1, seed)


def test_restore_without_a_move_gives_the_hand_back(env: HanabiEnv) -> None:
    state = env.state
    remembered = state.player_hands()[1]
    resample_hand(state, 1, seed=3)
    restore_hand(state, 1, remembered, removed_card=None, removed_index=-1, seed=4)
    assert state.player_hands()[1] == remembered


def test_restore_out_of_range_raises(env: HanabiEnv) -> None:
    remembered = env.state.player_hands()[1]
    with pytest.raises(EngineCallError):
        restore_hand(env.state, 1, remembered, None, len(remembered), seed=0)


def test_candidate_counts_cover_the_deck_and_the_position(env: HanabiEnv) -> None:
    state = env.state
    counts = candidate_counts(state, 1, 0, use_knowledge=False)
    assert counts.shape == (env.game.num_colors(), env.game.num_ranks())
    # The deck, plus the card now at position 0.
    assert counts.sum() == state.deck_size() + 1
    card = state.player_hands()[1][0]
    excluded = candidate_counts(state, 1, 0, use_knowledge=False, excluded=[card])
    assert excluded[card.color(), card.rank()] == counts[card.color(), card.rank()] - 1


def test_candidate_counts_follow_hints(env: HanabiEnv) -> None:
    state = env.state
    target = (state.cur_player() + 1) % state.num_players()
    hinted = state.player_hands()[target][0]
    state.apply_move(HanabiMove.get_reveal_rank_move(1, hinted.rank()))
    counts = candidate_counts(state, target, 0)
    assert counts[:, hinted.rank()].sum() == counts.sum() > 0


def test_candidate_counts_out_of_range_raises(env: HanabiEnv) -> None:
    with pytest.raises(EngineCallError):
        candidate_counts(env.state, 1, env.game.hand_size())
    with pytest.raises(EngineCallError):
        candidate_counts(env.state, env.state.num_players(), 0)


@pytest.mark.parametrize("determinization", list(DeterminizationType))
def test_determinized_rollout_keeps_hands_consistent(
    determinization: DeterminizationType,
) -> None:
    game = GameConfig(players=3)
    real = make_env(game, seed=8)
    searcher = real.state.cur_player()
    random.seed(1)
    env = DeterminizedEnv(game.to_params(), searcher, determinization)
    env.state = real.state.copy()
    env.replace_hand(searcher)
    for _ in range(9):
        if env.state.is_terminal():
            break
        moves = [m for m in env.state.legal_moves() if m.type().name != "PLAY"]
        env.step(moves[0])
        hand_sizes = {len(hand) for hand in env.state.player_hands()}
        assert hand_sizes <= {5, 4}
        assert _cards_in_game(env.state) == 50


def test_determinized_env_is_reproducible_from_random_seed() -> None:
    game = GameConfig(players=2)
    real = make_env(game, seed=2)
    hands = []
    for _ in range(2):
        random.seed(5)
        env = DeterminizedEnv(game.to_params(), searching_player=0)
        env.state = real.state.copy()
        env.replace_hand(0)
        hands.append(env.state.player_hands()[0])
    assert hands[0] == hands[1]
