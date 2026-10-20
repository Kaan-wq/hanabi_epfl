import random

import pytest

from hanabi_search.config import GameConfig
from hanabi_search.env import (
    DeterminizationType,
    DeterminizedEnv,
    HanabiEnv,
    HanabiMove,
    HanabiState,
    RolloutReward,
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


def test_determinized_env_is_reproducible_from_its_rng() -> None:
    game = GameConfig(players=2)
    real = make_env(game, seed=2)
    hands = []
    for _ in range(2):
        env = DeterminizedEnv(game.to_params(), searching_player=0, rng=random.Random(5))
        env.state = real.state.copy()
        env.replace_hand(0)
        hands.append(env.state.player_hands()[0])
    assert hands[0] == hands[1]


def test_advance_changes_the_state_like_step() -> None:
    game = GameConfig(players=3)
    states = []
    for use_step in (True, False):
        real = make_env(game, seed=6)  # Deals use the real game's RNG, so one per run.
        env = DeterminizedEnv(
            game.to_params(), searching_player=real.state.cur_player(), rng=random.Random(2)
        )
        env.state = real.state.copy()
        for _ in range(6):
            move = next(m for m in env.state.legal_moves() if m.type().name != "PLAY")
            if use_step:
                env.step(move)
            else:
                env.advance(move)
        states.append(env.state.to_json())
    assert states[0] == states[1]


@pytest.mark.parametrize("game", [GameConfig(players=p) for p in (2, 3, 5)])
def test_resampling_records_no_moves(game: GameConfig) -> None:
    env = make_env(game, seed=9)
    advance_randomly(env, 7, seed=9)
    state = env.state
    player = (state.cur_player() + 1) % game.players
    history = [str(item) for item in state.move_history()]
    last_moves = {
        observer: [str(item) for item in state.observation(observer).last_moves()]
        for observer in range(game.players)
    }
    own_encoding = env.vectorized_observation(state.observation(player))
    cur_player = state.cur_player()

    resample_hand(state, player, seed=1)

    assert [str(item) for item in state.move_history()] == history
    assert state.cur_player() == cur_player
    for observer in range(game.players):
        assert [str(item) for item in state.observation(observer).last_moves()] == (
            last_moves[observer]
        )
    # The resampled player cannot see their own cards: their view is unchanged.
    assert (env.vectorized_observation(state.observation(player)) == own_encoding).all()


def test_sampling_errors_carry_the_engine_reason(env: HanabiEnv) -> None:
    with pytest.raises(EngineCallError, match=r"player 2 is not in \[0, 2\)"):
        resample_hand(env.state, 2, seed=0)
    remembered = env.state.player_hands()[1]
    with pytest.raises(EngineCallError, match=r"removed_index 5 is not in \[-1, 5\)"):
        restore_hand(env.state, 1, remembered, None, len(remembered), seed=0)


@pytest.mark.parametrize("game", [GameConfig(players=p) for p in (2, 3, 4, 5)] + [SMALL_GAME])
@pytest.mark.parametrize("reward", list(RolloutReward))
def test_rollout_plays_to_the_end_and_returns_the_reward(
    game: GameConfig, reward: RolloutReward
) -> None:
    real = make_env(game, seed=4)
    env = DeterminizedEnv(game.to_params(), real.state.cur_player(), rollout_reward=reward)
    env.state = real.state.copy()
    full_deck = _full_deck_size(env)
    assert env.rollout(1000) == env.reward()
    assert env.state.is_terminal()
    assert _cards_in_game(env.state) == full_deck


def test_rollout_stops_after_max_steps() -> None:
    game = GameConfig(players=3)
    real = make_env(game, seed=4)
    env = DeterminizedEnv(game.to_params(), real.state.cur_player())
    env.state = real.state.copy()
    dealt = len(env.state.move_history())
    env.rollout(0)
    assert len(env.state.move_history()) == dealt
    env.rollout(5)
    moves = [item for item in env.state.move_history()[dealt:] if item.player() >= 0]
    assert len(moves) == 5


def test_rollout_rejects_negative_steps() -> None:
    game = GameConfig(players=2)
    env = DeterminizedEnv(game.to_params(), searching_player=0)
    env.state = make_env(game).state.copy()
    with pytest.raises(EngineCallError, match="max_steps -1 is negative"):
        env.rollout(-1)


@pytest.mark.parametrize("game", [*(GameConfig(players=p) for p in (2, 3, 4, 5)), SMALL_GAME])
@pytest.mark.parametrize("determinization", list(DeterminizationType))
def test_search_returns_a_legal_move_and_root_visits(
    game: GameConfig, determinization: DeterminizationType
) -> None:
    real = make_env(game, seed=4)
    env = DeterminizedEnv(game.to_params(), real.state.cur_player(), determinization)
    env.state = real.state.copy()
    before = str(env.state)
    result = env.search(iterations=100, max_depth=60, rollout_steps=3, exploration_weight=2.5)
    move = env.game.get_move(result.move_uid)
    assert env.state.move_is_legal(move)
    assert sum(result.root_visits.values()) == 99  # The first iteration visits only the root.
    assert all(env.state.move_is_legal(env.game.get_move(uid)) for uid in result.root_visits)
    assert str(env.state) == before


def test_search_rejects_bad_arguments_and_finished_games() -> None:
    game = GameConfig(players=2)
    env = DeterminizedEnv(game.to_params(), searching_player=0)
    env.state = make_env(game).state.copy()
    with pytest.raises(EngineCallError, match="iterations 0 is not positive"):
        env.search(iterations=0, max_depth=60, rollout_steps=3, exploration_weight=2.5)
    env.rollout(1000)
    with pytest.raises(EngineCallError, match="the game is over"):
        env.search(iterations=10, max_depth=60, rollout_steps=3, exploration_weight=2.5)
