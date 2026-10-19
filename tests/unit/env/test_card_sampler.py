import random

import pytest

from hanabi_search.config import GameConfig
from hanabi_search.env import (
    CardSampler,
    DeckCounts,
    DeckShape,
    DeterminizationType,
    DeterminizedEnv,
    HanabiCard,
    HanabiEnv,
)
from tests.support import SMALL_GAME, advance_randomly, make_env


def _unseen_cards(env: HanabiEnv, player: int) -> list[HanabiCard]:
    """Cards the player cannot see: their own hand plus the deck."""
    deck = DeckCounts(DeckShape.from_game(env.game))
    state = env.state
    deck.remove_cards(state.discard_pile())
    deck.remove_by_hands(player, state.player_hands())
    deck.remove_by_fireworks(state.fireworks())
    return deck.cards()


@pytest.mark.parametrize("game", [GameConfig(players=3), SMALL_GAME])
def test_sampled_hand_is_consistent_with_knowledge(game: GameConfig) -> None:
    env = make_env(game, seed=4)
    advance_randomly(env, 12)
    if env.state.is_terminal():
        pytest.skip("game ended early")
    random.seed(0)
    player = env.state.cur_player()
    knowledge = env.state.observation(player).card_knowledge()[0]
    hand_size = len(env.state.player_hands()[player])
    sampler = CardSampler(DeckShape.from_game(env.game))
    state = env.state
    hand = sampler.sample_hand(
        player, hand_size, state.player_hands(), state.discard_pile(), state.fireworks(), knowledge
    )
    assert len(hand) == hand_size
    unseen = _unseen_cards(env, player)
    for card, card_knowledge in zip(hand, knowledge, strict=True):
        assert card_knowledge.color_plausible(card.color())
        assert card_knowledge.rank_plausible(card.rank())
        unseen.remove(card)  # Raises if the hand uses more copies than exist.


def test_valid_cards_excludes_visible_cards(env: HanabiEnv) -> None:
    player = env.state.cur_player()
    sampler = CardSampler(DeckShape.from_game(env.game))
    state = env.state
    cards = sampler.valid_cards(
        player, 0, state.player_hands(), state.discard_pile(), state.fireworks()
    )
    other_hand = state.player_hands()[1 - player]
    for card in set(other_hand):
        full_copies = env.game.num_cards(card.color(), card.rank())
        assert cards.count(card) == full_copies - other_hand.count(card)


def test_deck_copy_keeps_an_empty_deck_empty() -> None:
    # Regression: copying a deck with no cards left used to reset it to a full deck.
    deck = DeckCounts(DeckShape.from_game(make_env(GameConfig()).game))
    deck.remove_cards(deck.cards())
    assert deck.total_count == 0
    assert deck.copy().cards() == []


def test_sample_card_returns_none_when_nothing_fits() -> None:
    env = make_env(GameConfig())
    sampler = CardSampler(DeckShape.from_game(env.game))
    everything = DeckCounts(DeckShape.from_game(env.game)).cards()
    assert sampler.sample_card(0, 0, [[], []], [], [0] * 5, None, everything) is None


@pytest.mark.parametrize("determinization", list(DeterminizationType))
def test_determinized_rollout_keeps_hands_consistent(
    determinization: DeterminizationType,
) -> None:
    game = GameConfig(players=3)
    real = make_env(game, seed=8)
    searcher = real.state.cur_player()
    env = DeterminizedEnv(game.to_params(), searcher, determinization)
    env.state = real.state.copy()
    random.seed(1)
    env.replace_hand(searcher)
    for _ in range(9):
        if env.state.is_terminal():
            break
        moves = [m for m in env.state.legal_moves() if m.type().name != "PLAY"]
        env.step(moves[0])
        hand_sizes = {len(hand) for hand in env.state.player_hands()}
        assert hand_sizes <= {5, 4}
        assert (
            sum(map(len, env.state.player_hands()))
            + env.state.deck_size()
            + len(env.state.discard_pile())
            + sum(env.state.fireworks())
            == 50
        )
