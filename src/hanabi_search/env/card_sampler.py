"""Sampling hidden cards consistent with what a player knows (for MCTS re-determinization).

To be replaced by an engine C API function (planned engine work), at which point
this module becomes a thin wrapper.
"""

import random
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Self

from hanabi_search.env.cards import HanabiCard, HanabiCardKnowledge
from hanabi_search.env.game import HanabiGame


@dataclass(frozen=True, slots=True)
class DeckShape:
    """Composition of the full deck.

    Attributes:
        num_colors: Number of colors.
        num_ranks: Number of ranks.
        initial_counts: Copies of each card in the full deck, indexed by
            `color * num_ranks + rank`.
        cards: One `HanabiCard` per index, in the same order.
    """

    num_colors: int
    num_ranks: int
    initial_counts: tuple[int, ...]
    cards: tuple[HanabiCard, ...]

    @classmethod
    def from_game(cls, game: HanabiGame) -> Self:
        """Reads the deck composition of `game`."""
        num_colors, num_ranks = game.num_colors(), game.num_ranks()
        pairs = [(color, rank) for color in range(num_colors) for rank in range(num_ranks)]
        return cls(
            num_colors=num_colors,
            num_ranks=num_ranks,
            initial_counts=tuple(game.num_cards(color, rank) for color, rank in pairs),
            cards=tuple(HanabiCard(color, rank) for color, rank in pairs),
        )


class DeckCounts:
    """Mutable multiset of cards that could still be in a hidden position.

    Args:
        shape: Composition of the full deck; the counts start full.
    """

    __slots__ = ("_shape", "card_count", "total_count")

    def __init__(self, shape: DeckShape) -> None:
        self._shape = shape
        self.card_count = list(shape.initial_counts)
        self.total_count = sum(shape.initial_counts)

    def copy(self) -> DeckCounts:
        """Returns an independent copy of the current counts."""
        other = DeckCounts.__new__(DeckCounts)
        other._shape = self._shape
        other.card_count = list(self.card_count)
        other.total_count = self.total_count
        return other

    def reset(self) -> None:
        """Restores the full deck."""
        self.card_count = list(self._shape.initial_counts)
        self.total_count = sum(self._shape.initial_counts)

    def cards(self) -> list[HanabiCard]:
        """Returns every remaining card, one entry per copy, in index order."""
        return [
            card
            for card, count in zip(self._shape.cards, self.card_count, strict=True)
            for _ in range(count)
        ]

    def remove_card(self, color: int, rank: int) -> None:
        """Removes one copy of (color, rank) if any is left."""
        index = color * self._shape.num_ranks + rank
        if self.card_count[index] > 0:
            self.card_count[index] -= 1
            self.total_count -= 1

    def remove_cards(self, cards: Sequence[HanabiCard]) -> None:
        """Removes one copy of each card in `cards`."""
        for card in cards:
            self.remove_card(card.color(), card.rank())

    def remove_by_knowledge(self, knowledge: HanabiCardKnowledge) -> None:
        """Removes every card that hints have ruled out."""
        num_ranks = self._shape.num_ranks
        for color in range(self._shape.num_colors):
            color_plausible = knowledge.color_plausible(color)
            for rank in range(num_ranks):
                if not (color_plausible and knowledge.rank_plausible(rank)):
                    index = color * num_ranks + rank
                    self.total_count -= self.card_count[index]
                    self.card_count[index] = 0

    def remove_by_hands(
        self, player: int, hands: Sequence[Sequence[HanabiCard]], card_index: int = -1
    ) -> None:
        """Removes cards in all hands.

        `player`'s hand is skipped entirely if `card_index` is -1; otherwise only
        the card at `card_index` is skipped.
        """
        for other_player, hand in enumerate(hands):
            if other_player == player and card_index == -1:
                continue
            for index, card in enumerate(hand):
                if other_player == player and index == card_index:
                    continue
                self.remove_card(card.color(), card.rank())

    def remove_by_own_hand(
        self, player: int, hands: Sequence[Sequence[HanabiCard]], card_index: int
    ) -> None:
        """Removes `player`'s cards except the one at `card_index`."""
        for index, card in enumerate(hands[player]):
            if index != card_index:
                self.remove_card(card.color(), card.rank())

    def remove_by_fireworks(self, fireworks: Sequence[int]) -> None:
        """Removes one copy of every card already played on the fireworks."""
        num_ranks = self._shape.num_ranks
        for color, height in enumerate(fireworks):
            for index in range(color * num_ranks, color * num_ranks + height):
                if self.card_count[index] > 0:
                    self.card_count[index] -= 1
                    self.total_count -= 1

    def __str__(self) -> str:
        num_ranks = self._shape.num_ranks
        rows = [
            f"color {color}: {self.card_count[color * num_ranks : (color + 1) * num_ranks]}"
            for color in range(self._shape.num_colors)
        ]
        return "\n".join([*rows, f"total: {self.total_count}"])


class CardSampler:
    """Samples cards for hidden positions, consistent with the visible game state.

    Uses Python's `random` module, so seeding it makes sampling reproducible.

    Args:
        shape: Composition of the full deck.
    """

    __slots__ = ("_deck", "_shape")

    def __init__(self, shape: DeckShape) -> None:
        self._shape = shape
        self._deck = DeckCounts(shape)

    def sample_hand(
        self,
        player: int,
        hand_size: int,
        player_hands: Sequence[Sequence[HanabiCard]],
        discard_pile: Sequence[HanabiCard],
        fireworks: Sequence[int],
        card_knowledge: Sequence[HanabiCardKnowledge] | None = None,
        additional_cards: Sequence[HanabiCard] = (),
    ) -> list[HanabiCard]:
        """Samples a full hand for `player`, card by card.

        Retries from scratch whenever a position has no consistent card left.

        Args:
            player: Absolute id of the player whose hand is sampled.
            hand_size: Number of cards to sample.
            player_hands: Current hands of all players.
            discard_pile: Discarded cards.
            fireworks: Firework heights by color.
            card_knowledge: The player's knowledge per hand position, if any.
            additional_cards: Further cards known not to be available.

        Returns:
            The sampled hand.
        """
        deck = self._deck
        deck.reset()
        deck.remove_cards(discard_pile)
        deck.remove_by_hands(player, player_hands, card_index=-1)
        deck.remove_by_fireworks(fireworks)
        deck.remove_cards(additional_cards)

        sampled_hand: list[HanabiCard] = []
        while len(sampled_hand) < hand_size:
            sampled_hand = []
            for card_index in range(hand_size):
                candidates = deck.copy()
                candidates.remove_cards(sampled_hand)
                candidates.remove_by_own_hand(player, player_hands, card_index)
                if card_knowledge:
                    candidates.remove_by_knowledge(card_knowledge[card_index])
                cards = candidates.cards()
                if not cards:
                    break
                sampled_hand.append(random.choice(cards))
        return sampled_hand

    def sample_card(
        self,
        player: int,
        card_index: int,
        player_hands: Sequence[Sequence[HanabiCard]],
        discard_pile: Sequence[HanabiCard],
        fireworks: Sequence[int],
        card_knowledge: Sequence[HanabiCardKnowledge] | None = None,
        additional_cards: Sequence[HanabiCard] = (),
    ) -> HanabiCard | None:
        """Samples one card for `player`'s position `card_index`, or `None` if none fits."""
        cards = self.valid_cards(
            player,
            card_index,
            player_hands,
            discard_pile,
            fireworks,
            card_knowledge,
            additional_cards,
        )
        return random.choice(cards) if cards else None

    def valid_cards(
        self,
        player: int,
        card_index: int,
        player_hands: Sequence[Sequence[HanabiCard]],
        discard_pile: Sequence[HanabiCard],
        fireworks: Sequence[int],
        card_knowledge: Sequence[HanabiCardKnowledge] | None = None,
        additional_cards: Sequence[HanabiCard] = (),
    ) -> list[HanabiCard]:
        """Returns every card (one entry per copy) that fits `player`'s position `card_index`."""
        deck = self._deck
        deck.reset()
        deck.remove_cards(discard_pile)
        deck.remove_by_hands(player, player_hands, card_index)
        deck.remove_by_fireworks(fireworks)
        deck.remove_cards(additional_cards)
        if card_knowledge is not None:
            deck.remove_by_knowledge(card_knowledge[card_index])
        return deck.cards()
