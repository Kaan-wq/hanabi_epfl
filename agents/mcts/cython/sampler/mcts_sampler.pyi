import random
import numpy as np
from pyhanabi import HanabiCard
#cimport numpy as cnp

# Precomputed cards and initial deck configuration
PRECOMPUTED_CARDS: list[HanabiCard]
INIT_DECK: list[int]

class MCTS_Sampler:
    """Sampler for re-determinization in MCTS"""
    deck: HanabiDeck

    def __init__(self) -> None: ...

    def sample_hand(
        self,
        player: int,
        original_hand_size: int,
        player_hands: list,
        discard_pile: list,
        fireworks: list,
        card_knowledge: list = None,
        additional_cards: list = None,
    ) -> list: ...

    def sample_card(
        self,
        player: int,
        card_index: int,
        player_hands: list,
        discard_pile: list,
        fireworks: list,
        card_knowledge: list = None,
        additional_cards: list = None,
    ) -> HanabiCard: ...

    def valid_cards(
        self,
        player: int,
        card_index: int,
        player_hands: list,
        discard_pile: list,
        fireworks: list,
        card_knowledge: list = None,
        additional_cards: list = None,
    ) -> list: ...

class HanabiDeck:
    """Deck of Hanabi cards for sampling hands and cards"""
    num_ranks: int
    num_colors: int
    total_count: int
    #card_count: cnp.int_t[::1]

    def __init__(self, card_count=None, total_count=None) -> None: ...
    def get_deck(self) -> list[HanabiCard]: ...
    def remove_by_knowledge(self, card_knowledge) -> None: ...
    def remove_by_cards(self, cards: list) -> None: ...
    def remove_by_hands(self, player: int, hands: list, card_index: int = -1) -> None: ...
    def remove_by_own_hand(self, player: int, hands: list, card_index: int) -> None: ...
    def remove_by_fireworks(self, fireworks: list) -> None: ...
    def reset_deck(self) -> None: ...
    def remove_card(self, color: int, rank: int) -> None: ...
    def is_empty(self) -> bool: ...
    def __str__(self) -> str: ...