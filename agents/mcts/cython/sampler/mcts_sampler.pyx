# mcts_sampler.pyx

import random
import numpy as np
from pyhanabi import HanabiCard

# Precomputed cards and initial deck configuration
PRECOMPUTED_CARDS = [
    HanabiCard(color, rank)
    for color in range(5)
    for rank in range(5)
]

INIT_DECK = [
    3, 2, 2, 2, 1,
    3, 2, 2, 2, 1,
    3, 2, 2, 2, 1,
    3, 2, 2, 2, 1,
    3, 2, 2, 2, 1
]

cdef class MCTS_Sampler:
    """Sampler for re-determinization in MCTS"""
    cdef HanabiDeck deck

    def __init__(self):
        self.deck = HanabiDeck()

    def sample_hand(
        self,
        int player,
        int original_hand_size,
        list player_hands,
        list discard_pile,
        list fireworks,
        card_knowledge=None,
        additional_cards=None,
    ):
        if additional_cards is None:
            additional_cards = []

        sampled_hand = []

        self.deck.reset_deck()
        self.deck.remove_by_cards(discard_pile)
        self.deck.remove_by_hands(player, player_hands, -1)
        self.deck.remove_by_fireworks(fireworks)
        self.deck.remove_by_cards(additional_cards)

        while len(sampled_hand) < original_hand_size:
            sampled_hand = []
            
            for card_idx in range(original_hand_size):
                deck_card = HanabiDeck(self.deck.card_count, self.deck.total_count)
                deck_card.remove_by_cards(sampled_hand)
                deck_card.remove_by_own_hand(player, player_hands, card_idx)

                if card_knowledge:
                    deck_card.remove_by_knowledge(card_knowledge[card_idx])

                sampled_cards = deck_card.get_deck()
                
                if sampled_cards:
                    sampled_hand.append(random.choice(sampled_cards))
                else:
                    break

        return sampled_hand

    def sample_card(
        self,
        int player,
        int card_index,
        list player_hands,
        list discard_pile,
        list fireworks,
        card_knowledge=None,
        additional_cards=None,
    ):
        if additional_cards is None:
            additional_cards = []

        valid_cards = self.valid_cards(
            player,
            card_index,
            player_hands,
            discard_pile,
            fireworks,
            card_knowledge,
            additional_cards,
        )
        if valid_cards:
            return random.choice(valid_cards)
        else:
            return None

    def valid_cards(
        self,
        int player,
        int card_index,
        list player_hands,
        list discard_pile,
        list fireworks,
        card_knowledge=None,
        additional_cards=None,
    ):
        if additional_cards is None:
            additional_cards = []

        self.deck.reset_deck()
        self.deck.remove_by_cards(discard_pile)
        self.deck.remove_by_hands(player, player_hands, card_index)
        self.deck.remove_by_fireworks(fireworks)
        self.deck.remove_by_cards(additional_cards)

        if card_knowledge is not None:
            self.deck.remove_by_knowledge(card_knowledge[card_index])

        return self.deck.get_deck()

cdef class HanabiDeck:
    """Deck of Hanabi cards for sampling hands and cards"""
    cdef int num_ranks, num_colors, total_count
    cdef object card_count

    def __init__(self, card_count=None, total_count=None):
        self.num_ranks = 5
        self.num_colors = 5
        
        if card_count is not None and total_count is not None:
            self.card_count = card_count.copy()
            self.total_count = total_count
        else:
            self.reset_deck()

    def get_deck(self):
        valid_card_indices = np.nonzero(self.card_count)[0]
        deck_list = np.repeat(valid_card_indices, self.card_count[valid_card_indices])
        return [PRECOMPUTED_CARDS[idx] for idx in deck_list]

    def remove_by_knowledge(self, card_knowledge):
        cdef int color, rank
        for color in range(self.num_colors):
            color_plausible = card_knowledge.color_plausible(color)
            for rank in range(self.num_ranks):
                if not (color_plausible and card_knowledge.rank_plausible(rank)):
                    card_idx = color * self.num_ranks + rank
                    self.total_count -= self.card_count[card_idx]
                    self.card_count[card_idx] = 0

    def remove_by_cards(self, list cards):
        for card in cards:
            self.remove_card(card.color(), card.rank())

    def remove_by_hands(self, int player, list hands, int card_index=-1):
        cdef int other_player, idx
        for other_player in range(len(hands)):
            if other_player == player and card_index == -1:
                continue
            for idx, card in enumerate(hands[other_player]):
                if other_player == player and idx == card_index:
                    continue
                self.remove_card(card.color(), card.rank())

    def remove_by_own_hand(self, int player, list hands, int card_index):
        cdef int idx
        for idx, card in enumerate(hands[player]):
            if idx == card_index:
                continue
            self.remove_card(card.color(), card.rank())

    def remove_by_fireworks(self, list fireworks):
        cdef int color, firework, idx, start_idx, end_idx
        for color, firework in enumerate(fireworks):
            if firework > 0:
                start_idx = color * self.num_ranks
                end_idx = start_idx + firework
                for idx in range(start_idx, end_idx):
                    if self.card_count[idx] > 0:
                        self.card_count[idx] -= 1
                        self.total_count -= 1

    def reset_deck(self):
        self.card_count = np.array(INIT_DECK, dtype=np.int32)
        self.total_count = 50

    def remove_card(self, int color, int rank):
        cdef int card_idx = color * self.num_ranks + rank
        if self.card_count[card_idx] == 0:
            return
        self.card_count[card_idx] -= 1
        self.total_count -= 1

    def is_empty(self):
        return self.total_count == 0

    def __str__(self):
        output = []
        output.append(f"\n{'=' * 30}DECK{'=' * 30}\n")
        color_names = ["R", "Y", "G", "W", "B"]

        header = "          " + "     ".join(
            f"Rank {rank+1}" for rank in range(self.num_ranks)
        )
        output.append(header)

        max_counts = [3, 2, 2, 2, 1]
        max_counts_line = "Max Qty:  " + "".join(
            f"   {count}       " for count in max_counts
        )
        output.append(max_counts_line)
        output.append("          " + "-" * (10 * self.num_ranks))

        for color in range(self.num_colors):
            color_char = color_names[color]
            row = f"{color_char}       |"
            for rank in range(self.num_ranks):
                index = color * self.num_ranks + rank
                count = self.card_count[index]
                row += f"   {count:2d}      "
            output.append(row)

        output.append(f"\nTotal cards remaining in deck: {self.total_count}")
        output.append(f"{'=' * 64}\n")

        return "\n".join(output)
