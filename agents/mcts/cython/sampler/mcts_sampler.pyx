import numpy as np
cimport numpy as cnp
from pyhanabi import HanabiCard
from libc.stdlib cimport rand


cdef cnp.int8_t[:, :] PRECOMPUTED_CARDS = np.array([
    [color, rank]
    for color in range(5)
    for rank in range(5)
], dtype=np.int8)

cdef cnp.int8_t[:] INIT_DECK = np.array([
    3, 2, 2, 2, 1,
    3, 2, 2, 2, 1,
    3, 2, 2, 2, 1,
    3, 2, 2, 2, 1,
    3, 2, 2, 2, 1
], dtype=np.int8)

PRECOMPUTED_HANABI_CARDS = [
    HanabiCard(color, rank)
    for color in range(5)
    for rank in range(5)
]

cdef class MCTS_Sampler:
    """Sampler for re-determinization in MCTS"""
    cdef HanabiDeck deck

    def __cinit__(self):
        self.deck = HanabiDeck()

    cpdef list sample_hand(
        self,
        int player,
        int original_hand_size,
        list player_hands,
        list discard_pile,
        list fireworks,
        card_knowledge=None,
        list additional_cards=None,
    ):
        if additional_cards is None:
            additional_cards = []

        cdef list sampled_hand = []
        cdef HanabiDeck deck_card
        cdef list sampled_cards
        cdef int card_idx

        self.deck.reset_deck()
        self.deck.remove_by_cards(discard_pile)
        self.deck.remove_by_hands(player, player_hands, -1)
        self.deck.remove_by_fireworks(fireworks)
        self.deck.remove_by_cards(additional_cards)

        while len(sampled_hand) < original_hand_size:
            sampled_hand.clear()

            for card_idx in range(original_hand_size):
                deck_card = HanabiDeck(self.deck.card_count, self.deck.total_count)
                deck_card.remove_by_cython_cards(sampled_hand)
                deck_card.remove_by_own_hand(player, player_hands, card_idx)

                if card_knowledge:
                    deck_card.remove_by_knowledge(card_knowledge[card_idx])

                sampled_cards = deck_card.get_deck()

                if sampled_cards:
                    sampled_hand.append(sampled_cards[rand() % len(sampled_cards)])
                else:
                    break

        cdef CythonCard card
        for card_idx in range(len(sampled_hand)):
            card = sampled_hand[card_idx]
            sampled_hand[card_idx] = HanabiCard(card.color(), card.rank())

        return sampled_hand

    cpdef object sample_card(
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

        cdef list valid_cards = self.valid_cards(
            player,
            card_index,
            player_hands,
            discard_pile,
            fireworks,
            card_knowledge,
            additional_cards,
            return_hanabi_card=False,
        )
        cdef object sampled_card
        if valid_cards:
            sampled_card = valid_cards[rand() % len(valid_cards)]
            return HanabiCard(sampled_card._color, sampled_card._rank)
        else:
            return None

    cpdef list valid_cards(
        self,
        int player,
        int card_index,
        list player_hands,
        list discard_pile,
        list fireworks,
        card_knowledge=None,
        additional_cards=None,
        bint return_hanabi_card=True,
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

        if return_hanabi_card:
            return self.deck.get_hanabi_deck()
        else:
            return self.deck.get_deck()


cdef class HanabiDeck:
    """Deck of Hanabi cards for sampling hands and cards"""
    cdef readonly int num_ranks, num_colors
    cdef int total_count
    cdef cnp.int8_t[:] card_count

    def __cinit__(self, cnp.int8_t[:] card_count=None, int total_count=0):
        self.num_ranks = 5
        self.num_colors = 5

        if card_count is not None and total_count != 0:
            self.card_count = card_count.copy()
            self.total_count = total_count
        else:
            self.reset_deck()

    cdef inline list get_deck(self):
        cdef int total = self.total_count
        cdef list deck_list = [None] * total
        cdef int i, j, count, pos = 0
        cdef int max_idx = self.num_colors * self.num_ranks
        cdef int card_color, card_rank
        for i in range(max_idx):
            count = self.card_count[i]
            if count > 0:
                card_color = PRECOMPUTED_CARDS[i, 0]
                card_rank = PRECOMPUTED_CARDS[i, 1]
                for j in range(count):
                    deck_list[pos] = CythonCard(card_color, card_rank)
                    pos += 1
        return deck_list

    cdef inline list get_hanabi_deck(self):
        cdef int total = self.total_count
        cdef list deck_list = [None] * total
        cdef int i, j, count, pos = 0
        cdef int max_idx = self.num_colors * self.num_ranks
        cdef object card
        for i in range(max_idx):
            count = self.card_count[i]
            if count > 0:
                card = PRECOMPUTED_HANABI_CARDS[i]
                for j in range(count):
                    deck_list[pos] = card
                    pos += 1
        return deck_list

    cdef void remove_by_knowledge(self, card_knowledge):
        cdef int color, rank, card_idx
        cdef bint color_plausible, rank_plausible
        for color in range(self.num_colors):
            color_plausible = card_knowledge.color_plausible(color)
            for rank in range(self.num_ranks):
                rank_plausible = card_knowledge.rank_plausible(rank)
                if not (color_plausible and rank_plausible):
                    card_idx = color * self.num_ranks + rank
                    self.total_count -= self.card_count[card_idx]
                    self.card_count[card_idx] = 0

    cdef void remove_by_cards(self, list cards):
        cdef int c_len = len(cards)
        cdef int i
        cdef object card
        for i in range(c_len):
            card = cards[i]
            self.remove_card(card.color(), card.rank())

    cdef void remove_by_cython_cards(self, list cards):
        cdef int c_len = len(cards)
        cdef int i
        cdef CythonCard card
        for i in range(c_len):
            card = cards[i]
            self.remove_card(card._color, card._rank)

    cdef void remove_by_hands(self, int player, list hands, int card_index=-1):
        cdef int num_players = len(hands)
        cdef int other_player, hand_size, idx
        cdef object card
        for other_player in range(num_players):
            if other_player == player and card_index == -1:
                continue
            hand_size = len(hands[other_player])
            for idx in range(hand_size):
                if other_player == player and idx == card_index:
                    continue
                card = hands[other_player][idx]
                self.remove_card(card.color(), card.rank())

    cdef void remove_by_own_hand(self, int player, list hands, int card_index):
        cdef int hand_size = len(hands[player])
        cdef int idx
        cdef object card
        for idx in range(hand_size):
            if idx == card_index:
                continue
            card = hands[player][idx]
            self.remove_card(card.color(), card.rank())

    cdef void remove_by_fireworks(self, list fireworks):
        cdef int fireworks_len = len(fireworks)
        cdef int color, firework, idx, start_idx, end_idx
        for color in range(fireworks_len):
            firework = <int>fireworks[color]
            if firework > 0:
                start_idx = color * self.num_ranks
                end_idx = start_idx + firework
                for idx in range(start_idx, end_idx):
                    if self.card_count[idx] > 0:
                        self.card_count[idx] -= 1
                        self.total_count -= 1

    cdef inline void reset_deck(self):
        self.card_count = INIT_DECK.copy()
        self.total_count = 50

    cdef inline void remove_card(self, int color, int rank):
        cdef int card_idx = color * self.num_ranks + rank
        if self.card_count[card_idx] == 0:
            return
        self.card_count[card_idx] -= 1
        self.total_count -= 1


cdef class CythonCard:
    """Hanabi card, with a color and a rank."""
    cdef readonly int _color
    cdef readonly int _rank

    def __cinit__(self, int color, int rank):
        self._color = color
        self._rank = rank

    cdef inline int color(self):
        return self._color

    cdef inline int rank(self):
        return self._rank
