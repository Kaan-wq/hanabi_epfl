from pyhanabi import HanabiCard
from libc.stdlib cimport rand
from libc.string cimport memcpy
from cpython.list cimport PyList_GET_ITEM, PyList_Append, PyList_GET_SIZE
cdef struct CythonCard:
    int color
    int rank

cdef inline CythonCard make_card(int color, int rank) noexcept nogil:
    cdef CythonCard card
    card.color = color
    card.rank = rank
    return card

cdef int[25][2] PRECOMPUTED_CARDS = [
    [0, 0], [0, 1], [0, 2], [0, 3], [0, 4],
    [1, 0], [1, 1], [1, 2], [1, 3], [1, 4],
    [2, 0], [2, 1], [2, 2], [2, 3], [2, 4],
    [3, 0], [3, 1], [3, 2], [3, 3], [3, 4],
    [4, 0], [4, 1], [4, 2], [4, 3], [4, 4],
]

cdef int[25] INIT_DECK = [
    3, 2, 2, 2, 1,
    3, 2, 2, 2, 1,
    3, 2, 2, 2, 1,
    3, 2, 2, 2, 1,
    3, 2, 2, 2, 1
]

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
        cdef int card_idx, deck_size
        cdef CythonCard[50] deck_buffer

        self.deck.reset_deck()
        self.deck.remove_by_cards(discard_pile)
        self.deck.remove_by_hands(player, player_hands, -1)
        self.deck.remove_by_fireworks(fireworks)
        self.deck.remove_by_cards(additional_cards)

        while PyList_GET_SIZE(sampled_hand) < original_hand_size:
            sampled_hand.clear()

            for card_idx in range(original_hand_size):
                deck_card = HanabiDeck(self.deck)
                deck_card.remove_by_cython_cards(sampled_hand)
                deck_card.remove_by_own_hand(player, player_hands, card_idx)

                if card_knowledge:
                    deck_card.remove_by_knowledge(card_knowledge[card_idx])

                deck_size = deck_card.get_deck_into_buffer(deck_buffer)

                if deck_size > 0:
                    PyList_Append(sampled_hand, deck_buffer[rand() % deck_size])
                else:
                    break

        cdef CythonCard card
        for card_idx in range(PyList_GET_SIZE(sampled_hand)):
            card = <CythonCard><object>PyList_GET_ITEM(sampled_hand, card_idx)
            # PyList_SET_ITEM(sampled_hand, card_idx, HanabiCard(card.color, card.rank))
            sampled_hand[card_idx] = HanabiCard(card.color, card.rank)

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
        cdef CythonCard sampled_card
        if valid_cards:
            sampled_card = <CythonCard><object>PyList_GET_ITEM(valid_cards, rand() % PyList_GET_SIZE(valid_cards))
            return HanabiCard(sampled_card.color, sampled_card.rank)
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
    cdef int num_ranks, num_colors
    cdef int total_count
    cdef int[25] card_count

    def __cinit__(self, HanabiDeck source_deck = None):
        self.num_ranks = 5
        self.num_colors = 5

        if source_deck is not None:
            memcpy(&self.card_count[0], &source_deck.card_count[0], 25 * sizeof(int))
            self.total_count = source_deck.total_count
        else:
            self.reset_deck()

    cdef inline int get_deck_into_buffer(self, CythonCard* buffer) noexcept nogil:
        cdef int i, j, count, pos = 0
        cdef int card_color, card_rank
        cdef int max_idx = self.num_colors * self.num_ranks
        for i in range(max_idx):
            count = self.card_count[i]
            if count > 0:
                card_color = PRECOMPUTED_CARDS[i][0]
                card_rank = PRECOMPUTED_CARDS[i][1]
                for j in range(count):
                    buffer[pos].color = card_color
                    buffer[pos].rank = card_rank
                    pos += 1
        return pos

    cdef inline list get_deck(self):
        cdef int total = self.total_count
        # cdef list deck_list = PyList_New(total)
        cdef list deck_list = [None] * total
        cdef int i, j, count, pos = 0
        cdef int max_idx = self.num_colors * self.num_ranks
        cdef int card_color, card_rank
        cdef CythonCard card
        for i in range(max_idx):
            count = self.card_count[i]
            if count > 0:
                card_color = PRECOMPUTED_CARDS[i][0]
                card_rank = PRECOMPUTED_CARDS[i][1]
                for j in range(count):
                    card = make_card(card_color, card_rank)
                    # PyList_SET_ITEM(deck_list, pos, card)
                    deck_list[pos] = card
                    pos += 1
        return deck_list

    cdef inline list get_hanabi_deck(self):
        cdef int total = self.total_count
        # cdef list deck_list = PyList_New(total)
        cdef list deck_list = [None] * total
        cdef int i, j, count, pos = 0
        cdef int max_idx = self.num_colors * self.num_ranks
        cdef object card
        for i in range(max_idx):
            count = self.card_count[i]
            if count > 0:
                card = PRECOMPUTED_HANABI_CARDS[i]
                for j in range(count):
                    # PyList_SET_ITEM(deck_list, pos, card)
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
        cdef int c_len = PyList_GET_SIZE(cards)
        cdef int i
        cdef object card
        for i in range(c_len):
            card = <object>PyList_GET_ITEM(cards, i)
            self.remove_card(card.color(), card.rank())

    cdef void remove_by_cython_cards(self, list cards):
        cdef int c_len = PyList_GET_SIZE(cards)
        cdef int i
        cdef CythonCard card
        for i in range(c_len):
            card = <CythonCard><object>PyList_GET_ITEM(cards, i)
            self.remove_card(card.color, card.rank)

    cdef void remove_by_hands(self, int player, list hands, int card_index=-1):
        cdef int num_players = PyList_GET_SIZE(hands)
        cdef int other_player, hand_size, idx
        cdef object card, hand
        for other_player in range(num_players):
            if other_player == player and card_index == -1:
                continue
            hand = <object>PyList_GET_ITEM(hands, other_player)
            hand_size = PyList_GET_SIZE(hand)
            for idx in range(hand_size):
                if other_player == player and idx == card_index:
                    continue
                card = <object>PyList_GET_ITEM(hand, idx)
                self.remove_card(card.color(), card.rank())

    cdef void remove_by_own_hand(self, int player, list hands, int card_index):
        cdef object hand = <object>PyList_GET_ITEM(hands, player)
        cdef int hand_size = PyList_GET_SIZE(hand)
        cdef int idx
        cdef object card
        for idx in range(hand_size):
            if idx == card_index:
                continue
            card = <object>PyList_GET_ITEM(hand, idx)
            self.remove_card(card.color(), card.rank())

    cdef void remove_by_fireworks(self, list fireworks):
        cdef int fireworks_len = PyList_GET_SIZE(fireworks)
        cdef int color, firework, idx, start_idx, end_idx
        for color in range(fireworks_len):
            firework = <int>(<object>PyList_GET_ITEM(fireworks, color))
            if firework > 0:
                start_idx = color * self.num_ranks
                end_idx = start_idx + firework
                for idx in range(start_idx, end_idx):
                    if self.card_count[idx] > 0:
                        self.card_count[idx] -= 1
                        self.total_count -= 1

    cdef inline void reset_deck(self) noexcept nogil:
        memcpy(&self.card_count[0], &INIT_DECK[0], 25 * sizeof(int))
        self.total_count = 50

    cdef inline void remove_card(self, int color, int rank) noexcept nogil:
        cdef int card_idx = color * self.num_ranks + rank
        if self.card_count[card_idx] == 0:
            return
        self.card_count[card_idx] -= 1
        self.total_count -= 1

