from libcpp.string cimport string
from libcpp.vector cimport vector
from libcpp.unordered_map cimport unordered_map
from libcpp.memory cimport unique_ptr, shared_ptr
from libcpp.pair cimport pair
from libc.stdint cimport int8_t, uint8_t
from cython.operator cimport dereference as deref
import numpy as np
cimport numpy as np


# Utility functions
cdef dict COLOR_CHAR = {
    0: 'R',
    1: 'Y',
    2: 'G',
    3: 'W',
    4: 'B'
}

CHANCE_PLAYER_ID = -1

def color_idx_to_char(int color_idx):
    if color_idx == -1:
        return None
    return COLOR_CHAR[color_idx]

def color_char_to_idx(str color_char):
    for idx, char in COLOR_CHAR.items():
        if char == color_char:
            return idx
    raise ValueError(f"Invalid color: {color_char}. Should be one of {list(COLOR_CHAR.values())}")

# Module initialization
from enum import IntEnum

class PyHanabiMoveType(IntEnum):
    INVALID = 0
    PLAY = 1
    DISCARD = 2
    REVEAL_COLOR = 3
    REVEAL_RANK = 4
    DEAL = 5
    RETURN = 6
    DEAL_SPECIFIC = 7

class PyAgentObservationType(IntEnum):
    MINIMAL = 0
    CARD_KNOWLEDGE = 1
    SEER = 2

class PyEndOfGameType(IntEnum):
    NOT_FINISHED = 0
    OUT_OF_LIFE_TOKENS = 1
    OUT_OF_CARDS = 2
    COMPLETED_FIREWORKS = 3

class ObservationEncoderType(IntEnum):
    CANONICAL = 0
    

# Python wrapper for HanabiCard
cdef class PyHanabiCard:
    cdef HanabiCard c_card

    def __cinit__(self, int color=-1, int rank=-1):
        self.c_card = HanabiCard(color, rank)

    @property
    def color(self):
        return self.c_card.Color()

    @property
    def rank(self):
        return self.c_card.Rank()

    def __str__(self):
        if self.valid():
            return COLOR_CHAR[self.color] + str(self.rank + 1)
        return "XX"

    def __repr__(self):
        return self.__str__()

    def __eq__(self, PyHanabiCard other):
        return self.color == other.color and self.rank == other.rank

    def valid(self):
        return self.c_card.IsValid()

    def to_dict(self):
        return {
            "color": color_idx_to_char(self.color),
            "rank": self.rank
        }


# Python wrapper for CardKnowledge
cdef class PyCardKnowledge:
    cdef HanabiHand.CardKnowledge* c_knowledge

    def __cinit__(self):
        pass  # Will be initialized externally

    def __init__(self):
        raise TypeError("CardKnowledge cannot be instantiated directly")

    def color(self):
        if self.c_knowledge.ColorHinted():
            return self.c_knowledge.Color()
        return None

    def color_plausible(self, int color):
        return self.c_knowledge.ColorPlausible(color)

    def rank(self):
        if self.c_knowledge.RankHinted():
            return self.c_knowledge.Rank()
        return None

    def rank_plausible(self, int rank):
        return self.c_knowledge.RankPlausible(rank)

    def __str__(self):
        return self.c_knowledge.ToString().decode('utf-8')

    def __repr__(self):
        return self.__str__()

    def to_dict(self):
        return {
            "color": color_idx_to_char(self.color()),
            "rank": self.rank()
        }


# Python wrapper for HanabiMove
cdef class PyHanabiMove:
    cdef HanabiMove* c_move

    def __cinit__(self, int move_type, int8_t card_index=-1,
                  int8_t target_offset=-1, int8_t color=-1, int8_t rank=-1):
        # Keep the conversion from Python enum values to C++ enum values
        cdef HanabiMove.Type cpp_move_type = <HanabiMove.Type>move_type
        self.c_move = new HanabiMove(cpp_move_type, card_index, target_offset, color, rank)

    def __dealloc__(self):
        if self.c_move != NULL:
            del self.c_move
    
    @property
    def type(self):
        return self.c_move.MoveType()
    
    @property
    def card_index(self):
        return self.c_move.CardIndex()
    
    @property
    def target_offset(self):
        return self.c_move.TargetOffset()
    
    @property
    def color(self):
        return self.c_move.Color()
    
    @property
    def rank(self):
        return self.c_move.Rank()
    
    def __str__(self):
        return self.c_move.ToString().decode('utf-8')
    
    def __repr__(self):
        return self.__str__()
    
    def __eq__(self, PyHanabiMove other):
        return deref(self.c_move) == deref(other.c_move)
    
    def __hash__(self):
        return hash((self.type, self.card_index, self.target_offset,
                    self.color, self.rank))

    def to_dict(self):
        move_dict = {"action_type": PyHanabiMoveType(self.type).name}
        
        if self.type in [PyHanabiMoveType.PLAY, PyHanabiMoveType.DISCARD]:
            move_dict["card_index"] = self.card_index
        elif self.type == PyHanabiMoveType.REVEAL_COLOR:
            move_dict["target_offset"] = self.target_offset
            move_dict["color"] = color_idx_to_char(self.color)
        elif self.type == PyHanabiMoveType.REVEAL_RANK:
            move_dict["target_offset"] = self.target_offset
            move_dict["rank"] = self.rank
        elif self.type == PyHanabiMoveType.DEAL:
            move_dict["color"] = color_idx_to_char(self.color)
            move_dict["rank"] = self.rank
        elif self.type == PyHanabiMoveType.DEAL_SPECIFIC:
            move_dict["color"] = color_idx_to_char(self.color)
            move_dict["rank"] = self.rank
        elif self.type == PyHanabiMoveType.RETURN:
            move_dict["card_index"] = self.card_index
            
        return move_dict

    @staticmethod
    def get_discard_move(int card_index):
        return PyHanabiMove(PyHanabiMoveType.DISCARD, card_index=card_index)

    @staticmethod
    def get_play_move(int card_index):
        return PyHanabiMove(PyHanabiMoveType.PLAY, card_index=card_index)

    @staticmethod
    def get_reveal_color_move(int target_offset, int color):
        return PyHanabiMove(PyHanabiMoveType.REVEAL_COLOR, 
                           target_offset=target_offset, color=color)

    @staticmethod
    def get_reveal_rank_move(int target_offset, int rank):
        return PyHanabiMove(PyHanabiMoveType.REVEAL_RANK, 
                           target_offset=target_offset, rank=rank)

    @staticmethod
    def get_return_move(int card_index, int player):
        return PyHanabiMove(PyHanabiMoveType.RETURN, card_index=card_index)

    @staticmethod
    def get_deal_specific_move(int card_index, int player, int color, int rank):
        return PyHanabiMove(PyHanabiMoveType.DEAL_SPECIFIC, card_index=card_index,
                           target_offset=player, color=color, rank=rank)


# Python wrapper for HanabiHistoryItem
cdef class PyHanabiHistoryItem:
    cdef HanabiHistoryItem* c_item

    def __cinit__(self, PyHanabiMove move):
        self.c_item = HanabiHistoryItem(move.c_move)

    def __dealloc__(self):
        if self.c_item != NULL:
            del self.c_item

    @property
    def move(self):
        cdef PyHanabiMove py_move = PyHanabiMove(HanabiMoveType.kInvalid)
        py_move.c_move = self.c_item.move
        return py_move

    @property
    def player(self):
        return self.c_item.player

    @property
    def scored(self):
        return self.c_item.scored

    @property
    def information_token(self):
        return self.c_item.information_token

    @property
    def color(self):
        return self.c_item.color

    @property
    def rank(self):
        return self.c_item.rank

    def card_info_revealed(self):
        revealed = []
        bitmask = self.c_item.reveal_bitmask
        for i in range(8):
            if bitmask & (1 << i):
                revealed.append(i)
        return revealed

    def card_info_newly_revealed(self):
        revealed = []
        bitmask = self.c_item.newly_revealed_bitmask
        for i in range(8):
            if bitmask & (1 << i):
                revealed.append(i)
        return revealed

    @property
    def deal_to_player(self):
        return self.c_item.deal_to_player

    def __str__(self):
        return self.c_item.ToString().decode('utf-8')

    def __repr__(self):
        return self.__str__()


# Python wrapper for HanabiState
cdef class PyHanabiState:
    cdef HanabiState* c_state
    cdef PyHanabiGame game
    cdef int _num_players
    cdef int _max_hand_size

    def __cinit__(self, PyHanabiGame game, int start_player=-1):
        self.game = game
        self.c_state = new HanabiState(game.c_game, start_player)
        self._num_players = self.num_players()
        self._max_hand_size = 5

    def __dealloc__(self):
        if self.c_state != NULL:
            del self.c_state

    def copy(self):
        cdef PyHanabiState new_state = PyHanabiState(self.game)
        del new_state.c_state
        new_state.c_state = new HanabiState(deref(self.c_state))
        return new_state

    def observation(self, int player):
        return PyHanabiObservation(self, player)

    def apply_move(self, PyHanabiMove move):
        self.c_state.ApplyMove(move.c_move)

    def turns_to_play(self):
        return self.c_state.TurnsToPlay()

    def cur_player(self):
        return self.c_state.CurPlayer()

    def deck_size(self):
        return self.c_state.Deck().Size()

    def discard_pile(self):
        cdef vector[HanabiCard] c_discards = self.c_state.DiscardPile()
        return [PyHanabiCard(c.Color(), c.Rank()) for c in c_discards]

    def fireworks(self):
        return [level for level in self.c_state.Fireworks()]

    def progress(self):
        return sum(self.fireworks())

    def score(self):
        if self.life_tokens() == 0:
            return 0
        return self.progress()

    def deal_random_card(self):
        self.c_state.ApplyRandomChance()

    def deal_specific_card(self, int color, int rank, int card_index):
        assert self.cur_player() == CHANCE_PLAYER_ID
        move = PyHanabiMove.get_deal_specific_move(color, rank, card_index)
        self.apply_move(move)

    def remove_knowledge(self, int player, int card_index):
        self.c_state.RemoveKnowledge(player, card_index)

    def player_hands(self):
        hands = []
        for pid in range(self._num_players):
            player_hand = []
            hand_size = self.c_state.Hands()[pid].Cards().size()
            for i in range(hand_size):
                c_card = self.c_state.Hands()[pid].Cards()[i]
                player_hand.append(PyHanabiCard(c_card.Color(), c_card.Rank()))
            hands.append(player_hand)
        return hands

    def information_tokens(self):
        return self.c_state.InformationTokens()

    def end_of_game_status(self):
        return EndOfGameType(self.c_state.EndOfGameStatus())

    def is_terminal(self):
        return self.c_state.IsTerminal()

    def legal_moves(self):
        if self.is_terminal():
            return []
        moves = self.c_state.LegalMoves(self.cur_player())
        return [PyHanabiMove(m.MoveType(), m.CardIndex(), m.TargetOffset(),
                            m.Color(), m.Rank()) for m in moves]

    def move_is_legal(self, PyHanabiMove move):
        return self.c_state.MoveIsLegal(move.c_move)

    def card_playable_on_fireworks(self, int color, int rank):
        return self.c_state.CardPlayableOnFireworks(color, rank)

    def life_tokens(self):
        return self.c_state.LifeTokens()

    def num_players(self):
        return self.c_state.NumPlayers()

    def move_history(self):
        history = []
        for item in self.c_state.MoveHistory():
            py_move = PyHanabiMove(item.move.MoveType(), item.move.CardIndex(),
                                 item.move.TargetOffset(), item.move.Color(),
                                 item.move.Rank())
            py_item = PyHanabiHistoryItem(py_move)
            history.append(py_item)
        return history

    def __str__(self):
        return self.c_state.ToString().decode('utf-8')

    def __repr__(self):
        return self.__str__()


# Python wrapper for HanabiGame
cdef class PyHanabiGame:
    cdef HanabiGame* c_game

    def __cinit__(self, dict params=None):
        if params is None:
            params = {}
        cdef unordered_map[string, string] c_params
        for key, value in params.items():
            c_params[key.encode('utf-8')] = str(value).encode('utf-8')
        self.c_game = new HanabiGame(c_params)

    def __dealloc__(self):
        if self.c_game != NULL:
            del self.c_game

    def new_initial_state(self):
        return PyHanabiState(self)

    def parameter_string(self):
        params = self.c_game.Parameters()
        return {k.decode('utf-8'): v.decode('utf-8') for k, v in params}

    def num_players(self):
        return self.c_game.NumPlayers()

    def num_colors(self):
        return self.c_game.NumColors()

    def num_ranks(self):
        return self.c_game.NumRanks()

    def hand_size(self):
        return self.c_game.HandSize()

    def max_information_tokens(self):
        return self.c_game.MaxInformationTokens()

    def max_life_tokens(self):
        return self.c_game.MaxLifeTokens()

    def observation_type(self):
        return AgentObservationType(self.c_game.ObservationType())

    def max_moves(self):
        return self.c_game.MaxMoves()

    def num_cards(self, int color, int rank):
        return self.c_game.NumberCardInstances(color, rank)

    def get_move_uid(self, PyHanabiMove move):
        return self.c_game.GetMoveUid(move.c_move)

    def get_move(self, int move_uid):
        c_move = self.c_game.GetMove(move_uid)
        return PyHanabiMove(c_move.MoveType(), c_move.CardIndex(),
                          c_move.TargetOffset(), c_move.Color(), c_move.Rank())


cdef class PyHanabiObservation:
    cdef HanabiObservation* c_obs
    cdef PyHanabiGame game
    cdef int _max_hand_size
    
    def __cinit__(self, PyHanabiState state, int player):
        self.game = state.game
        self.c_obs = new HanabiObservation(deref(state.c_state), player)
        self._max_hand_size = 5

    def __dealloc__(self):
        if self.c_obs != NULL:
            del self.c_obs

    def __str__(self):
        return self.c_obs.ToString().decode('utf-8')

    def __repr__(self):
        return self.__str__()

    def cur_player_offset(self):
        return self.c_obs.CurPlayerOffset()

    def num_players(self):
        return self.c_obs.NumPlayers()

    def observed_hands(self):
        hands = []
        for pid in range(self.num_players()):
            player_hand = []
            hand_size = self.c_obs.Hands()[pid].Cards().size()
            for i in range(hand_size):
                c_card = self.c_obs.Hands()[pid].Cards()[i]
                player_hand.append(PyHanabiCard(c_card.Color(), c_card.Rank()))
            hands.append(player_hand)
        return hands

    def card_knowledge(self):
        cdef vector[HanabiHand] hands = self.c_obs.Hands()
        knowledge = []
        for pid in range(self.num_players()):
            player_knowledge = []
            hand_size = hands[pid].Cards().size()
            for i in range(hand_size):
                card_knowledge = PyCardKnowledge()
                card_knowledge.c_knowledge = &hands[pid].Knowledge()[i]
                player_knowledge.append(card_knowledge)
            knowledge.append(player_knowledge)
        return knowledge

    def discard_pile(self):
        cdef vector[HanabiCard] c_discards = self.c_obs.DiscardPile()
        return [PyHanabiCard(c.Color(), c.Rank()) for c in c_discards]

    def fireworks(self):
        return [level for level in self.c_obs.Fireworks()]

    def deck_size(self):
        return self.c_obs.DeckSize()

    def last_moves(self):
        history_items = []
        for item in self.c_obs.LastMoves():
            py_move = PyHanabiMove(item.move.MoveType(), item.move.CardIndex(),
                                 item.move.TargetOffset(), item.move.Color(),
                                 item.move.Rank())
            history_items.append(PyHanabiHistoryItem(py_move))
        return history_items

    def information_tokens(self):
        return self.c_obs.InformationTokens()

    def life_tokens(self):
        return self.c_obs.LifeTokens()

    def legal_moves(self):
        if self.cur_player_offset() != 0:
            return []
        moves = self.c_obs.LegalMoves()
        return [PyHanabiMove(m.MoveType(), m.CardIndex(), m.TargetOffset(),
                            m.Color(), m.Rank()) for m in moves]

    def card_playable_on_fireworks(self, int color, int rank):
        return self.c_obs.CardPlayableOnFireworks(color, rank)

# ObservationEncoder wrapper
cdef class PyObservationEncoder:
    cdef ObservationEncoder* c_encoder
    cdef PyHanabiGame game

    def __cinit__(self, PyHanabiGame game, int type):
        self.game = game
        if type == ObservationEncoderType.kCanonical:
            self.c_encoder = new CanonicalObservationEncoder(game.c_game)
        else:
            raise ValueError("Unsupported encoder type")

    def __dealloc__(self):
        if self.c_encoder != NULL:
            del self.c_encoder

    def shape(self):
        return self.c_encoder.Shape()

    def encode(self, PyHanabiObservation observation):
        return self.c_encoder.Encode(deref(observation.c_obs))
