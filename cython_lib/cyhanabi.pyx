import json
import enum
from cpython.bool cimport bool
from libc.stdlib cimport malloc, free
from libc.string cimport memcpy, strdup
from cyhanabi cimport DeleteString
from cyhanabi cimport pyhanabi_card_t, CardValid
from cyhanabi cimport pyhanabi_card_knowledge_t, CardKnowledgeToString, ColorWasHinted, KnownColor, ColorIsPlausible, RankWasHinted, KnownRank, RankIsPlausible
from cyhanabi cimport pyhanabi_move_t, DeleteMoveList, NumMoves, GetMove, DeleteMove, MoveToString, MoveType, CardIndex, TargetOffset, MoveColor, MoveRank, GetDiscardMove, GetReturnMove, GetPlayMove, GetRevealColorMove, GetRevealRankMove, GetDealSpecificMove
from cyhanabi cimport pyhanabi_history_item_t, DeleteHistoryItem, HistoryItemToString, HistoryItemMove, HistoryItemPlayer, HistoryItemScored, HistoryItemInformationToken, HistoryItemColor, HistoryItemRank, HistoryItemRevealBitmask, HistoryItemNewlyRevealedBitmask, HistoryItemDealToPlayer
from cyhanabi cimport pyhanabi_state_t, NewState, CopyState, DeleteState, StateParentGame, StateApplyMove, StateRemoveKnowledge, StateCurPlayer, StateDealCard, StateDeckSize, StateFireworks, StateDiscardPileSize, StateGetDiscard, StateGetHandSize, StateGetHandCard, StateEndOfGameStatus, StateInformationTokens, StateLegalMoves, StateLifeTokens, StateNumPlayers, StateScore, StateTurnsToPlay, StateToString, MoveIsLegal, CardPlayableOnFireworks, StateLenMoveHistory, StateGetMoveHistory
from cyhanabi cimport pyhanabi_game_t, DeleteGame, NewDefaultGame, NewGame, GameParamString, NumPlayers, NumColors, NumRanks, HandSize, MaxInformationTokens, MaxLifeTokens, ObservationType, NumCards, GetMoveUid, GetMoveByUid, MaxMoves
from cyhanabi cimport pyhanabi_observation_t, NewObservation, DeleteObservation, ObsToString, ObsCurPlayerOffset, ObsNumPlayers, ObsGetHandSize, ObsGetHandCard, ObsGetHandCardKnowledge, ObsDiscardPileSize, ObsGetDiscard, ObsFireworks, ObsDeckSize, ObsNumLastMoves, ObsGetLastMove, ObsInformationTokens, ObsLifeTokens, ObsNumLegalMoves, ObsGetLegalMove, ObsCardPlayableOnFireworks
from cyhanabi cimport pyhanabi_observation_encoder_t, NewObservationEncoder, DeleteObservationEncoder, ObservationShape, EncodeObservation
from cyhanabi cimport MoveToJson, MoveFromJson, GameToJson, GameFromJson, HistoryItemFromJson, HistoryItemToJson, StateToJson, StateFromJson

cdef char[5] COLOR_CHAR = [b"R", b"Y", b"G", b"W", b"B"]
cdef int CHANCE_PLAYER_ID = -1


def color_idx_to_char(color_idx):
    """Helper function for converting color index to a character.

    Args:
        color_idx: int, index into color char vector.

    Returns:
        color_char: str, A single character representing a color.

    Raises:
        AssertionError: If index is not in range.
    """
    assert isinstance(color_idx, int)
    if color_idx == -1:
        return None
    else:
        return COLOR_CHAR[color_idx]


def color_char_to_idx(color_char):
    """Helper function for converting color character to index.

    Args:
        color_char: str, Character representing a color.

    Returns:
        color_idx: int, Index into a color array \in [0, num_colors -1]

    Raises:
        ValueError: If color_char is not a valid color.
    """
    assert isinstance(color_char, str)
    try:
        return next(idx for (idx, c) in enumerate(COLOR_CHAR) if c == color_char)
    except StopIteration:
        raise ValueError("Invalid color: {}. Should be one of {}.".format(color_char, COLOR_CHAR))


class HanabiMoveType(enum.IntEnum):
    """Move types, consistent with hanabi_lib/hanabi_move.h."""
    INVALID = 0
    PLAY = 1
    DISCARD = 2
    REVEAL_COLOR = 3
    REVEAL_RANK = 4
    DEAL = 5
    RETURN = 6
    DEAL_SPECIFIC = 7


class AgentObservationType(enum.IntEnum):
    MINIMAL = 0
    CARD_KNOWLEDGE = 1
    SEER = 2


class HanabiEndOfGameType(enum.IntEnum):
    NOT_FINISHED = 0
    OUT_OF_LIFE_TOKENS = 1
    OUT_OF_CARDS = 2
    COMPLETED_FIREWORKS = 3


class HanabiCard(object):
  """Hanabi card, with a color and a rank.

  Python implementation of C++ HanabiCard class.
  """

  def __init__(self, color, rank):
    """A simple HanabiCard object.

    Args:
      color: an integer, starting at 0. Colors are in this order RYGWB.
      rank: an integer, starting at 0 (representing a 1 card). In the standard
          game, the largest value is 4 (representing a 5 card).
    """
    self._color = color
    self._rank = rank

  def color(self):
    return self._color

  def rank(self):
    return self._rank

  def __str__(self):
    if self.valid():
      return COLOR_CHAR[self._color] + str(self._rank + 1)
    else:
      return "XX"

  def __repr__(self):
    return self.__str__()

  def __eq__(self, other):
    return self._color == other.color() and self._rank == other.rank()

  def valid(self):
    return self._color >= 0 and self._rank >= 0

  def to_dict(self):
    """Serialize to dict.

    Returns:
      d: dict, containing color and rank of card.
    """
    return {"color": color_idx_to_char(self._color), "rank": self._rank}


cdef class HanabiCardKnowledge(object):
    """Cython wrapper for C++ HanabiCardKnowledge class."""
    cdef pyhanabi_card_knowledge_t* _knowledge

    def __cinit__(self):
        self._knowledge = NULL

    @staticmethod
    cdef from_ptr(pyhanabi_card_knowledge_t* knowledge):
        """Factory method to create from C++ pointer."""
        cdef HanabiCardKnowledge instance = HanabiCardKnowledge()
        instance._knowledge = knowledge
        return instance

    cdef color(self):
        if ColorWasHinted(self._knowledge):
            return KnownColor(self._knowledge)
        else:
            return None
    
    cdef color_plausible(self, color_index):
        return ColorIsPlausible(self._knowledge, color_index)

    cdef rank(self):
        if RankWasHinted(self._knowledge):
            return KnownRank(self._knowledge)
        else:
            return None

    cdef rank_plausible(self, rank_index):
        return RankIsPlausible(self._knowledge, rank_index)

    def __str__(self):
        c_string = CardKnowledgeToString(self._knowledge)
        string = c_string.decode("utf-8")
        DeleteString(c_string)
        return string

    def __repr__(self):
        return self.__str__()

    def to_dict(self):
        return {"color": color_idx_to_char(self.color()), "rank": self.rank()}


cdef class HanabiMove(object):
    cdef pyhanabi_move_t* _move
    cdef int _type
    cdef int _card_index
    cdef int _target_offset
    cdef int _color
    cdef int _rank

    def __cinit__(self):
        self._move = NULL

    @staticmethod
    cdef from_ptr(pyhanabi_move_t* move):
        cdef HanabiMove instance = HanabiMove()
        instance._move = move
        instance._type = MoveType(instance._move)
        instance._card_index = CardIndex(instance._move)
        instance._target_offset = TargetOffset(instance._move)
        instance._color = MoveColor(instance._move)
        instance._rank = MoveRank(instance._move)
        return instance
    
    def type(self):
        return HanabiMoveType(self._type)

    def card_index(self):
        return self._card_index

    def target_offset(self):
        return self._target_offset

    def color(self):
        return self._color

    def rank(self):
        return self._rank

    @staticmethod
    def get_discard_move(int card_index):
        cdef pyhanabi_move_t* c_move = <pyhanabi_move_t*> malloc(sizeof(pyhanabi_move_t))
        if not GetDiscardMove(card_index, c_move):
            free(c_move)
            return None
        return HanabiMove.from_ptr(c_move)

    @staticmethod
    def get_return_move(int card_index, int player):
        cdef pyhanabi_move_t* c_move = <pyhanabi_move_t*> malloc(sizeof(pyhanabi_move_t))
        if not GetReturnMove(card_index, player, c_move):
            free(c_move)
            return None
        return HanabiMove.from_ptr(c_move)

    @staticmethod
    def get_play_move(int card_index):
        cdef pyhanabi_move_t* c_move = <pyhanabi_move_t*> malloc(sizeof(pyhanabi_move_t))
        if not GetPlayMove(card_index, c_move):
            free(c_move)
            return None
        return HanabiMove.from_ptr(c_move)

    @staticmethod
    def get_reveal_color_move(int target_offset, int color):
        cdef pyhanabi_move_t* c_move = <pyhanabi_move_t*> malloc(sizeof(pyhanabi_move_t))
        if not GetRevealColorMove(target_offset, color, c_move):
            free(c_move)
            return None
        return HanabiMove.from_ptr(c_move)

    @staticmethod
    def get_reveal_rank_move(int target_offset, int rank):
        cdef pyhanabi_move_t* c_move = <pyhanabi_move_t*> malloc(sizeof(pyhanabi_move_t))
        if not GetRevealRankMove(target_offset, rank, c_move):
            free(c_move)
            return None
        return HanabiMove.from_ptr(c_move)

    @staticmethod
    def get_deal_specific_move(int card_index, int player, int color, int rank):
        cdef pyhanabi_move_t* c_move = <pyhanabi_move_t*> malloc(sizeof(pyhanabi_move_t))
        if not GetDealSpecificMove(card_index, player, color, rank, c_move):
            free(c_move)
            return None
        return HanabiMove.from_ptr(c_move)

    def to_json(self):
        """Serialize move to JSON."""
        cdef char* json_str = MoveToJson(self._move)
        if json_str == NULL:
            raise ValueError("Serialization failed: MoveToJSON returned NULL.")
        py_json = json_str.decode('utf-8')
        DeleteString(json_str)
        return py_json

    @classmethod
    def from_json(cls, str json_str):
        """Deserialize move from JSON."""
        cdef pyhanabi_move_t* c_move = <pyhanabi_move_t*>malloc(sizeof(pyhanabi_move_t))
        if not MoveFromJson(json_str.encode('ascii'), c_move):
            free(c_move)
            raise ValueError("Failed to deserialize HanabiMove from JSON")
        return HanabiMove.from_ptr(c_move)

    def __hash__(self):
        """Hash function for a move."""
        return hash((self._type, self._card_index, self._target_offset, self._color, self._rank))

    def __eq__(self, other):
        if not isinstance(other, HanabiMove):
            return NotImplemented
        return (self._type == other._type and
                self._card_index == other._card_index and
                self._target_offset == other._target_offset and
                self._color == other._color and
                self._rank == other._rank)

    def __str__(self):
        cdef char* c_string = MoveToString(self._move)
        if c_string == NULL:
            return ""
        string = c_string.decode('utf-8')
        DeleteString(c_string)
        return string

    def __repr__(self):
        return self.__str__()

    def __dealloc__(self):
        if self._move != NULL:
            DeleteMove(self._move)
            self._move = NULL

    def to_dict(self):
        move_type = self.type()
        move_dict = {}
        move_dict["action_type"] = move_type.name

        if move_type in (HanabiMoveType.PLAY, HanabiMoveType.DISCARD):
            move_dict["card_index"] = self.card_index()
        elif move_type == HanabiMoveType.REVEAL_COLOR:
            move_dict["target_offset"] = self.target_offset()
            move_dict["color"] = color_idx_to_char(self.color())
        elif move_type == HanabiMoveType.REVEAL_RANK:
            move_dict["target_offset"] = self.target_offset()
            move_dict["rank"] = self.rank()
        elif move_type == HanabiMoveType.DEAL:
            move_dict["color"] = color_idx_to_char(self.color())
            move_dict["rank"] = self.rank()
        elif move_type == HanabiMoveType.DEAL_SPECIFIC:
            move_dict["color"] = color_idx_to_char(self.color())
            move_dict["rank"] = self.rank()
        elif move_type == HanabiMoveType.RETURN:
            move_dict["card_index"] = self.card_index()
        else:
            raise ValueError(f"Unsupported move: {self}")
        return move_dict
                

cdef class HanabiHistoryItem(object):
    cdef pyhanabi_history_item_t* _item

    def __cinit__(self):
        self._item = NULL

    @staticmethod
    cdef from_ptr(pyhanabi_history_item_t* item):
        cdef HanabiHistoryItem instance = HanabiHistoryItem()
        instance._item = item
        return instance
    
    def move(self):
        cdef pyhanabi_move_t* c_move = <pyhanabi_move_t*> malloc(sizeof(pyhanabi_move_t))
        HistoryItemMove(self._item, c_move)
        return HanabiMove.from_ptr(c_move)

    def player(self):
        return HistoryItemPlayer(self._item)

    def scored(self):
        """Play move succeeded in placing card on fireworks."""
        return <bint>HistoryItemScored(self._item)

    def information_token(self):
        """Play/Discard move increased the number of information tokens."""
        return <bint>HistoryItemInformationToken(self._item)

    def color(self):
        """Color index of card that was Played/Discarded."""
        return HistoryItemColor(self._item)

    def rank(self):
        """Rank index of card that was Played/Discarded."""
        return HistoryItemRank(self._item)

    def card_info_revealed(self):
        cdef int bitmask = HistoryItemRevealBitmask(self._item)
        revealed = []
        for i in range(8):
            if bitmask & (1 << i):
                revealed.append(i)
        return revealed

    def card_info_newly_revealed(self):
        cdef int bitmask = HistoryItemNewlyRevealedBitmask(self._item)
        revealed = []
        for i in range(8):
            if bitmask & (1 << i):
                revealed.append(i)
        return revealed

    def deal_to_player(self):
        return HistoryItemDealToPlayer(self._item)
    
    def to_json(self):
        """Serialize history item to JSON."""
        cdef char* json_str = HistoryItemToJson(self._item)
        if json_str == NULL:
            raise ValueError("Serialization failed: HistoryItemToJson returned NULL.")
        py_json = json_str.decode('utf-8')
        DeleteString(json_str)
        return py_json
    
    @classmethod
    def from_json(cls, str json_str):
        """Deserialize history item from JSON."""
        cdef pyhanabi_history_item_t* c_item = <pyhanabi_history_item_t*>malloc(sizeof(pyhanabi_history_item_t))
        if not HistoryItemFromJson(json_str.encode('ascii'), c_item):
            free(c_item)
            raise ValueError("Failed to deserialize HanabiHistoryItem from JSON")
        return HanabiHistoryItem.from_ptr(c_item)

    def __str__(self):
        cdef char* c_string = HistoryItemToString(self._item)
        if c_string == NULL:
            return ""
        string = c_string.decode('utf-8')
        DeleteString(c_string)
        return string

    def __repr__(self):
        return self.__str__()

    def __dealloc__(self):
        if self._item != NULL:
            DeleteHistoryItem(self._item)
            self._item = NULL
        

cdef class HanabiState(object):
    cdef pyhanabi_state_t* _state
    cdef pyhanabi_game_t* _game
    cdef int _num_players
    cdef int _max_hand_size
    cdef list _card_pool
    cdef pyhanabi_card_t* _discard_pool

    def __cinit__(self):
        self._state = NULL
        self._game = NULL
        self._discard_pool = NULL
        self._card_pool = []
        self._num_players = 0
        self._max_hand_size = 5

    @staticmethod
    cdef HanabiState create(object game, pyhanabi_state_t* c_state, pyhanabi_game_t* c_game):
        """Factory method to create HanabiState."""
        # Instantiate state
        cdef HanabiState instance = HanabiState()
        instance._state = <pyhanabi_state_t*>malloc(sizeof(pyhanabi_state_t))
        if instance._state == NULL:
            raise MemoryError()

        # Initialize based on parameters
        if c_state == NULL and game is not None:
            instance._game = <pyhanabi_game_t*>game._game
            NewState(instance._game, instance._state)
        elif c_game != NULL:
            instance._game = c_game
            CopyState(c_state, instance._state)
        elif c_state != NULL:
            instance._game = <pyhanabi_game_t*>StateParentGame(c_state)
            CopyState(c_state, instance._state)
        else:
            raise ValueError("Invalid parameters")

        # Initialize other members
        instance._num_players = StateNumPlayers(instance._state)
        instance._max_hand_size = 5

        # Initialize card pool
        cdef pyhanabi_card_t* card
        for i in range(instance._num_players * instance._max_hand_size):
            card = <pyhanabi_card_t*>malloc(sizeof(pyhanabi_card_t))
            if card == NULL:
                raise MemoryError()
            instance._card_pool.append(<size_t>card)

        # Initialize discard pool
        instance._discard_pool = <pyhanabi_card_t*>malloc(sizeof(pyhanabi_card_t))
        if instance._discard_pool == NULL:
            raise MemoryError()

        return instance

    def __init__(self, game=None, state=None, parent_game=None):
        if state is None and game is not None:
            self = HanabiState.create(game, NULL, NULL)
        elif state is not None and parent_game is not None:
            self = HanabiState.create(None, <pyhanabi_state_t*>state, <pyhanabi_game_t*>parent_game)
        elif state is not None:
            self = HanabiState.create(None, <pyhanabi_state_t*>state, NULL)
        else:
            raise ValueError("Invalid parameters")
    
    def copy(self):
        return HanabiState.create(None, self._state, NULL)

    #def observation(self, int player):
    #    return HanabiObservation.create(self._state, self._game, player)

    def apply_move(self, HanabiMove move):
        StateApplyMove(self._state, move._move)

    def turns_to_play(self):
        return StateTurnsToPlay(self._state)

    def cur_player(self):
        return StateCurPlayer(self._state)

    def deck_size(self):
        return StateDeckSize(self._state)

    def discard_pile(self):
        discards = []
        cdef int pile_size = StateDiscardPileSize(self._state)
        for index in range(pile_size):
            StateGetDiscard(self._state, index, self._discard_pool)
            discards.append(HanabiCard(self._discard_pool.color, self._discard_pool.rank))
        return discards

    def fireworks(self):
        return [StateFireworks(self._state, c) for c in range(5)]

    def progress(self):
        return sum(self.fireworks())

    def score(self):
        if self.life_tokens() == 0:
            return 0
        return self.progress()

    def deal_random_card(self):
        StateDealCard(self._state)

    def deal_specific_card(self, color, rank, card_index):
        assert self.cur_player() == CHANCE_PLAYER_ID
        move = HanabiMove.get_deal_specific_move(color, rank, card_index)
        self.apply_move(move)

    def remove_knowledge(self, player, card_index):
        StateRemoveKnowledge(self._state, player, card_index)

    def player_hands(self):
        hand_list = []
        cdef int pid, hand_size, i
        cdef pyhanabi_card_t* c_card
        for pid in range(self._num_players):
            player_hand = []
            hand_size = StateGetHandSize(self._state, pid)
            for i in range(hand_size):
                c_card = <pyhanabi_card_t*><size_t>self._card_pool[pid * self._max_hand_size + i]
                StateGetHandCard(self._state, pid, i, c_card)
                player_hand.append(HanabiCard(c_card.color, c_card.rank))
            hand_list.append(player_hand)
        return hand_list

    def information_tokens(self):
        return StateInformationTokens(self._state)

    def end_of_game_status(self):
        return HanabiEndOfGameType(StateEndOfGameStatus(self._state))

    def is_terminal(self):
        return (StateEndOfGameStatus(self._state) != HanabiEndOfGameType.NOT_FINISHED)

    def legal_moves(self):
        if self.is_terminal():
            return []
        cdef void* c_movelist = StateLegalMoves(self._state)
        cdef int num_moves = NumMoves(c_movelist)
        moves = []
        cdef pyhanabi_move_t* c_move
        for i in range(num_moves):
            c_move = <pyhanabi_move_t*>malloc(sizeof(pyhanabi_move_t))
            if c_move == NULL:
                DeleteMoveList(c_movelist)
                raise MemoryError()
            GetMove(c_movelist, i, c_move)
            moves.append(HanabiMove.from_ptr(c_move))
        DeleteMoveList(c_movelist)
        return moves

    def move_is_legal(self, HanabiMove move):
        return <bint>MoveIsLegal(self._state, move._move)

    def card_playable_on_fireworks(self, color, rank):
        return <bint>CardPlayableOnFireworks(self._state, color, rank)

    def life_tokens(self):
        return StateLifeTokens(self._state)

    def num_players(self):
        return StateNumPlayers(self._state)

    def score(self):
        return StateScore(self._state)

    def move_history(self):
        history = []
        cdef int history_len = StateLenMoveHistory(self._state)
        cdef pyhanabi_history_item_t* c_history_item
        for i in range(history_len):
            c_history_item = <pyhanabi_history_item_t*>malloc(sizeof(pyhanabi_history_item_t))
            if c_history_item == NULL:
                raise MemoryError()
            StateGetMoveHistory(self._state, i, c_history_item)
            history.append(HanabiHistoryItem.from_ptr(c_history_item))
        return history
    
    def to_json(self):
        cdef char* json_state = StateToJson(self._state)
        if json_state == NULL:
            raise ValueError("Serialization failed: StateToJson returned NULL.")
        json_state_result = json_state.decode('utf-8')
        DeleteString(json_state)

        cdef char* json_game = GameToJson(self._game)
        if json_game == NULL:
            raise ValueError("Serialization failed: GameToJson returned NULL.")
        json_game_result = json_game.decode('utf-8')
        DeleteString(json_game)

        return json.dumps({"HanabiGame": json_game_result, "HanabiState": json_state_result})
    
    @classmethod
    def from_json(cls, str json_str):
        combined_json = json.loads(json_str)
        if 'HanabiGame' not in combined_json or 'HanabiState' not in combined_json:
            raise ValueError("JSON must contain both 'HanabiGame' and 'HanabiState' fields.")

        cdef pyhanabi_game_t* c_game = <pyhanabi_game_t*>malloc(sizeof(pyhanabi_game_t))
        if c_game == NULL:
            raise MemoryError()
        if not GameFromJson(combined_json['HanabiGame'].encode('ascii'), c_game):
            free(c_game)
            raise ValueError("Failed to deserialize HanabiGame from JSON")

        cdef pyhanabi_state_t* c_state = <pyhanabi_state_t*>malloc(sizeof(pyhanabi_state_t))
        if c_state == NULL:
            free(c_game)
            raise MemoryError()
        if not StateFromJson(combined_json['HanabiState'].encode('ascii'), c_state, c_game):
            free(c_state)
            free(c_game)
            raise ValueError("Failed to deserialize HanabiState from JSON")

        return HanabiState.create(None, c_state, c_game)

    def __str__(self):
        cdef char* c_string = StateToString(self._state)
        if c_string == NULL:
            return ""
        string = c_string.decode('utf-8')
        DeleteString(c_string)
        return string

    def __repr__(self):
        return self.__str__()
        
    def __dealloc__(self):
        if self._state != NULL:
            DeleteState(self._state)
            self._state = NULL
            
        if self._discard_pool != NULL:
            free(self._discard_pool)
            self._discard_pool = NULL
            
        for card in self._card_pool:
            free(<pyhanabi_card_t*>card)
        self._card_pool.clear()


cdef class HanabiGame:
    cdef pyhanabi_game_t* _game

    def __cinit__(self):
        self._game = NULL

    @staticmethod
    cdef from_ptr(pyhanabi_game_t* game):
        cdef HanabiGame instance = HanabiGame()
        instance._game = game
        return instance

    def __init__(self, dict params=None):
        cdef:
            int param_count, i = 0
            const char** c_array
            bytes key_bytes, value_bytes
            
        if params is None:
            self._game = <pyhanabi_game_t*>malloc(sizeof(pyhanabi_game_t))
            if self._game == NULL:
                raise MemoryError()
            NewDefaultGame(self._game)
            return

        param_count = len(params) * 2
        c_array = <const char**>malloc(param_count * sizeof(const char*))
        if c_array == NULL:
            raise MemoryError()

        try:
            for key, value in params.items():
                key_bytes = str(key).encode('ascii')
                value_bytes = str(value).encode('ascii')
                
                c_array[i] = strdup(<const char*>key_bytes)
                c_array[i+1] = strdup(<const char*>value_bytes)
                if c_array[i] == NULL or c_array[i+1] == NULL:
                    raise MemoryError()
                i += 2

            self._game = <pyhanabi_game_t*>malloc(sizeof(pyhanabi_game_t))
            if self._game == NULL:
                raise MemoryError()
            NewGame(self._game, param_count, c_array)
        finally:
            for j in range(i):
                if c_array[j] != NULL:
                    free(<void*>c_array[j])
            free(c_array)

    def __dealloc__(self):
        if self._game != NULL:
            DeleteGame(self._game)
            self._game = NULL

    def new_initial_state(self):
        return HanabiState.create(self, NULL, NULL)

    def num_players(self): 
        return NumPlayers(self._game)

    def num_colors(self): 
        return NumColors(self._game)

    def num_ranks(self): 
        return NumRanks(self._game)

    def hand_size(self): 
        return HandSize(self._game)

    def max_information_tokens(self): 
        return MaxInformationTokens(self._game)

    def max_life_tokens(self): 
        return MaxLifeTokens(self._game)

    def max_moves(self): 
        return MaxMoves(self._game)

    def observation_type(self):
        return AgentObservationType(ObservationType(self._game))

    def num_cards(self, int color, int rank):
        return NumCards(self._game, color, rank)

    def get_move_uid(self, HanabiMove move):
        if move is None or move._move == NULL:
            return 0
        return GetMoveUid(self._game, move._move)

    def get_move(self, int move_uid):
        cdef pyhanabi_move_t* move = <pyhanabi_move_t*>malloc(sizeof(pyhanabi_move_t))
        if move == NULL:
            raise MemoryError()
        GetMoveByUid(self._game, move_uid, move)
        return HanabiMove.from_ptr(move)

    def parameter_string(self):
        cdef char* c_string = GameParamString(self._game)
        if c_string == NULL:
            return ""
        string = c_string.decode('utf-8')
        DeleteString(c_string)
        return string

    def to_json(self):
        cdef char* c_string = GameToJson(self._game)
        if c_string == NULL:
            raise ValueError("Serialization failed: GameToJSON returned NULL")
        string = c_string.decode('utf-8')
        DeleteString(c_string)
        return string

    @classmethod
    def from_json(cls, str json_str):
        cdef pyhanabi_game_t* c_game = <pyhanabi_game_t*>malloc(sizeof(pyhanabi_game_t))
        if not GameFromJson(json_str.encode('ascii'), c_game):
            free(c_game)
            raise ValueError("Failed to deserialize HanabiGame from JSON")
        return HanabiGame.from_ptr(c_game)