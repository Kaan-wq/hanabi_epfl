import json
import enum
from cyhanabi cimport DeleteString
from cyhanabi cimport pyhanabi_card_knowledge_t, CardKnowledgeToString, ColorWasHinted, KnownColor, ColorIsPlausible, RankWasHinted, KnownRank, RankIsPlausible
from cyhanabi cimport pyhanabi_move_t, DeleteMoveList, NumMoves, GetMove, DeleteMove, MoveToString, MoveType, CardIndex, TargetOffset, MoveColor, MoveRank, GetDiscardMove, GetReturnMove, GetPlayMove, GetRevealColorMove, GetRevealRankMove, GetDealSpecificMove
from libc.stdlib cimport malloc, free

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

    
