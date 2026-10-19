"""Moves and move-history items (wrappers of the engine's HanabiMove/HanabiHistoryItem)."""

import enum
from typing import Self

from hanabi_search.env._pyhanabi import CData, ffi, json_arg, lib, take_string
from hanabi_search.env.cards import color_idx_to_char
from hanabi_search.env.observation_types import MoveDict
from hanabi_search.exceptions import EngineCallError, SerializationError

# Bits in a history item's reveal bitmask (engine supports up to 8 cards per hand).
_REVEAL_BITMASK_BITS = 8


class HanabiMoveType(enum.IntEnum):
    """Move types, consistent with engine/hanabi_lib/hanabi_move.h."""

    INVALID = 0
    PLAY = 1
    DISCARD = 2
    REVEAL_COLOR = 3
    REVEAL_RANK = 4
    DEAL = 5
    # Fork additions used for MCTS re-determinization (see engine/CHANGES.md).
    RETURN = 6
    DEAL_SPECIFIC = 7


def _new_move(build: str, *args: int) -> CData:
    c_move = ffi.new("pyhanabi_move_t*")
    if not getattr(lib, build)(*args, c_move):
        raise EngineCallError(f"{build}{args} failed")
    return c_move


class HanabiMove:
    """An agent move or chance event.

    Args:
        move: Engine `pyhanabi_move_t*`; the new object takes ownership.
    """

    __slots__ = ("_card_index", "_color", "_move", "_rank", "_target_offset", "_type")

    def __init__(self, move: CData) -> None:
        self._move: CData | None = move
        self._type = HanabiMoveType(lib.MoveType(move))
        self._card_index = int(lib.CardIndex(move))
        self._target_offset = int(lib.TargetOffset(move))
        self._color = int(lib.MoveColor(move))
        self._rank = int(lib.MoveRank(move))

    @property
    def c_move(self) -> CData:
        """The underlying engine move."""
        assert self._move is not None
        return self._move

    def type(self) -> HanabiMoveType:
        """Returns the move type."""
        return self._type

    def card_index(self) -> int:
        """Returns the 0-based card index for PLAY, DISCARD, RETURN and DEAL_SPECIFIC."""
        return self._card_index

    def target_offset(self) -> int:
        """Returns the target player offset for REVEAL_* moves.

        For RETURN and DEAL_SPECIFIC this is the absolute player id instead.
        """
        return self._target_offset

    def color(self) -> int:
        """Returns the 0-based color index for REVEAL_COLOR and DEAL* moves."""
        return self._color

    def rank(self) -> int:
        """Returns the 0-based rank index for REVEAL_RANK and DEAL* moves."""
        return self._rank

    @classmethod
    def get_play_move(cls, card_index: int) -> Self:
        """Returns a move playing the card at `card_index`."""
        return cls(_new_move("GetPlayMove", card_index))

    @classmethod
    def get_discard_move(cls, card_index: int) -> Self:
        """Returns a move discarding the card at `card_index`."""
        return cls(_new_move("GetDiscardMove", card_index))

    @classmethod
    def get_reveal_color_move(cls, target_offset: int, color: int) -> Self:
        """Returns a color hint to the player `target_offset` seats away."""
        return cls(_new_move("GetRevealColorMove", target_offset, color))

    @classmethod
    def get_reveal_rank_move(cls, target_offset: int, rank: int) -> Self:
        """Returns a rank hint to the player `target_offset` seats away."""
        return cls(_new_move("GetRevealRankMove", target_offset, rank))

    @classmethod
    def get_return_move(cls, card_index: int, player: int) -> Self:
        """Returns a move putting `player`'s card at `card_index` back in the deck.

        Args:
            card_index: Index of the card in the player's hand.
            player: Absolute player id (not an offset).
        """
        return cls(_new_move("GetReturnMove", card_index, player))

    @classmethod
    def get_deal_specific_move(cls, card_index: int, player: int, color: int, rank: int) -> Self:
        """Returns a chance move dealing a chosen card to `player` at `card_index`.

        Args:
            card_index: Index in the player's hand where the card is inserted.
            player: Absolute player id (not an offset).
            color: Color index of the card to deal.
            rank: Rank index of the card to deal.
        """
        return cls(_new_move("GetDealSpecificMove", card_index, player, color, rank))

    def to_json(self) -> str:
        """Serializes the move to a JSON string."""
        c_string = lib.MoveToJson(self.c_move)
        if c_string == ffi.NULL:
            raise SerializationError("MoveToJson returned NULL")
        return take_string(c_string)

    @classmethod
    def from_json(cls, json_str: str) -> Self:
        """Deserializes a move produced by `to_json`."""
        c_move = ffi.new("pyhanabi_move_t*")
        if not lib.MoveFromJson(json_arg(json_str), c_move):
            raise SerializationError(f"Cannot deserialize HanabiMove from {json_str!r}")
        return cls(c_move)

    def to_dict(self) -> MoveDict:
        """Returns the move in dict form (see `MoveDict`).

        Raises:
            ValueError: For INVALID moves.
        """
        move_type = self._type
        move: MoveDict = {"action_type": move_type.name}
        match move_type:
            case HanabiMoveType.PLAY | HanabiMoveType.DISCARD | HanabiMoveType.RETURN:
                move["card_index"] = self._card_index
            case HanabiMoveType.REVEAL_COLOR:
                move["target_offset"] = self._target_offset
                move["color"] = color_idx_to_char(self._color)
            case HanabiMoveType.REVEAL_RANK:
                move["target_offset"] = self._target_offset
                move["rank"] = self._rank
            case HanabiMoveType.DEAL | HanabiMoveType.DEAL_SPECIFIC:
                move["color"] = color_idx_to_char(self._color)
                move["rank"] = self._rank
            case HanabiMoveType.INVALID:
                raise ValueError(f"Unsupported move: {self}")
        return move

    def __hash__(self) -> int:
        return hash((self._type, self._card_index, self._target_offset, self._color, self._rank))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, HanabiMove):
            return NotImplemented
        return (
            self._type == other._type
            and self._card_index == other._card_index
            and self._target_offset == other._target_offset
            and self._color == other._color
            and self._rank == other._rank
        )

    def __str__(self) -> str:
        return take_string(lib.MoveToString(self.c_move))

    def __repr__(self) -> str:
        return str(self)

    def __del__(self) -> None:
        if self._move is not None:
            lib.DeleteMove(self._move)
            self._move = None


def _bitmask_indices(bitmask: int) -> list[int]:
    return [i for i in range(_REVEAL_BITMASK_BITS) if bitmask & (1 << i)]


class HanabiHistoryItem:
    """A move that was made, with its side effects (card identity, scoring, ...).

    Args:
        item: Engine `pyhanabi_history_item_t*`; the new object takes ownership.
    """

    __slots__ = ("_item",)

    def __init__(self, item: CData) -> None:
        self._item: CData | None = item

    @property
    def _c_item(self) -> CData:
        assert self._item is not None
        return self._item

    def move(self) -> HanabiMove:
        """Returns the move that was made."""
        c_move = ffi.new("pyhanabi_move_t*")
        lib.HistoryItemMove(self._c_item, c_move)
        return HanabiMove(c_move)

    def player(self) -> int:
        """Returns the acting player (relative to the observer in observations)."""
        return int(lib.HistoryItemPlayer(self._c_item))

    def scored(self) -> bool:
        """Returns whether a PLAY move added the card to the fireworks."""
        return bool(lib.HistoryItemScored(self._c_item))

    def information_token(self) -> bool:
        """Returns whether a PLAY/DISCARD move gained an information token."""
        return bool(lib.HistoryItemInformationToken(self._c_item))

    def color(self) -> int:
        """Returns the color index of the played/discarded card."""
        return int(lib.HistoryItemColor(self._c_item))

    def rank(self) -> int:
        """Returns the rank index of the played/discarded card."""
        return int(lib.HistoryItemRank(self._c_item))

    def card_info_revealed(self) -> list[int]:
        """Returns the hand indices matched by a hint (e.g. all red cards)."""
        return _bitmask_indices(int(lib.HistoryItemRevealBitmask(self._c_item)))

    def card_info_newly_revealed(self) -> list[int]:
        """Returns the hand indices for which a hint revealed something new."""
        return _bitmask_indices(int(lib.HistoryItemNewlyRevealedBitmask(self._c_item)))

    def deal_to_player(self) -> int:
        """Returns the player a DEAL move dealt to."""
        return int(lib.HistoryItemDealToPlayer(self._c_item))

    def to_json(self) -> str:
        """Serializes the history item to a JSON string."""
        c_string = lib.HistoryItemToJson(self._c_item)
        if c_string == ffi.NULL:
            raise SerializationError("HistoryItemToJson returned NULL")
        return take_string(c_string)

    @classmethod
    def from_json(cls, json_str: str) -> Self:
        """Deserializes a history item produced by `to_json`."""
        c_item = ffi.new("pyhanabi_history_item_t*")
        if not lib.HistoryItemFromJson(json_arg(json_str), c_item):
            raise SerializationError(f"Cannot deserialize HanabiHistoryItem from {json_str!r}")
        return cls(c_item)

    def __str__(self) -> str:
        return take_string(lib.HistoryItemToString(self._c_item))

    def __repr__(self) -> str:
        return str(self)

    def __del__(self) -> None:
        if self._item is not None:
            lib.DeleteHistoryItem(self._item)
            self._item = None
