"""Cards, card knowledge and color helpers (wrappers of the engine's card types)."""

from typing import TYPE_CHECKING

from hanabi_search.env._pyhanabi import CData, lib, take_string
from hanabi_search.env.observation_types import CardDict, KnowledgeDict

if TYPE_CHECKING:
    from hanabi_search.env.observation import HanabiObservation

# Color order used by the engine (engine/hanabi_lib/util.cc).
COLOR_CHAR = ("R", "Y", "G", "W", "B")


def color_idx_to_char(color_idx: int) -> str | None:
    """Converts a color index to its character, or `None` for -1 (unknown).

    Args:
        color_idx: Color index in [0, num_colors), or -1.

    Returns:
        The color character, or `None` if the color is unknown.

    Raises:
        IndexError: If the index is neither -1 nor a valid color.
    """
    if color_idx == -1:
        return None
    if not 0 <= color_idx < len(COLOR_CHAR):
        raise IndexError(f"Invalid color index {color_idx}")
    return COLOR_CHAR[color_idx]


def color_char_to_idx(color_char: str) -> int:
    """Converts a color character to its index.

    Args:
        color_char: One of `COLOR_CHAR`.

    Returns:
        The color index.

    Raises:
        ValueError: If `color_char` is not a valid color.
    """
    try:
        return COLOR_CHAR.index(color_char)
    except ValueError:
        raise ValueError(f"Invalid color: {color_char}. Should be one of {COLOR_CHAR}.") from None


class HanabiCard:
    """A card with a color and a rank; both are -1 for a hidden card.

    Args:
        color: 0-based color index (order RYGWB), or -1 if unknown.
        rank: 0-based rank index (0 is a "1" card), or -1 if unknown.
    """

    __slots__ = ("_color", "_rank")

    def __init__(self, color: int, rank: int) -> None:
        self._color = color
        self._rank = rank

    def color(self) -> int:
        """Returns the 0-based color index, or -1 if hidden."""
        return self._color

    def rank(self) -> int:
        """Returns the 0-based rank index, or -1 if hidden."""
        return self._rank

    def valid(self) -> bool:
        """Returns whether the card's identity is known."""
        return self._color >= 0 and self._rank >= 0

    def to_dict(self) -> CardDict:
        """Returns the card as `{"color": char | None, "rank": int}`."""
        return {"color": color_idx_to_char(self._color), "rank": self._rank}

    def __str__(self) -> str:
        return COLOR_CHAR[self._color] + str(self._rank + 1) if self.valid() else "XX"

    def __repr__(self) -> str:
        return str(self)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, HanabiCard):
            return NotImplemented
        return self._color == other._color and self._rank == other._rank

    def __hash__(self) -> int:
        return hash((self._color, self._rank))


class HanabiCardKnowledge:
    """What a player knows about one of their cards from hints.

    `color()`/`rank()` only report directly revealed values; the `*_plausible`
    methods also account for negative information (e.g. "not red" because the
    card was not included in a red hint).

    Args:
        knowledge: Engine `pyhanabi_card_knowledge_t*`.
        owner: The observation the knowledge points into; kept alive with it.
    """

    __slots__ = ("_knowledge", "_owner")

    def __init__(self, knowledge: CData, owner: HanabiObservation) -> None:
        self._knowledge = knowledge
        self._owner = owner

    def color(self) -> int | None:
        """Returns the color index if it was directly hinted, else `None`."""
        return int(lib.KnownColor(self._knowledge)) if lib.ColorWasHinted(self._knowledge) else None

    def color_plausible(self, color_index: int) -> bool:
        """Returns whether no hint rules out `color_index`."""
        return bool(lib.ColorIsPlausible(self._knowledge, color_index))

    def rank(self) -> int | None:
        """Returns the rank index if it was directly hinted, else `None`."""
        return int(lib.KnownRank(self._knowledge)) if lib.RankWasHinted(self._knowledge) else None

    def rank_plausible(self, rank_index: int) -> bool:
        """Returns whether no hint rules out `rank_index`."""
        return bool(lib.RankIsPlausible(self._knowledge, rank_index))

    def to_dict(self) -> KnowledgeDict:
        """Returns the hinted color and rank (`None` where not hinted)."""
        color = self.color()
        return {"color": None if color is None else color_idx_to_char(color), "rank": self.rank()}

    def __str__(self) -> str:
        return take_string(lib.CardKnowledgeToString(self._knowledge))

    def __repr__(self) -> str:
        return str(self)
