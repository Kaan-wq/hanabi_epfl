"""Typed shapes of the dict observations and actions exchanged with agents."""

from typing import TYPE_CHECKING, NotRequired, TypedDict

if TYPE_CHECKING:
    from hanabi_search.env.moves import HanabiMove
    from hanabi_search.env.observation import HanabiObservation


class CardDict(TypedDict):
    """A card as seen by an observer; both fields are hidden (None / -1) for own cards."""

    color: str | None
    rank: int


class KnowledgeDict(TypedDict):
    """Directly hinted color and rank of a card; `None` where not hinted."""

    color: str | None
    rank: int | None


class MoveDict(TypedDict):
    """An action in dict form, e.g. `{"action_type": "PLAY", "card_index": 0}`.

    Which optional keys are present depends on `action_type`:
    PLAY/DISCARD use `card_index`; REVEAL_COLOR uses `target_offset` and `color`;
    REVEAL_RANK uses `target_offset` and `rank`; RETURN uses `card_index` and
    `player`; DEAL/DEAL_SPECIFIC use `color` and `rank`.
    """

    action_type: str
    card_index: NotRequired[int]
    target_offset: NotRequired[int]
    color: NotRequired[str | None]
    rank: NotRequired[int]
    player: NotRequired[int]


class ObservationDict(TypedDict):
    """One player's observation. Player-indexed lists are relative to the observer."""

    current_player: int
    current_player_offset: int
    life_tokens: int
    information_tokens: int
    max_information_tokens: int
    num_players: int
    num_colors: int
    num_ranks: int
    # Number of copies of each rank per color in the full deck, indexed by rank.
    cards_per_rank: list[int]
    deck_size: int
    turns_to_play: int
    fireworks: dict[str, int]
    legal_moves: list[MoveDict]
    legal_moves_as_int: list[int]
    observed_hands: list[list[CardDict]]
    discard_pile: list[CardDict]
    card_knowledge: list[list[KnowledgeDict]]
    pyhanabi: HanabiObservation


class AllPlayerObservations(TypedDict):
    """Observations of every player after a step."""

    current_player: int
    player_observations: list[ObservationDict]


# What an agent may return from `act`: a dict move, an engine move, or a move uid.
type Action = MoveDict | HanabiMove | int
