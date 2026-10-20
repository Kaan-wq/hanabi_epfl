"""A player's view of a state, and its vector encoding (engine HanabiObservation)."""

import enum
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from hanabi_search.env._pyhanabi import CData, ffi, lib, take_string
from hanabi_search.env.cards import HanabiCard, HanabiCardKnowledge
from hanabi_search.env.game import HanabiGame
from hanabi_search.env.moves import HanabiHistoryItem, HanabiMove
from hanabi_search.exceptions import EngineCallError


@dataclass(frozen=True, slots=True)
class HandArrays:
    """Every hand as the observer sees it; player axes are offsets (0 is the observer).

    Attributes:
        sizes: Number of cards in each hand, shape (num_players,), int32.
        colors: Color of each card, shape (num_players, hand_size), int8; -1 for
            the observer's own cards (hidden) and past the end of a hand.
        ranks: Rank of each card, same shape and convention as `colors`.
    """

    sizes: npt.NDArray[np.int32]
    colors: npt.NDArray[np.int8]
    ranks: npt.NDArray[np.int8]


@dataclass(frozen=True, slots=True)
class KnowledgeArrays:
    """What each player knows about their cards from hints; axes as in `HandArrays`.

    Attributes:
        hinted_colors: Directly hinted color, shape (num_players, hand_size),
            int8; -1 if not hinted and past the end of a hand.
        hinted_ranks: Directly hinted rank, same shape and convention.
        plausible: Shape (num_players, hand_size, num_colors, num_ranks), bool:
            whether no hint (negative ones included) rules out that card at
            that position; False past the end of a hand.
    """

    hinted_colors: npt.NDArray[np.int8]
    hinted_ranks: npt.NDArray[np.int8]
    plausible: npt.NDArray[np.bool_]


def _read_only[T: np.generic](array: npt.NDArray[T]) -> npt.NDArray[T]:
    array.flags.writeable = False
    return array


class HanabiObservation:
    """What one player can see: never their own cards, and seats relative to them.

    Player indices in an observation are offsets from the observer, who is
    always player 0. Build it with `HanabiState.observation(player)`.

    The array methods fill a whole field with one engine call; their results
    are cached (an observation never changes) and read-only.

    Args:
        game: The game the observed state belongs to; kept alive with the observation.
        c_state: Engine state to observe.
        player: Absolute id of the observing player.
    """

    __slots__ = ("_game", "_hands", "_knowledge", "_knowledge_pool", "_observation", "_unseen")

    def __init__(self, game: HanabiGame, c_state: CData, player: int) -> None:
        self._game = game
        self._observation: CData | None = ffi.new("pyhanabi_observation_t*")
        if not lib.NewObservation(c_state, player, self._observation):
            self._observation = None  # Nothing for __del__ to free.
            raise EngineCallError(f"Player {player} is out of range")
        # Reused buffers for `card_knowledge`, one slot per (player, card); made on first use.
        self._knowledge_pool: list[CData] | None = None
        self._hands: HandArrays | None = None
        self._knowledge: KnowledgeArrays | None = None
        self._unseen: npt.NDArray[np.int32] | None = None

    @property
    def c_observation(self) -> CData:
        """The underlying engine observation."""
        assert self._observation is not None
        return self._observation

    def cur_player_offset(self) -> int:
        """Returns the acting player's offset from the observer."""
        return int(lib.ObsCurPlayerOffset(self.c_observation))

    def num_players(self) -> int:
        """Returns the number of players."""
        return int(lib.ObsNumPlayers(self.c_observation))

    def observed_hands(self) -> list[list[HanabiCard]]:
        """Returns every hand by offset; the observer's own cards are hidden (invalid)."""
        hands = self.hand_arrays()
        return [
            [
                HanabiCard(color, rank)
                for color, rank in zip(colors[:size], ranks[:size], strict=True)
            ]
            for size, colors, ranks in zip(
                hands.sizes.tolist(), hands.colors.tolist(), hands.ranks.tolist(), strict=True
            )
        ]

    def hand_arrays(self) -> HandArrays:
        """Returns every hand as arrays (see `HandArrays`)."""
        if self._hands is None:
            game = self._game
            shape = (game.num_players(), game.hand_size())
            sizes = np.empty(shape[0], dtype=np.int32)
            colors = np.empty(shape, dtype=np.int8)
            ranks = np.empty(shape, dtype=np.int8)
            if not lib.pyhanabi_observation_hands(
                self.c_observation,
                ffi.from_buffer("int32_t[]", sizes),
                ffi.from_buffer("int8_t[]", colors),
                ffi.from_buffer("int8_t[]", ranks),
                *shape,
            ):
                raise EngineCallError("pyhanabi_observation_hands rejected the game's shape")
            self._hands = HandArrays(_read_only(sizes), _read_only(colors), _read_only(ranks))
        return self._hands

    def card_knowledge(self) -> list[list[HanabiCardKnowledge]]:
        """Returns, per player offset and card, the knowledge accumulated from hints."""
        hand_size = self._game.hand_size()
        if self._knowledge_pool is None:
            self._knowledge_pool = [
                ffi.new("pyhanabi_card_knowledge_t*") for _ in range(self.num_players() * hand_size)
            ]
        knowledge = []
        for offset in range(self.num_players()):
            hand = []
            for index in range(lib.ObsGetHandSize(self.c_observation, offset)):
                c_knowledge = self._knowledge_pool[offset * hand_size + index]
                lib.ObsGetHandCardKnowledge(self.c_observation, offset, index, c_knowledge)
                hand.append(HanabiCardKnowledge(c_knowledge, self))
            knowledge.append(hand)
        return knowledge

    def knowledge_arrays(self) -> KnowledgeArrays:
        """Returns what each player knows from hints, as arrays (see `KnowledgeArrays`)."""
        if self._knowledge is None:
            game = self._game
            shape = (game.num_players(), game.hand_size())
            cards = (game.num_colors(), game.num_ranks())
            hinted_colors = np.empty(shape, dtype=np.int8)
            hinted_ranks = np.empty(shape, dtype=np.int8)
            plausible = np.empty((*shape, *cards), dtype=np.uint8)
            if not lib.pyhanabi_observation_card_knowledge(
                self.c_observation,
                ffi.from_buffer("int8_t[]", hinted_colors),
                ffi.from_buffer("int8_t[]", hinted_ranks),
                ffi.from_buffer("uint8_t[]", plausible),
                *shape,
                *cards,
            ):
                raise EngineCallError("pyhanabi_observation_card_knowledge rejected the shape")
            self._knowledge = KnowledgeArrays(
                _read_only(hinted_colors),
                _read_only(hinted_ranks),
                _read_only(plausible.view(np.bool_)),  # The engine writes only 0 and 1.
            )
        return self._knowledge

    def plausible_cards(self) -> npt.NDArray[np.bool_]:
        """Returns which cards the hints still allow at each position of each hand.

        Returns:
            Read-only mask of shape (num_players, hand_size, num_colors, num_ranks):
            entry [offset, index, color, rank] is whether the card at `index` in
            the hand of player `offset` can be (color, rank). Positions past the
            end of a hand are all False.
        """
        return self.knowledge_arrays().plausible

    def unseen_card_counts(self) -> npt.NDArray[np.int32]:
        """Returns how many copies of each card the observer cannot see.

        The full deck minus the other players' hands, the discard pile and the
        fireworks: what is left is in the deck or in the observer's own hand.

        Returns:
            Read-only counts of shape (num_colors, num_ranks), dtype int32.
        """
        if self._unseen is None:
            game = self._game
            counts = np.empty((game.num_colors(), game.num_ranks()), dtype=np.int32)
            if not lib.pyhanabi_observation_unseen_card_counts(
                self.c_observation, ffi.from_buffer("int32_t[]", counts), counts.size
            ):
                raise EngineCallError("pyhanabi_observation_unseen_card_counts rejected the shape")
            self._unseen = _read_only(counts)
        return self._unseen

    def discard_pile(self) -> list[HanabiCard]:
        """Returns discarded cards, oldest first."""
        colors, ranks = self.discard_pile_arrays()
        return [HanabiCard(c, r) for c, r in zip(colors.tolist(), ranks.tolist(), strict=True)]

    def discard_pile_arrays(self) -> tuple[npt.NDArray[np.int8], npt.NDArray[np.int8]]:
        """Returns the colors and ranks of the discarded cards, oldest first.

        Returns:
            Two arrays of shape (num_discards,), dtype int8.
        """
        # No card has more than 3 copies (engine NumberCardInstances), so this bounds the deck.
        capacity = 3 * self._game.num_colors() * self._game.num_ranks()
        colors = np.empty(capacity, dtype=np.int8)
        ranks = np.empty(capacity, dtype=np.int8)
        count = lib.pyhanabi_observation_discard_pile(
            self.c_observation,
            ffi.from_buffer("int8_t[]", colors),
            ffi.from_buffer("int8_t[]", ranks),
            capacity,
        )
        if count < 0:
            raise EngineCallError("Discard pile larger than the deck")
        return colors[:count], ranks[:count]

    def fireworks(self) -> list[int]:
        """Returns the height of each firework, ordered by color index."""
        heights = np.empty(self._game.num_colors(), dtype=np.int32)
        if not lib.pyhanabi_observation_fireworks(
            self.c_observation, ffi.from_buffer("int32_t[]", heights), heights.size
        ):
            raise EngineCallError("pyhanabi_observation_fireworks rejected the number of colors")
        return heights.tolist()

    def deck_size(self) -> int:
        """Returns the number of cards left in the deck."""
        return int(lib.ObsDeckSize(self.c_observation))

    def last_moves(self) -> list[HanabiHistoryItem]:
        """Returns moves since the observer last acted, most recent first.

        Initial deals are skipped. Moves are relative to the observer.
        """
        items = []
        for index in range(lib.ObsNumLastMoves(self.c_observation)):
            c_item = ffi.new("pyhanabi_history_item_t*")
            lib.ObsGetLastMove(self.c_observation, index, c_item)
            items.append(HanabiHistoryItem(c_item))
        return items

    def information_tokens(self) -> int:
        """Returns the number of information tokens left."""
        return int(lib.ObsInformationTokens(self.c_observation))

    def life_tokens(self) -> int:
        """Returns the number of life tokens left."""
        return int(lib.ObsLifeTokens(self.c_observation))

    def legal_moves(self) -> list[HanabiMove]:
        """Returns the observer's legal moves (empty unless the observer is acting)."""
        moves = []
        for index in range(lib.ObsNumLegalMoves(self.c_observation)):
            c_move = ffi.new("pyhanabi_move_t*")
            lib.ObsGetLegalMove(self.c_observation, index, c_move)
            moves.append(HanabiMove(c_move))
        return moves

    def legal_move_uids(self) -> npt.NDArray[np.int32]:
        """Returns the ids (see `HanabiGame.get_move_uid`) of the observer's legal moves.

        Returns:
            Shape (num_legal_moves,), dtype int32, in the order of `legal_moves`;
            empty unless the observer is acting.
        """
        uids = np.empty(self._game.max_moves(), dtype=np.int32)
        count = lib.pyhanabi_observation_legal_move_uids(
            self.c_observation, ffi.from_buffer("int32_t[]", uids), uids.size
        )
        if count < 0:
            raise EngineCallError("More legal moves than the game's action space")
        return uids[:count]

    def card_playable_on_fireworks(self, color: int, rank: int) -> bool:
        """Returns whether the card (color, rank) can be played right now."""
        return bool(lib.ObsCardPlayableOnFireworks(self.c_observation, color, rank))

    def __str__(self) -> str:
        return take_string(lib.ObsToString(self.c_observation))

    def __repr__(self) -> str:
        return str(self)

    def __del__(self) -> None:
        if self._observation is not None:
            lib.DeleteObservation(self._observation)
            self._observation = None


# Identifies the observation layout networks are trained on; stored with every
# checkpoint. Bump it whenever the encoding changes (see CLAUDE.md).
OBSERVATION_ENCODER_VERSION = "hle-canonical-1"


class ObservationEncoderType(enum.IntEnum):
    """Encoder types, consistent with engine/hanabi_lib/observation_encoder.h."""

    CANONICAL = 0


class ObservationEncoder:
    """Encodes observations as flat bit vectors for neural networks.

    Args:
        game: The game whose observations will be encoded.
        encoder_type: Which encoding to use.
    """

    __slots__ = ("_encoder", "_game", "_length")

    def __init__(
        self,
        game: HanabiGame,
        encoder_type: ObservationEncoderType = ObservationEncoderType.CANONICAL,
    ) -> None:
        self._game = game
        self._encoder: CData | None = ffi.new("pyhanabi_observation_encoder_t*")
        if not lib.NewObservationEncoder(self._encoder, game.c_game, encoder_type):
            self._encoder = None  # Nothing for __del__ to free.
            raise EngineCallError(f"Unknown observation encoder type {encoder_type}")
        self._length = int(np.prod(self.shape()))

    @property
    def _c_encoder(self) -> CData:
        assert self._encoder is not None
        return self._encoder

    def shape(self) -> list[int]:
        """Returns the shape of an encoded observation."""
        shape = take_string(lib.ObservationShape(self._c_encoder))
        return [int(dim) for dim in shape.split(",")]

    def encode(self, observation: HanabiObservation) -> npt.NDArray[np.uint8]:
        """Returns `observation` as 0/1 values, shape (`shape()[0]`,), dtype uint8."""
        bits = np.empty(self._length, dtype=np.uint8)
        if not lib.pyhanabi_observation_encode(
            self._c_encoder,
            observation.c_observation,
            ffi.from_buffer("uint8_t[]", bits),
            bits.size,
        ):
            raise EngineCallError("pyhanabi_observation_encode rejected the encoding length")
        return bits

    def __del__(self) -> None:
        if self._encoder is not None:
            lib.DeleteObservationEncoder(self._encoder)
            self._encoder = None
