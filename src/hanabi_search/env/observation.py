"""A player's view of a state, and its vector encoding (engine HanabiObservation)."""

import enum

import numpy as np
import numpy.typing as npt

from hanabi_search.env._pyhanabi import CData, card_values, ffi, lib, take_string
from hanabi_search.env.cards import HanabiCard, HanabiCardKnowledge
from hanabi_search.env.game import HanabiGame
from hanabi_search.env.moves import HanabiHistoryItem, HanabiMove
from hanabi_search.exceptions import EngineCallError


class HanabiObservation:
    """What one player can see: never their own cards, and seats relative to them.

    Player indices in an observation are offsets from the observer, who is
    always player 0. Build it with `HanabiState.observation(player)`.

    Args:
        game: The game the observed state belongs to; kept alive with the observation.
        c_state: Engine state to observe.
        player: Absolute id of the observing player.
    """

    __slots__ = ("_game", "_knowledge_pool", "_observation", "_plausible_cards", "_unseen_counts")

    def __init__(self, game: HanabiGame, c_state: CData, player: int) -> None:
        self._game = game
        self._observation: CData | None = ffi.new("pyhanabi_observation_t*")
        if not lib.NewObservation(c_state, player, self._observation):
            self._observation = None  # Nothing for __del__ to free.
            raise EngineCallError(f"Player {player} is out of range")
        # Reused buffers for card knowledge, one slot per (player, card).
        self._knowledge_pool = [
            ffi.new("pyhanabi_card_knowledge_t*")
            for _ in range(self.num_players() * game.hand_size())
        ]
        # Computed on first use; an observation never changes.
        self._unseen_counts: npt.NDArray[np.int32] | None = None
        self._plausible_cards: npt.NDArray[np.bool_] | None = None

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
        c_card = ffi.new("pyhanabi_card_t*")
        hands = []
        for offset in range(self.num_players()):
            hand = []
            for index in range(lib.ObsGetHandSize(self.c_observation, offset)):
                lib.ObsGetHandCard(self.c_observation, offset, index, c_card)
                hand.append(HanabiCard(*card_values(c_card)))
            hands.append(hand)
        return hands

    def card_knowledge(self) -> list[list[HanabiCardKnowledge]]:
        """Returns, per player offset and card, the knowledge accumulated from hints."""
        hand_size = self._game.hand_size()
        knowledge = []
        for offset in range(self.num_players()):
            hand = []
            for index in range(lib.ObsGetHandSize(self.c_observation, offset)):
                c_knowledge = self._knowledge_pool[offset * hand_size + index]
                lib.ObsGetHandCardKnowledge(self.c_observation, offset, index, c_knowledge)
                hand.append(HanabiCardKnowledge(c_knowledge, self))
            knowledge.append(hand)
        return knowledge

    def discard_pile(self) -> list[HanabiCard]:
        """Returns discarded cards, oldest first."""
        c_card = ffi.new("pyhanabi_card_t*")
        discards = []
        for index in range(lib.ObsDiscardPileSize(self.c_observation)):
            lib.ObsGetDiscard(self.c_observation, index, c_card)
            discards.append(HanabiCard(*card_values(c_card)))
        return discards

    def fireworks(self) -> list[int]:
        """Returns the height of each firework, ordered by color index."""
        return [
            int(lib.ObsFireworks(self.c_observation, color))
            for color in range(self._game.num_colors())
        ]

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

    def unseen_card_counts(self) -> npt.NDArray[np.int32]:
        """Returns how many copies of each card the observer cannot see.

        The full deck minus the other players' hands, the discard pile and the
        fireworks: what is left is in the deck or in the observer's own hand.

        Returns:
            Read-only counts of shape (num_colors, num_ranks), dtype int32.
        """
        if self._unseen_counts is None:
            game = self._game
            counts = np.array(
                [
                    [game.num_cards(color, rank) for rank in range(game.num_ranks())]
                    for color in range(game.num_colors())
                ],
                dtype=np.int32,
            )
            # The observer's own cards are hidden (invalid), so they are skipped.
            seen = [card for hand in self.observed_hands() for card in hand if card.valid()]
            for card in seen + self.discard_pile():
                counts[card.color(), card.rank()] -= 1
            for color, height in enumerate(self.fireworks()):
                counts[color, :height] -= 1
            counts.flags.writeable = False
            self._unseen_counts = counts
        return self._unseen_counts

    def plausible_cards(self) -> npt.NDArray[np.bool_]:
        """Returns which cards the hints still allow at each position of each hand.

        Returns:
            Read-only mask of shape (num_players, hand_size, num_colors, num_ranks),
            dtype bool: entry [offset, index, color, rank] is whether the card at
            `index` in the hand of player `offset` (from the observer) can be
            (color, rank). Positions past the end of a hand are all False.
        """
        if self._plausible_cards is None:
            game = self._game
            num_colors, num_ranks = game.num_colors(), game.num_ranks()
            shape = (self.num_players(), game.hand_size())
            colors = np.zeros((*shape, num_colors), dtype=np.bool_)
            ranks = np.zeros((*shape, num_ranks), dtype=np.bool_)
            for offset, hand in enumerate(self.card_knowledge()):
                for index, knowledge in enumerate(hand):
                    colors[offset, index] = [
                        knowledge.color_plausible(c) for c in range(num_colors)
                    ]
                    ranks[offset, index] = [knowledge.rank_plausible(r) for r in range(num_ranks)]
            mask = colors[..., :, None] & ranks[..., None, :]
            mask.flags.writeable = False
            self._plausible_cards = mask
        return self._plausible_cards

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

    __slots__ = ("_encoder", "_game")

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

    @property
    def _c_encoder(self) -> CData:
        assert self._encoder is not None
        return self._encoder

    def shape(self) -> list[int]:
        """Returns the shape of an encoded observation."""
        shape = take_string(lib.ObservationShape(self._c_encoder))
        return [int(dim) for dim in shape.split(",")]

    def encode(self, observation: HanabiObservation) -> list[int]:
        """Returns `observation` as a list of 0/1 values of length `shape()[0]`."""
        encoding = take_string(lib.EncodeObservation(self._c_encoder, observation.c_observation))
        return [int(bit) for bit in encoding.split(",")]

    def __del__(self) -> None:
        if self._encoder is not None:
            lib.DeleteObservationEncoder(self._encoder)
            self._encoder = None
