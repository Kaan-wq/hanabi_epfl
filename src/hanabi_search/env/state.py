"""Full game state (wrapper of the engine's HanabiState)."""

import enum
import json
from typing import Self

import numpy as np

from hanabi_search.env._pyhanabi import (
    CData,
    card_values,
    engine_message,
    ffi,
    int_values,
    lib,
    take_string,
)
from hanabi_search.env.cards import HanabiCard
from hanabi_search.env.game import HanabiGame
from hanabi_search.env.moves import HanabiHistoryItem, HanabiMove
from hanabi_search.env.observation import HanabiObservation, HandArrays
from hanabi_search.exceptions import EngineCallError, IllegalMoveError, SerializationError

CHANCE_PLAYER_ID = -1


class HanabiEndOfGameType(enum.IntEnum):
    """End-of-game conditions, consistent with engine/hanabi_lib/hanabi_state.h."""

    NOT_FINISHED = 0
    OUT_OF_LIFE_TOKENS = 1
    OUT_OF_CARDS = 2
    COMPLETED_FIREWORKS = 3


class HanabiState:
    """Complete state of a game, including every hand and the deck.

    Chance events (dealing) are explicit: when `cur_player()` is
    `CHANCE_PLAYER_ID`, a card must be dealt before anyone can act.

    A state keeps its `HanabiGame` alive: the engine state refers to it.

    Args:
        game: The game this state belongs to.
    """

    __slots__ = ("_game", "_state")

    def __init__(self, game: HanabiGame) -> None:
        self._game = game
        self._state: CData | None = ffi.new("pyhanabi_state_t*")
        lib.NewState(game.c_game, self._state)

    @classmethod
    def _copy_of(cls, game: HanabiGame, c_state: CData) -> Self:
        state = cls.__new__(cls)
        state._game = game
        state._state = ffi.new("pyhanabi_state_t*")
        lib.CopyState(c_state, state._state)
        return state

    @property
    def c_state(self) -> CData:
        """The underlying engine state."""
        assert self._state is not None
        return self._state

    @property
    def game(self) -> HanabiGame:
        """The game this state belongs to."""
        return self._game

    def copy(self) -> Self:
        """Returns an independent copy that shares this state's game."""
        return self._copy_of(self._game, self.c_state)

    def assign(self, other: HanabiState) -> None:
        """Makes this state a copy of `other`, reusing this state's engine memory.

        Cheaper than `other.copy()` when one state is reset over and over. This
        state then shares `other`'s game.
        """
        lib.pyhanabi_state_assign(self.c_state, other.c_state)
        self._game = other._game

    def observation(self, player: int) -> HanabiObservation:
        """Returns `player`'s view of this state."""
        return HanabiObservation(self._game, self.c_state, player)

    def apply_move(self, move: HanabiMove) -> None:
        """Applies `move` for the acting player (or chance).

        Raises:
            IllegalMoveError: If `move` is not legal in this state; nothing changes.
        """
        if not lib.StateApplyMove(self.c_state, move.c_move):
            raise IllegalMoveError(f"{engine_message(f'Illegal move {move}')}\nState:\n{self}")

    def deal_random_card(self) -> None:
        """Deals a random card.

        Raises:
            EngineCallError: If it is not a chance turn or the deck is empty.
        """
        if not lib.StateDealCard(self.c_state):
            raise EngineCallError(engine_message("No card can be dealt"))

    def remove_knowledge(self, player: int, card_index: int) -> None:
        """Resets all hint knowledge about one of `player`'s cards.

        `card_index` is a knowledge slot, which outlives its card across a return move.

        Raises:
            EngineCallError: If `player` or `card_index` is out of range.
        """
        if not lib.StateRemoveKnowledge(self.c_state, player, card_index):
            raise EngineCallError(
                engine_message(f"Cannot remove the knowledge of player {player} at {card_index}")
            )

    def cur_player(self) -> int:
        """Returns the acting player, or `CHANCE_PLAYER_ID` if a card must be dealt."""
        return int(lib.StateCurPlayer(self.c_state))

    def turns_to_play(self) -> int:
        """Returns the number of turns left once the deck is empty."""
        return int(lib.StateTurnsToPlay(self.c_state))

    def deck_size(self) -> int:
        """Returns the number of cards left in the deck."""
        return int(lib.StateDeckSize(self.c_state))

    def discard_pile(self) -> list[HanabiCard]:
        """Returns discarded cards, oldest first."""
        c_card = ffi.new("pyhanabi_card_t*")
        discards = []
        for index in range(lib.StateDiscardPileSize(self.c_state)):
            lib.StateGetDiscard(self.c_state, index, c_card)
            discards.append(HanabiCard(*card_values(c_card)))
        return discards

    def fireworks(self) -> list[int]:
        """Returns the height of each firework, ordered by color index."""
        return [int(lib.StateFireworks(self.c_state, c)) for c in range(self._game.num_colors())]

    def progress(self) -> int:
        """Returns the number of cards on the fireworks, even if all lives are lost."""
        return sum(self.fireworks())

    def score(self) -> int:
        """Returns the game score: the fireworks total, or 0 if all lives are lost."""
        return int(lib.StateScore(self.c_state))

    def player_hands(self) -> list[list[HanabiCard]]:
        """Returns every player's hand, cards ordered oldest to newest."""
        hands = self.hand_arrays()
        return [
            _cards(colors[:size], ranks[:size])
            for size, colors, ranks in zip(
                hands.sizes.tolist(), hands.colors.tolist(), hands.ranks.tolist(), strict=True
            )
        ]

    def player_hand(self, player: int) -> list[HanabiCard]:
        """Returns `player`'s hand, cards ordered oldest to newest.

        Raises:
            IndexError: If `player` is not in [0, num_players).
        """
        if not 0 <= player < self._game.num_players():
            raise IndexError(f"Player {player} is out of range")
        hands = self.hand_arrays()
        size = int(hands.sizes[player])
        return _cards(hands.colors[player, :size].tolist(), hands.ranks[player, :size].tolist())

    def hand_arrays(self) -> HandArrays:
        """Returns every hand as arrays (see `HandArrays`); players are absolute ids.

        Returns:
            Fresh arrays: the state can change, so nothing is cached.
        """
        game = self._game
        shape = (game.num_players(), game.hand_size())
        sizes = np.empty(shape[0], dtype=np.int32)
        colors = np.empty(shape, dtype=np.int8)
        ranks = np.empty(shape, dtype=np.int8)
        if not lib.pyhanabi_state_hands(
            self.c_state,
            ffi.from_buffer("int32_t[]", sizes),
            ffi.from_buffer("int8_t[]", colors),
            ffi.from_buffer("int8_t[]", ranks),
            *shape,
        ):
            raise EngineCallError(engine_message("Cannot read the hands"))
        return HandArrays(sizes, colors, ranks)

    def information_tokens(self) -> int:
        """Returns the number of information tokens left."""
        return int(lib.StateInformationTokens(self.c_state))

    def life_tokens(self) -> int:
        """Returns the number of life tokens left."""
        return int(lib.StateLifeTokens(self.c_state))

    def num_players(self) -> int:
        """Returns the number of players."""
        return int(lib.StateNumPlayers(self.c_state))

    def end_of_game_status(self) -> HanabiEndOfGameType:
        """Returns why the game ended, or NOT_FINISHED."""
        return HanabiEndOfGameType(lib.StateEndOfGameStatus(self.c_state))

    def is_terminal(self) -> bool:
        """Returns whether the game has ended."""
        return self.end_of_game_status() != HanabiEndOfGameType.NOT_FINISHED

    def legal_moves(self) -> list[HanabiMove]:
        """Returns the acting player's legal moves (empty once the game has ended).

        The moves are shared with `HanabiGame.all_moves` (moves are immutable).
        """
        moves = self._game.all_moves()
        return [moves[uid] for uid in self.legal_move_uids()]

    def legal_move_uids(self) -> list[int]:
        """Returns the ids of the acting player's legal moves, in `legal_moves` order."""
        buffer = self._game.move_uid_buffer()
        count = lib.pyhanabi_state_legal_move_uids(self.c_state, buffer, len(buffer))
        if count < 0:
            raise EngineCallError(engine_message("Cannot read the legal moves"))
        return int_values(buffer, count)

    def move_is_legal(self, move: HanabiMove) -> bool:
        """Returns whether `move` is legal for the acting player."""
        return bool(lib.MoveIsLegal(self.c_state, move.c_move))

    def card_playable_on_fireworks(self, color: int, rank: int) -> bool:
        """Returns whether the card (color, rank) can be played right now."""
        return bool(lib.CardPlayableOnFireworks(self.c_state, color, rank))

    def move_history(self) -> list[HanabiHistoryItem]:
        """Returns every move made so far, oldest first."""
        history = []
        for index in range(lib.StateLenMoveHistory(self.c_state)):
            c_item = ffi.new("pyhanabi_history_item_t*")
            lib.StateGetMoveHistory(self.c_state, index, c_item)
            history.append(HanabiHistoryItem(c_item))
        return history

    def to_json(self) -> str:
        """Serializes the state together with its game (parameters and RNG state)."""
        c_state_json = lib.StateToJson(self.c_state)
        if c_state_json == ffi.NULL:
            raise SerializationError(engine_message("Cannot serialize the state"))
        return json.dumps(
            {"HanabiGame": self._game.to_json(), "HanabiState": take_string(c_state_json)}
        )

    @classmethod
    def from_json(cls, json_str: str) -> Self:
        """Deserializes a state produced by `to_json`, with a new copy of its game."""
        try:
            payload = json.loads(json_str)
        except json.JSONDecodeError as error:
            raise SerializationError(f"Invalid JSON: {error}") from None
        if not isinstance(payload, dict) or {"HanabiGame", "HanabiState"} - payload.keys():
            raise SerializationError("JSON must contain 'HanabiGame' and 'HanabiState'")
        game = HanabiGame.from_json(payload["HanabiGame"])
        c_state = ffi.new("pyhanabi_state_t*")
        if not lib.StateFromJson(str(payload["HanabiState"]).encode(), c_state, game.c_game):
            raise SerializationError(engine_message("Cannot deserialize HanabiState"))
        # Only take ownership once the engine state exists, so __del__ never frees garbage.
        state = cls.__new__(cls)
        state._game = game
        state._state = c_state
        return state

    def __str__(self) -> str:
        return take_string(lib.StateToString(self.c_state))

    def __repr__(self) -> str:
        return str(self)

    def __del__(self) -> None:
        if self._state is not None:
            lib.DeleteState(self._state)
            self._state = None


def _cards(colors: list[int], ranks: list[int]) -> list[HanabiCard]:
    return [HanabiCard(color, rank) for color, rank in zip(colors, ranks, strict=True)]
