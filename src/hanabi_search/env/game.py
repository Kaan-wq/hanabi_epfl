"""Game parameters (wrapper of the engine's HanabiGame)."""

import enum
from collections.abc import Mapping
from typing import Self

from hanabi_search.env._pyhanabi import CData, ffi, json_arg, lib, take_string
from hanabi_search.env.moves import HanabiMove
from hanabi_search.exceptions import SerializationError

type GameParams = Mapping[str, int | bool | str]


class AgentObservationType(enum.IntEnum):
    """Observation types, consistent with engine/hanabi_lib/hanabi_game.h.

    MINIMAL is what a human sees, without memory of past hints. CARD_KNOWLEDGE
    adds per-card knowledge from past hints, including negative information.
    SEER shows every card, including the observer's own.
    """

    MINIMAL = 0
    CARD_KNOWLEDGE = 1
    SEER = 2


class HanabiGame:
    """Parameters of one Hanabi variant, plus the RNG used for dealing.

    Recognized parameters: `players` (2-5), `colors` (1-5), `ranks` (1-5),
    `hand_size`, `max_information_tokens`, `max_life_tokens`, `seed` (-1 for a
    random seed), `random_start_player` and `observation_type`
    (an `AgentObservationType` value). Missing parameters take engine defaults.

    Args:
        params: Game parameters; `None` for the engine's default game.
    """

    __slots__ = ("_game",)

    def __init__(self, params: GameParams | None = None) -> None:
        self._game: CData | None = ffi.new("pyhanabi_game_t*")
        if params is None:
            lib.NewDefaultGame(self._game)
            return
        # Keep the encoded strings alive until NewGame has copied them.
        strings = [
            ffi.new("char[]", text.encode("ascii"))
            for key, value in params.items()
            for text in (key, str(value))
        ]
        c_array = ffi.new(f"char * [{len(strings)}]", strings)
        lib.NewGame(self._game, len(strings), c_array)

    @classmethod
    def _from_c_game(cls, c_game: CData) -> Self:
        game = cls.__new__(cls)
        game._game = c_game
        return game

    @property
    def c_game(self) -> CData:
        """The underlying engine game."""
        assert self._game is not None
        return self._game

    def parameter_string(self) -> str:
        """Returns all parameter values as a string."""
        return take_string(lib.GameParamString(self.c_game))

    def num_players(self) -> int:
        """Returns the number of players."""
        return int(lib.NumPlayers(self.c_game))

    def num_colors(self) -> int:
        """Returns the number of card colors."""
        return int(lib.NumColors(self.c_game))

    def num_ranks(self) -> int:
        """Returns the number of card ranks."""
        return int(lib.NumRanks(self.c_game))

    def hand_size(self) -> int:
        """Returns the maximum number of cards in a hand."""
        return int(lib.HandSize(self.c_game))

    def max_information_tokens(self) -> int:
        """Returns the initial (and maximum) number of information tokens."""
        return int(lib.MaxInformationTokens(self.c_game))

    def max_life_tokens(self) -> int:
        """Returns the initial number of life tokens."""
        return int(lib.MaxLifeTokens(self.c_game))

    def max_score(self) -> int:
        """Returns the score of a perfect game (every firework completed)."""
        return self.num_colors() * self.num_ranks()

    def observation_type(self) -> AgentObservationType:
        """Returns the observation type agents receive."""
        return AgentObservationType(lib.ObservationType(self.c_game))

    def max_moves(self) -> int:
        """Returns the number of distinct non-chance moves (the action space size)."""
        return int(lib.MaxMoves(self.c_game))

    def num_cards(self, color: int, rank: int) -> int:
        """Returns how many copies of (color, rank) the full deck contains."""
        return int(lib.NumCards(self.c_game, color, rank))

    def get_move_uid(self, move: HanabiMove) -> int:
        """Returns the move's id in [0, max_moves()), or -1 for chance moves."""
        return int(lib.GetMoveUid(self.c_game, move.c_move))

    def get_move(self, move_uid: int) -> HanabiMove:
        """Returns the move with the given id."""
        c_move = ffi.new("pyhanabi_move_t*")
        lib.GetMoveByUid(self.c_game, move_uid, c_move)
        return HanabiMove(c_move)

    def to_json(self) -> str:
        """Serializes the game, including its RNG state, to a JSON string."""
        c_string = lib.GameToJson(self.c_game)
        if c_string == ffi.NULL:
            raise SerializationError("GameToJson returned NULL")
        return take_string(c_string)

    @classmethod
    def from_json(cls, json_str: str) -> Self:
        """Deserializes a game produced by `to_json`."""
        c_game = ffi.new("pyhanabi_game_t*")
        if not lib.GameFromJson(json_arg(json_str), c_game):
            raise SerializationError("Cannot deserialize HanabiGame")
        return cls._from_c_game(c_game)

    def __del__(self) -> None:
        if self._game is not None:
            lib.DeleteGame(self._game)
            self._game = None
