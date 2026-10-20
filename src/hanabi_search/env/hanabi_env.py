"""Gym-like environment over the engine, producing dict observations for agents."""

from dataclasses import dataclass

from hanabi_search.env.cards import COLOR_CHAR, color_char_to_idx
from hanabi_search.env.game import GameParams, HanabiGame
from hanabi_search.env.moves import HanabiMove, HanabiMoveType
from hanabi_search.env.observation import (
    HanabiObservation,
    ObservationEncoder,
    ObservationEncoderType,
)
from hanabi_search.env.observation_types import (
    Action,
    AllPlayerObservations,
    MoveDict,
    ObservationDict,
)
from hanabi_search.env.state import CHANCE_PLAYER_ID, HanabiState
from hanabi_search.exceptions import IllegalMoveError

_MOVE_TYPE_NAMES = frozenset(move_type.name for move_type in HanabiMoveType)


@dataclass(frozen=True, slots=True)
class StepResult:
    """Outcome of `HanabiEnv.step`.

    Attributes:
        observations: Every player's observation after the move (and any deals).
        reward: Environment reward after the move (the game score by default).
        done: Whether the move ended the game.
    """

    observations: AllPlayerObservations
    reward: int
    done: bool


class HanabiEnv:
    """One game of Hanabi at a time, stepped by agent actions.

    Call `reset()` before stepping. (No state is created earlier: creating one
    draws a start player from the game's RNG, which would shift later deals.)

    Args:
        params: Game parameters (see `HanabiGame`). Include `seed` for reproducible deals.
    """

    state: HanabiState

    def __init__(self, params: GameParams) -> None:
        self.game = HanabiGame(params)
        self.num_players = self.game.num_players()
        self.observation_encoder = ObservationEncoder(self.game, ObservationEncoderType.CANONICAL)
        self._cards_per_rank = [
            self.game.num_cards(0, rank) for rank in range(self.game.num_ranks())
        ]

    def reset(self) -> AllPlayerObservations:
        """Starts a new game and deals the initial hands."""
        self.state = HanabiState(self.game)
        self._deal_pending_cards()
        return self.observe_all()

    def step(self, action: Action) -> StepResult:
        """Applies the acting player's action, then deals any pending cards.

        Args:
            action: A dict move, an engine move, or a move uid.

        Returns:
            The observations, reward and whether the game ended.

        Raises:
            IllegalMoveError: If the action is not legal.
        """
        self.state.apply_move(self.to_move(action))
        done = self.state.is_terminal()
        self._deal_pending_cards()
        return StepResult(self.observe_all(), self.reward(), done)

    def reward(self) -> int:
        """Returns the reward for the current state: the game score."""
        return self.state.score()

    def score(self) -> int:
        """Returns the game score (0 if all lives were lost)."""
        return self.state.score()

    def progress(self) -> int:
        """Returns the number of cards on the fireworks."""
        return self.state.progress()

    def num_moves(self) -> int:
        """Returns the size of the action space."""
        return self.game.max_moves()

    def vectorized_observation_shape(self) -> list[int]:
        """Returns the shape of `vectorized_observation` outputs."""
        return self.observation_encoder.shape()

    def vectorized_observation(self, observation: HanabiObservation) -> list[int]:
        """Encodes an engine observation as a bit vector."""
        return self.observation_encoder.encode(observation)

    def observe_all(self) -> AllPlayerObservations:
        """Returns every player's observation of the current state."""
        return {
            "current_player": self.state.cur_player(),
            "player_observations": [self.observe(player) for player in range(self.num_players)],
        }

    def observe(self, player: int) -> ObservationDict:
        """Returns one player's observation of the current state.

        Cheaper than `observe_all` when only one player's view is needed.

        Args:
            player: Absolute player id, in [0, num_players).
        """
        return self.observation_dict(self.state.observation(player))

    def observation_dict(self, observation: HanabiObservation) -> ObservationDict:
        """Converts an engine observation of the current state to dict form."""
        legal_moves = observation.legal_moves()
        fireworks = self.state.fireworks()
        return {
            "current_player": self.state.cur_player(),
            "current_player_offset": observation.cur_player_offset(),
            "life_tokens": observation.life_tokens(),
            "information_tokens": observation.information_tokens(),
            "max_information_tokens": self.game.max_information_tokens(),
            "num_players": observation.num_players(),
            "num_colors": self.game.num_colors(),
            "num_ranks": self.game.num_ranks(),
            "cards_per_rank": list(self._cards_per_rank),
            "deck_size": observation.deck_size(),
            "turns_to_play": self.state.turns_to_play(),
            "fireworks": dict(zip(COLOR_CHAR, fireworks, strict=False)),
            "legal_moves": [move.to_dict() for move in legal_moves],
            "legal_moves_as_int": [self.game.get_move_uid(move) for move in legal_moves],
            "observed_hands": [
                [card.to_dict() for card in hand] for hand in observation.observed_hands()
            ],
            "discard_pile": [card.to_dict() for card in observation.discard_pile()],
            "card_knowledge": [
                [knowledge.to_dict() for knowledge in hand] for hand in observation.card_knowledge()
            ],
            "pyhanabi": observation,
        }

    def to_move(self, action: Action) -> HanabiMove:
        """Converts an action to an engine move.

        Raises:
            IllegalMoveError: If a dict action is not legal in the current state.
        """
        if isinstance(action, HanabiMove):
            return action
        if isinstance(action, int):
            return self.game.get_move(action)
        return self.build_move(action)

    def build_move(self, action: MoveDict) -> HanabiMove:
        """Builds an engine move from a dict action and checks that it is legal.

        RETURN moves (used for re-determinization) are not checked.

        Raises:
            IllegalMoveError: If the action is malformed or not legal.
        """
        action_type = action["action_type"]
        if action_type not in _MOVE_TYPE_NAMES:
            raise IllegalMoveError(f"Unknown action_type {action_type!r} in {action}")
        try:
            match action_type:
                case "PLAY":
                    move = HanabiMove.get_play_move(action["card_index"])
                case "DISCARD":
                    move = HanabiMove.get_discard_move(action["card_index"])
                case "RETURN":
                    return HanabiMove.get_return_move(action["card_index"], action["player"])
                case "REVEAL_RANK":
                    move = HanabiMove.get_reveal_rank_move(action["target_offset"], action["rank"])
                case "REVEAL_COLOR":
                    color = action["color"]
                    if color is None:
                        raise IllegalMoveError(f"REVEAL_COLOR without a color: {action}")
                    move = HanabiMove.get_reveal_color_move(
                        action["target_offset"], color_char_to_idx(color)
                    )
                case _:
                    raise IllegalMoveError(f"Agents cannot take {action_type} actions")
        except KeyError as missing:
            raise IllegalMoveError(f"Action {action} is missing {missing}") from None
        if self.state.is_terminal() or not self.state.move_is_legal(move):
            raise IllegalMoveError(f"Illegal action {action} ({move}) in state:\n{self.state}")
        return move

    def _deal_pending_cards(self) -> None:
        while self.state.cur_player() == CHANCE_PLAYER_ID:
            self.state.deal_random_card()
