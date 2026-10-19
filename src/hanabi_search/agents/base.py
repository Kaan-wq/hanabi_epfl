"""Agent interface shared by all policies."""

from abc import ABC, abstractmethod

from hanabi_search.config import GameConfig
from hanabi_search.env import Action, HanabiState, ObservationDict


class Agent(ABC):
    """A policy for one seat at the table.

    Every agent is called on every turn with its own observation; it returns an
    action on its own turn and `None` otherwise. Search agents also receive the
    true state, which they may only use through re-determinization (never to read
    their own hidden cards).

    Args:
        player_id: Absolute seat of this agent.
        game: The game being played.
    """

    def __init__(self, player_id: int, game: GameConfig) -> None:
        self.player_id = player_id
        self.game = game

    @abstractmethod
    def act(self, observation: ObservationDict, state: HanabiState) -> Action | None:
        """Returns this agent's action, or `None` if it is not this agent's turn.

        Args:
            observation: This agent's observation.
            state: The true game state (see class docstring for allowed uses).
        """

    def end_episode(self, final_score: int) -> None:  # noqa: B027 -- optional hook
        """Called once per finished game; override to record training data."""
