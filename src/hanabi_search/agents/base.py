"""Agent interface shared by all policies."""

import random
from abc import ABC, abstractmethod
from collections.abc import Callable, Generator

import numpy as np
import numpy.typing as npt

from hanabi_search.config import GameConfig
from hanabi_search.env import Action, HanabiState, ObservationDict
from hanabi_search.exceptions import HanabiSearchError

# A position a search agent needs the network to evaluate: its encoded
# observation as 0/1 values, shape (observation_size,).
type EvaluationRequest = npt.NDArray[np.uint8]
# The network's answer: move probabilities, shape (num_actions,), and a value in [-1, 1].
type Evaluation = tuple[npt.NDArray[np.float32], float]
# A decision in progress: yields requests, receives their evaluations, returns the action.
type ActSteps = Generator[EvaluationRequest, Evaluation, Action | None]


def run_steps(
    steps: ActSteps, evaluate: Callable[[EvaluationRequest], Evaluation]
) -> Action | None:
    """Runs a decision to completion, answering each request with `evaluate`."""
    try:
        request = next(steps)
        while True:
            request = steps.send(evaluate(request))
    except StopIteration as finished:
        action: Action | None = finished.value
        return action


class Agent(ABC):
    """A policy for one seat at the table.

    Every agent is called on every turn with its own observation; it returns an
    action on its own turn and `None` otherwise. Search agents also receive the
    true state, which they may only use through re-determinization (never to read
    their own hidden cards).

    All of an agent's randomness comes from `rng`, the generator of the game it is
    playing, which `begin_episode` hands over. A game is then a function of its
    seed alone, even when several games are played interleaved.

    Args:
        player_id: Absolute seat of this agent.
        game: The game being played.
    """

    def __init__(self, player_id: int, game: GameConfig) -> None:
        self.player_id = player_id
        self.game = game
        self.rng = random.Random()

    def begin_episode(self, rng: random.Random) -> None:
        """Called before each game with the game's random generator."""
        self.rng = rng

    @abstractmethod
    def act(self, observation: ObservationDict, state: HanabiState) -> Action | None:
        """Returns this agent's action, or `None` if it is not this agent's turn.

        Args:
            observation: This agent's observation.
            state: The true game state (see class docstring for allowed uses).
        """

    def act_steps(self, observation: ObservationDict, state: HanabiState) -> ActSteps:
        """Like `act`, but yields every position it needs the network to evaluate.

        The caller sends back each position's `Evaluation`, so it can batch the
        requests of many games into one forward pass. Agents without a network
        yield nothing.
        """
        yield from ()
        return self.act(observation, state)

    def evaluate(self, request: EvaluationRequest) -> Evaluation:
        """Answers one of this agent's evaluation requests by itself (batch of one).

        Raises:
            HanabiSearchError: If the agent has no network (it never makes requests).
        """
        raise HanabiSearchError(
            f"{type(self).__name__} has no network; unexpected request {request.shape}"
        )

    def end_episode(self, final_score: int) -> None:  # noqa: B027 -- optional hook
        """Called once per finished game; override to record training data."""
