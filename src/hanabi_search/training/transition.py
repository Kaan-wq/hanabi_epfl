"""Training examples produced by search agents."""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

type Vector = npt.NDArray[np.float32]


@dataclass(frozen=True, slots=True)
class Transition:
    """One search decision, labelled with the game's outcome.

    Attributes:
        observation: Encoded observation of the acting player, shape (observation_size,).
        policy: Normalized root visit counts over the action space, shape (num_actions,).
        value: Outcome target in [-1, 1]: `2 * score / max_score - 1`.
    """

    observation: Vector
    policy: Vector
    value: float


def value_target(final_score: int, max_score: int) -> float:
    """Maps a final score in [0, max_score] to a value target in [-1, 1]."""
    return 2 * (final_score / max_score) - 1


@dataclass(slots=True)
class PendingDecision:
    """A search decision whose outcome is not known yet.

    Attributes:
        observation: Encoded observation, shape (observation_size,).
        policy: Normalized root visit counts, shape (num_actions,).
    """

    observation: Vector
    policy: Vector

    def labelled(self, value: float) -> Transition:
        """Returns the decision as a training example with outcome `value`."""
        return Transition(self.observation, self.policy, value)


def visit_policy(visits: dict[int, float], num_actions: int) -> Vector:
    """Normalizes visit counts per move uid to a policy; uniform if there are no visits.

    Args:
        visits: Visit count per move uid.
        num_actions: Size of the action space.

    Returns:
        Policy target, shape (num_actions,).
    """
    policy = np.zeros(num_actions, dtype=np.float32)
    for uid, count in visits.items():
        policy[uid] = count
    total = policy.sum()
    if total > 0:
        return policy / total
    return np.full(num_actions, 1.0 / num_actions, dtype=np.float32)


# Receives the labelled examples of one finished game.
type TransitionSink = Callable[[list[Transition]], None]
