"""Prioritized experience replay over `Transition`s."""

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt

from hanabi_search.training.transition import Transition

logger = logging.getLogger(__name__)

_MIN_PRIORITY = 1e-6
_BETA_INCREMENT = 1e-4


@dataclass(frozen=True, slots=True)
class SampledBatch:
    """Transitions drawn from the buffer.

    Attributes:
        transitions: The sampled transitions.
        indices: Their positions in the buffer, shape (batch,).
        weights: Importance-sampling weights in (0, 1], shape (batch,).
    """

    transitions: list[Transition]
    indices: npt.NDArray[np.int64]
    weights: npt.NDArray[np.float32]


class PrioritizedReplayBuffer:
    """Ring buffer sampling transitions in proportion to `priority ** alpha`.

    Priorities are stored raw (training errors); `alpha` is applied once, when
    sampling. New transitions get the current maximum priority so they are seen soon.

    Args:
        capacity: Maximum number of transitions; the oldest are overwritten.
        alpha: Priority exponent (0 is uniform sampling).
        beta: Initial importance-sampling exponent; annealed towards 1.
        rng: Random generator used for sampling.
        save_path: If given, every added transition is appended to this JSONL file.
    """

    def __init__(
        self,
        capacity: int,
        alpha: float,
        beta: float,
        rng: np.random.Generator,
        save_path: Path | None = None,
    ) -> None:
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self._rng = rng
        self._transitions: list[Transition] = []
        self._priorities = np.zeros(capacity, dtype=np.float32)
        self._position = 0
        self._max_priority = 1.0
        self._save_path = save_path
        if save_path is not None:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            save_path.write_text("")

    def add(self, transitions: list[Transition]) -> None:
        """Adds transitions, overwriting the oldest once full."""
        for transition in transitions:
            if len(self._transitions) < self.capacity:
                self._transitions.append(transition)
            else:
                self._transitions[self._position] = transition
            self._priorities[self._position] = self._max_priority
            self._position = (self._position + 1) % self.capacity
        if self._save_path is not None:
            self._append_to_file(transitions)

    def sample(self, batch_size: int) -> SampledBatch:
        """Samples `batch_size` transitions (with replacement) by priority.

        Raises:
            ValueError: If the buffer holds fewer than `batch_size` transitions.
        """
        size = len(self._transitions)
        if size < batch_size:
            raise ValueError(f"Cannot sample {batch_size} from a buffer of {size}")
        probabilities = self._priorities[:size].astype(np.float64) ** self.alpha
        probabilities /= probabilities.sum()
        indices = self._rng.choice(size, batch_size, p=probabilities)
        weights = (size * probabilities[indices]) ** -self.beta
        weights /= weights.max()
        return SampledBatch(
            transitions=[self._transitions[i] for i in indices],
            indices=indices.astype(np.int64),
            weights=weights.astype(np.float32),
        )

    def update_priorities(
        self, indices: npt.NDArray[np.int64], errors: npt.NDArray[np.float32]
    ) -> None:
        """Sets raw priorities from per-sample errors and anneals beta towards 1.

        Args:
            indices: Buffer positions, shape (batch,).
            errors: Training errors of those transitions, shape (batch,).
        """
        priorities = np.abs(errors) + _MIN_PRIORITY
        self._priorities[indices] = priorities
        self._max_priority = max(self._max_priority, float(priorities.max()))
        self.beta = min(1.0, self.beta + _BETA_INCREMENT)

    def __len__(self) -> int:
        return len(self._transitions)

    def _append_to_file(self, transitions: list[Transition]) -> None:
        assert self._save_path is not None
        with self._save_path.open("a") as file:
            for transition in transitions:
                record = {
                    "observation": transition.observation.tolist(),
                    "policy": transition.policy.tolist(),
                    "value": transition.value,
                }
                file.write(json.dumps(record) + "\n")
