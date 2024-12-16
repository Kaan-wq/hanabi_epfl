import json
import logging
import os
import random
from typing import Any, List, Optional, Tuple

import numpy as np
import torch


class PrioritizedReplayBuffer:
    """Prioritized Experience Replay buffer with flexible storage. (memory, file, or hybrid)"""

    def __init__(
        self,
        capacity: int = 10000,
        storage_mode: str = "hybrid",
        file_path: Optional[str] = None,
        load_existing: bool = False,
        alpha: float = 0.7,  # Priority exponent
        beta: float = 0.4,  # Importance sampling
        log_level: int = logging.INFO,
    ):
        """Initialize buffer with specified capacity and storage configuration."""
        self.capacity = capacity
        self.storage_mode = storage_mode
        self.alpha = alpha
        self.beta = beta
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(log_level)

        self.buffer: List[Any] = []
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.pos = 0

        self.file_path = file_path or os.path.join("saved_data", "replay_buffer.jsonl")
        os.makedirs(os.path.dirname(self.file_path), exist_ok=True)

        if load_existing and os.path.exists(self.file_path):
            self.buffer = self.load_from_file()
        elif storage_mode in ["file", "hybrid"]:
            self._prepare_file_storage()

    def _prepare_file_storage(self) -> None:
        """Initialize or clear the storage file."""
        try:
            with open(self.file_path, "w") as f:
                f.write("")
        except IOError as e:
            self.logger.error(f"Failed to prepare file storage: {e}")

    def add(self, data: List[Any]) -> None:
        """Add new experiences to the buffer."""
        for item in data:
            if len(self.buffer) < self.capacity:
                self.buffer.append(item)
            else:
                self.buffer[self.pos] = item

            self.priorities[self.pos] = (
                self.priorities.max() if len(self.buffer) > 1 else 1.0
            )
            self.pos = (self.pos + 1) % self.capacity

        if self.storage_mode in ["file", "hybrid"]:
            self._save_to_file(data)

    def _save_to_file(self, data: List[Any]) -> None:
        """Save data to file with JSON serialization."""
        try:
            with open(self.file_path, "a") as f:
                json.dump([self._make_serializable(item) for item in data], f)
                f.write("\n")
        except Exception as e:
            self.logger.error(f"Error saving to file: {e}")

    def _make_serializable(self, item: Any) -> Any:
        """Convert tensors and tuples to JSON-serializable format."""
        if isinstance(item, torch.Tensor):
            return item.tolist()
        elif isinstance(item, tuple):
            return [self._make_serializable(x) for x in item]
        return item

    def load_from_file(self) -> List[Any]:
        """Load saved experiences from file."""
        try:
            with open(self.file_path, "r") as f:
                return [json.loads(line) for line in f]
        except FileNotFoundError:
            self.logger.warning(f"File {self.file_path} not found")
            return []

    def sample(self, batch_size: int) -> Tuple[List[Any], np.ndarray, np.ndarray]:
        """Sample batch with priorities and importance sampling weights."""
        if len(self.buffer) < batch_size:
            return self.buffer, None, None

        # Compute sampling probabilities
        probs = self.priorities[: len(self.buffer)] ** self.alpha
        probs /= probs.sum()

        # Sample indices based on priorities
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)

        # Compute importance sampling weights
        weights = (len(self.buffer) * probs[indices]) ** -self.beta
        weights /= weights.max()

        samples = [self.buffer[idx] for idx in indices]
        return samples, indices, weights

    def update_priorities(self, indices: np.ndarray, errors: np.ndarray) -> None:
        """Update priorities based on TD errors."""
        self.priorities[indices] = (np.abs(errors) + 1e-6) ** self.alpha
        self.beta = min(1.0, self.beta + 1e-4)

    def __len__(self) -> int:
        """Return current buffer size."""
        return len(self.buffer)

    def clear(self) -> None:
        """Clear all stored data."""
        self.buffer.clear()
        if self.storage_mode in ["file", "hybrid"]:
            self._prepare_file_storage()


class QualityPrioritizedBuffer(PrioritizedReplayBuffer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.values = []  # Track value/score for each entry
        self.value_threshold = float("-inf")  # Dynamic threshold

    def add(self, data: List[Any], value: float = None) -> None:
        """Add new experiences if they meet quality threshold."""
        if value is None:
            value = self._extract_value_from_data(data[0])

        # Update threshold if buffer is not empty
        if self.values:
            self.value_threshold = np.mean(self.values)

        # Only add data if it's better than threshold
        if value >= self.value_threshold:
            # Find worst performing samples to replace if buffer is full
            if len(self.buffer) >= self.capacity:
                worst_indices = np.argsort(self.values)[: len(data)]
                for idx, item in zip(worst_indices, data):
                    self.buffer[idx] = item
                    self.values[idx] = value
                    self.priorities[idx] = self.priorities.max()
            else:
                # Add new data
                for item in data:
                    self.buffer.append(item)
                    self.values.append(value)
                    self.priorities[self.pos] = (
                        self.priorities.max() if len(self.buffer) > 1 else 1.0
                    )
                    self.pos = (self.pos + 1) % self.capacity

            if self.storage_mode in ["file", "hybrid"]:
                self._save_to_file(data)

    def _extract_value_from_data(self, data_point):
        """Extract value from a data point (assuming it's in the last position)."""
        return data_point[2] if len(data_point) >= 3 else 0.0

    def load_mcts_data(self, file_path: str) -> None:
        """Load initial MCTS data."""
        try:
            with open(file_path, "r") as f:
                for line in f:
                    data = json.loads(line)
                    value = self._extract_value_from_data(data[0])
                    self.add(data, value)
            print(f"Loaded {len(self.buffer)} experiences from MCTS data")
            print(f"Initial value threshold: {self.value_threshold:.3f}")
        except FileNotFoundError:
            print(f"No MCTS data found at {file_path}")


def configure_replay_buffer(
    capacity: int = 10000,
    storage_mode: str = "hybrid",
    file_path: str = "saved_data/replay_buffer.jsonl",
    load_existing: bool = False,
    alpha: float = 0.6,
    beta: float = 0.4,
) -> PrioritizedReplayBuffer:
    """Create a configured prioritized replay buffer instance."""
    return PrioritizedReplayBuffer(
        capacity=capacity,
        storage_mode=storage_mode,
        file_path=file_path,
        load_existing=load_existing,
        alpha=alpha,
        beta=beta,
    )


def configure_quality_buffer(
    capacity: int = 10000,
    storage_mode: str = "hybrid",
    file_path: str = "saved_data/replay_buffer.jsonl",
    mcts_data_path: str = None,
    load_existing: bool = False,
    alpha: float = 0.6,
    beta: float = 0.4,
) -> QualityPrioritizedBuffer:
    """Create a configured quality-based prioritized replay buffer."""
    buffer = QualityPrioritizedBuffer(
        capacity=capacity,
        storage_mode=storage_mode,
        file_path=file_path,
        load_existing=load_existing,
        alpha=alpha,
        beta=beta,
    )

    if mcts_data_path:
        buffer.load_mcts_data(mcts_data_path)

    return buffer
