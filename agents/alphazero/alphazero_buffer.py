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
        self.values = np.zeros(capacity, dtype=np.float32)
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

    def _get_value_from_data(self, item: Tuple) -> float:
        """Extract value score from data tuple.
        Assumes item is (state, policy, value, root_policy)."""
        try:
            value = item[2]
            # Convert to float if it's a tensor
            if torch.is_tensor(value):
                return float(value.item())
            return float(value)
        except (IndexError, AttributeError) as e:
            self.logger.warning(f"Could not extract value from item: {e}")
            return 0.0
        
    def _find_lowest_value_index(self) -> int:
        """Find the index of the item with the lowest value score."""
        return int(np.argmin(self.values[:len(self.buffer)]))

    def add(self, data: List[Any]) -> None:
        """Add new experiences to the buffer, replacing lowest value items when full."""
        for item in data:
            current_value = self._get_value_from_data(item)
            
            if len(self.buffer) < self.capacity:
                self.buffer.append(item)
                self.values[self.pos] = current_value
                self.priorities[self.pos] = (
                    self.priorities.max() if len(self.buffer) > 1 else 1.0
                )
                self.pos = (self.pos + 1) % self.capacity
            else:
                lowest_value_idx = self._find_lowest_value_index()
                if current_value >= self.values[lowest_value_idx]:
                    self.buffer[lowest_value_idx] = item
                    self.values[lowest_value_idx] = current_value
                    self.priorities[lowest_value_idx] = self.priorities.max()
                    self.pos = (lowest_value_idx + 1) % self.capacity

        # Save to file if needed
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
