import json
import logging
import os
import random
from typing import Any, List, Optional

import torch


class ReplayBuffer:
    """Replay buffer with flexible storage modes (memory, file, hybrid) and data persistence."""

    def __init__(
        self,
        capacity: int = 10000,
        storage_mode: str = "hybrid",
        file_path: Optional[str] = None,
        load_existing: bool = False,
        log_level: int = logging.INFO,
    ):
        """Initialize buffer with specified capacity and storage configuration."""
        self.capacity = capacity
        self.storage_mode = storage_mode
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(log_level)

        self.buffer: List[Any] = []
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
        """Add new experiences to buffer while maintaining capacity limit."""
        self.buffer.extend(data)

        if len(self.buffer) > self.capacity:
            self.buffer = self.buffer[-self.capacity :]

        if self.storage_mode in ["file", "hybrid"]:
            self._save_to_file(data)

    def _save_to_file(self, data: List[Any]) -> None:
        """Save data to file with JSON serialization."""
        try:
            with open(self.file_path, "a") as f:
                for item in data:
                    serializable_item = self._make_serializable(item)
                    json.dump(serializable_item, f)
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

    def sample(self, batch_size: int) -> List[Any]:
        """Sample random batch of experiences."""
        if len(self.buffer) < batch_size:
            self.logger.warning(
                f"Requested batch size {batch_size} exceeds buffer size"
            )
            return self.buffer
        return random.sample(self.buffer, batch_size)

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
) -> ReplayBuffer:
    """Create a configured replay buffer instance."""
    return ReplayBuffer(
        capacity=capacity,
        storage_mode=storage_mode,
        file_path=file_path,
        load_existing=load_existing,
    )
