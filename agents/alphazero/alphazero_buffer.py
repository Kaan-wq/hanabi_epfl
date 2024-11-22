import random


class ReplayBuffer:
    def __init__(self, capacity):
        """Initialize the replay buffer."""
        self.capacity = capacity
        self.buffer = []

    def add(self, data):
        """Add data to the buffer."""
        self.buffer.extend(data)
        if len(self.buffer) > self.capacity:
            self.buffer = self.buffer[-self.capacity :]

    def sample(self, batch_size):
        """Sample a batch of data from the buffer."""
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.buffer)
