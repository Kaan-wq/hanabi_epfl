import random


class ReplayBuffer:
    def __init__(self, capacity):
        """Initialize the replay buffer."""
        self.capacity = capacity
        self.buffer = []
        self.file_path = "agents/alphazero/alphazero_data.txt"

        # Initialize or clear the file
        with open(self.file_path, 'w') as f:
            f.write('')  # Clear the file at the start

    def add(self, data):
        """Add data to the buffer."""
        self.buffer.extend(data)
        if len(self.buffer) > self.capacity:
            self.buffer = self.buffer[-self.capacity :]

        # Write the added data to the file
        with open(self.file_path, 'a') as f:
            for item in data:
                f.write(f"{item}\n")

    def sample(self, batch_size):
        """Sample a batch of data from the buffer."""
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.buffer)
