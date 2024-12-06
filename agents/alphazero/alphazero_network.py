from typing import List

import torch
import torch.nn as nn
from agents.alphazero.alphazero_agent import AlphaZero_Agent, AlphaZeroP_Agent
from agents.alphazero.alphazero_buffer import PrioritizedReplayBuffer
from agents.mcts.mcts_agent import MCTS_Agent, PMCTS_Agent
from rl_env import HanabiEnv
from torch import optim
from torch.utils.data import DataLoader, Dataset


class SimpleNetwork(nn.Module):
    def __init__(self, num_actions, obs_shape, hidden_size=256):
        super(SimpleNetwork, self).__init__()

        # Shared layers
        self.fc_shared = nn.Sequential(
            nn.Linear(obs_shape, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
        )

        # Policy head
        self.policy_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_actions),
        )

        # Value head
        self.value_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
            nn.Tanh(),
        )

    def forward(self, x):
        # Shared layers
        x = self.fc_shared(x)

        # Policy head
        policy_logits = self.policy_head(x)

        # Value head
        # value = self.value_head(x)

        return policy_logits  # , value


class AlphaZeroDataset(Dataset):
    def __init__(self, training_data, weights=None):
        self.states = torch.stack(
            [torch.as_tensor(d[0], dtype=torch.float32) for d in training_data]
        )
        self.policies = torch.stack(
            [torch.as_tensor(d[1], dtype=torch.float32) for d in training_data]
        )
        self.values = torch.stack(
            [torch.as_tensor(d[2], dtype=torch.float32) for d in training_data]
        )
        self.weights = torch.as_tensor(
            weights if weights is not None else torch.ones(len(training_data))
        )

    def __getitem__(self, idx):
        return self.states[idx], self.policies[idx], self.values[idx], self.weights[idx]

    def __len__(self):
        return len(self.states)


def prepare_data(training_data, batch_size=16, weights=None) -> DataLoader:
    dataset = AlphaZeroDataset(training_data, weights)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader


def train_network(
    replay_buffer: PrioritizedReplayBuffer,
    network: SimpleNetwork,
    optimizer: optim.Optimizer,
    device,
    batch_size=128,
):
    """Train the network using data from the replay buffer."""

    batch_size = batch_size
    if len(replay_buffer) < batch_size:
        return None

    batch_data, indices, weights = replay_buffer.sample(batch_size)
    dataloader = prepare_data(batch_data, batch_size, weights)

    network.train()
    total_loss = 0.0

    for states, policies, values, weights in dataloader:
        states, policies, weights = (
            states.to(device),
            policies.to(device),
            weights.to(device),
        )

        optimizer.zero_grad()
        policy_logits = network(states)
        policy_log_probs = nn.functional.log_softmax(policy_logits, dim=1)

        policy_losses = -torch.sum(policies * policy_log_probs, dim=1)
        policy_loss = torch.mean(policy_losses * weights)

        # Value loss (if value head is added)
        # value_loss = self.criterion_value(value.squeeze(-1), value_targets)
        # Total loss
        # loss = policy_loss + value_loss

        policy_loss.backward()
        optimizer.step()

        replay_buffer.update_priorities(indices, policy_losses.detach().cpu().numpy())
        total_loss += policy_loss.item()

    return total_loss / len(dataloader)


# ========================= Helper Functions =========================
def requires_mcts_data(agent_classes, record_data=False):
    """Check if any agent requires data collection."""
    data_collection_agents = (MCTS_Agent, PMCTS_Agent)
    return (
        any(
            issubclass(agent_class, data_collection_agents)
            for agent_class in agent_classes
        )
        and record_data
    )


def requires_training(agent_classes):
    """Check if any agent requires training."""
    training_agents = (AlphaZero_Agent, AlphaZeroP_Agent)
    return any(
        issubclass(agent_class, training_agents) for agent_class in agent_classes
    )


def initialize_training_components(
    env: HanabiEnv, device, from_pretrained=None, lr=1e-4, weight_decay=1e-4
):
    """Initialize network and optimizer for training."""

    num_actions = env.num_moves()
    obs_shape = env.vectorized_observation_shape()[0]
    network = SimpleNetwork(num_actions, obs_shape)
    network.to(device)

    if from_pretrained is not None:
        network.load_state_dict(torch.load(from_pretrained, map_location=device))

    optimizer = optim.AdamW(network.parameters(), lr=lr, weight_decay=weight_decay)
    criterion_value = nn.MSELoss()

    return network, optimizer, criterion_value, num_actions


def collect_alphazero_data(
    agents: List[AlphaZero_Agent],
    replay_buffer: PrioritizedReplayBuffer,
    final_score: int,
) -> None:
    """Collect training data from agents that require training."""
    z = 2 * (final_score / 25) - 1  # Value target

    for agent in agents:
        if not hasattr(agent, "training_data") or not agent.training_data:
            continue

        data = list(zip(*agent.training_data))
        processed_data = [
            (
                torch.tensor(state, dtype=torch.float32),
                torch.tensor(policy, dtype=torch.float32),
                torch.tensor(z, dtype=torch.float32),
                torch.tensor(root_policy, dtype=torch.float32),
            )
            for state, policy, value, root_policy in zip(*data)
        ]

        replay_buffer.add(processed_data)
        agent.training_data.clear()


def collect_mcts_data(
    agents: List[MCTS_Agent], replay_buffer: PrioritizedReplayBuffer, final_score: int
) -> None:
    """Collect and process MCTS training data from agents."""
    z = 2 * (final_score / 25) - 1

    for agent in agents:
        if not hasattr(agent, "training_data") or not agent.training_data:
            continue

        data = list(zip(*agent.training_data))
        processed_data = [
            (
                torch.tensor(state, dtype=torch.float32),
                torch.tensor(policy, dtype=torch.float32),
                torch.tensor(value, dtype=torch.float32),
                torch.tensor(z, dtype=torch.float32),
            )
            for state, policy, value in zip(*data)
        ]

        replay_buffer.add(processed_data)
        agent.training_data.clear()


# ======================================================================

# KL divergence loss
# Value head
# Hyperparameter tuning
# Training after each action
