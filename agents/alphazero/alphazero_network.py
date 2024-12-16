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
    def __init__(self, num_actions, obs_shape, hidden_sizes=[512, 256, 256]):
        super(SimpleNetwork, self).__init__()
        
        # Improved shared backbone
        backbone_layers = []
        prev_size = obs_shape
        
        for hidden_size in hidden_sizes:
            backbone_layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.GELU(),
            ])
            prev_size = hidden_size
            
        self.backbone = nn.Sequential(*backbone_layers)
        
        # Policy head
        self.policy_head = nn.Sequential(
            nn.Linear(hidden_sizes[-1], hidden_sizes[-1]),
            nn.GELU(),
            nn.Linear(hidden_sizes[-1], num_actions)
        )
        
        # Value head
        self.value_head = nn.Sequential(
            nn.Linear(hidden_sizes[-1], hidden_sizes[-1] // 2),
            nn.GELU(),
            nn.Linear(hidden_sizes[-1] // 2, 1),
            nn.Tanh()  # Bound values between -1 and 1
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights using He initialization"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, nonlinearity='relu')
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, x):
        shared_features = self.backbone(x)
        policy_logits = self.policy_head(shared_features)
        value = self.value_head(shared_features)
        return policy_logits, value
   

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
    value_loss_weight=10.0,
):
    """Train the network using data from the replay buffer, including both policy and value heads."""
    
    if len(replay_buffer) < batch_size:
        return None

    batch_data, indices, weights = replay_buffer.sample(batch_size)
    dataloader = prepare_data(batch_data, batch_size, weights)

    network.train()
    total_policy_loss = 0.0
    total_value_loss = 0.0
    total_loss = 0.0

    for states, policies, values, weights in dataloader:
        states, policies, values, weights = (
            states.to(device),
            policies.to(device),
            values.to(device),
            weights.to(device),
        )

        optimizer.zero_grad()
        
        # Forward pass through both heads
        policy_logits, predicted_values = network(states)
        
        # Policy loss calculation
        policy_log_probs = nn.functional.log_softmax(policy_logits, dim=1)
        policy_loss = torch.mean(-torch.sum(policies * policy_log_probs, dim=1) * weights)
        
        # Value loss calculation with increased weight
        value_loss = torch.mean(nn.functional.mse_loss(
            predicted_values.squeeze(-1), 
            values,
            reduction='none'
        ) * weights)

        # Combined loss with weighting
        loss = policy_loss + value_loss_weight * value_loss
        
        loss.backward()
        optimizer.step()

        # Update priorities based on both policy and value errors
        policy_errors = -torch.sum(policies * policy_log_probs, dim=1)
        value_errors = torch.abs(predicted_values.squeeze(-1) - values)
        priorities = (policy_errors + value_loss_weight * value_errors).detach()
        replay_buffer.update_priorities(indices, priorities.cpu().numpy())
        
        # Track losses
        total_policy_loss += policy_loss.item()
        total_value_loss += value_loss.item()
        total_loss += loss.item()

    # Calculate averages
    num_batches = len(dataloader)
    avg_policy_loss = total_policy_loss / num_batches
    avg_value_loss = total_value_loss / num_batches
    avg_total_loss = total_loss / num_batches

    return {
        'policy_loss': avg_policy_loss,
        'value_loss': avg_value_loss,
        'total_loss': avg_total_loss,
        'effective_policy_contribution': avg_policy_loss / avg_total_loss,
        'effective_value_contribution': (value_loss_weight * avg_value_loss) / avg_total_loss,
    }


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
                root_policy.clone().detach() if torch.is_tensor(root_policy) else torch.tensor(root_policy, dtype=torch.float32),
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
                torch.tensor(z, dtype=torch.float32),
            )
            for state, policy in zip(*data)
        ]

        replay_buffer.add(processed_data)
        agent.training_data.clear()


# ======================================================================

# Hyperparameter tuning
