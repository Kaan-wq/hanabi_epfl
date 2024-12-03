import torch
import torch.nn as nn
from agents.alphazero.alphazero_agent import AlphaZero_Agent, AlphaZeroP_Agent
from agents.mcts.mcts_agent import MCTS_Agent, PMCTS_Agent
from torch import optim
from torch.utils.data import DataLoader, Dataset
from agents.alphazero.alphazero_buffer import ReplayBuffer


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
        self.state_vectors = [data[0] for data in training_data]
        self.policy_targets = [data[1] for data in training_data]
        self.value_targets = [data[2] for data in training_data]
        self.weights = weights if weights is not None else torch.ones(len(training_data))

    def __len__(self):
        return len(self.state_vectors)

    def __getitem__(self, idx):
        state = torch.as_tensor(self.state_vectors[idx], dtype=torch.float32)
        policy_target = torch.as_tensor(self.policy_targets[idx], dtype=torch.float32)
        value_target = torch.as_tensor(self.value_targets[idx], dtype=torch.float32)
        weight = torch.as_tensor(self.weights[idx], dtype=torch.float32)
        return state, policy_target, value_target, weight


def prepare_data(training_data, batch_size=16, weights=None):
    dataset = AlphaZeroDataset(training_data, weights)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader


def train_network(replay_buffer: ReplayBuffer, network: SimpleNetwork, optimizer, device, batch_size=128):
    """Train the network using data from the replay buffer."""

    batch_size = batch_size
    if len(replay_buffer) < batch_size:
        return None

    batch_data, indices, weights = replay_buffer.sample(batch_size)
    dataloader = prepare_data(batch_data, batch_size, weights)

    network.train()
    total_loss = 0.0
    steps = 0

    for states, policy_targets, value_targets, batch_weights in dataloader:
        states = states.to(device)
        policy_targets = policy_targets.to(device)
        value_targets = value_targets.to(device)
        batch_weights = batch_weights.to(device)

        optimizer.zero_grad()
        policy_logits = network(states)
        policy_log_probs = nn.functional.log_softmax(policy_logits, dim=1)
        
        policy_losses = -torch.sum(policy_targets * policy_log_probs, dim=1)
        policy_loss = torch.mean(policy_losses * batch_weights)

        # Value loss (if value head is added)
        # value_loss = self.criterion_value(value.squeeze(-1), value_targets)

        # Total loss
        loss = policy_loss  # + value_loss

        loss.backward()
        optimizer.step()

        # Update priorities with policy losses
        replay_buffer.update_priorities(indices, policy_losses.detach().cpu().numpy())

        total_loss += loss.item()
        steps += 1

    avg_loss = total_loss / steps if steps > 0 else 0.0
    return avg_loss


# ========================= Helper Functions =========================
def requires_mcts_data(agent_classes, record_data=False):
    """Check if any agent requires data collection."""
    data_collection_agents = (MCTS_Agent, PMCTS_Agent)
    return any(
        issubclass(agent_class, data_collection_agents) for agent_class in agent_classes
    ) and record_data

def requires_training(agent_classes):
    """Check if any agent requires training."""
    training_agents = (AlphaZero_Agent, AlphaZeroP_Agent)
    return any(
        issubclass(agent_class, training_agents) for agent_class in agent_classes
    )


def initialize_training_components(
    env, device, from_pretrained=None, lr=1e-4, weight_decay=1e-4
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


def collect_alphazero_data(agents, replay_buffer, final_score):
    """Collect training data from agents that require training."""

    z = 2 * (final_score / 25) - 1  # Value target

    for agent in agents:
        if not hasattr(agent, "training_data"):
            continue

        for i, data in enumerate(agent.training_data):
            (
                state_vector,
                policy_targets,
                _,
                root_policy,
            ) = data
            state_vector = torch.tensor(state_vector, dtype=torch.float32)
            policy_targets = torch.tensor(policy_targets, dtype=torch.float32)
            value_target = torch.tensor(z, dtype=torch.float32)
            root_policy = torch.tensor(root_policy, dtype=torch.float32)

            agent.training_data[i] = (
                state_vector,
                policy_targets,
                value_target,
                root_policy,
            )

        replay_buffer.add(agent.training_data)
        agent.training_data.clear()

def collect_mcts_data(agents, replay_buffer, final_score):
    """Collect training data from agents."""

    z = 2 * (final_score / 25) - 1  # Value target

    for agent in agents:
        if not hasattr(agent, "training_data"):
            continue

        for i, data in enumerate(agent.training_data):
            (
                state_vector,
                policy_targets,
                value_targets
            ) = data
            state_vector = torch.tensor(state_vector, dtype=torch.float32)
            policy_targets = torch.tensor(policy_targets, dtype=torch.float32)
            value_targets = torch.tensor(value_targets, dtype=torch.float32)
            value = torch.tensor(z, dtype=torch.float32)

            agent.training_data[i] = (
                state_vector,
                policy_targets,
                value_targets, 
                value
            )

        replay_buffer.add(agent.training_data)
        agent.training_data.clear()
# ======================================================================

### Need memeory buffer (batch-size of 128) only after say 4 episodes to fill the buffer
### Forget value-head and only pure MCTS for uct formula
### Try pretraining