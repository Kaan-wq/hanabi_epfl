import copy

import torch
import torch.nn as nn
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
        #value = self.value_head(x)

        return policy_logits #, value


class AlphaZeroDataset(Dataset):
    def __init__(self, training_data):
        self.state_vectors = [data[0] for data in training_data]
        self.policy_targets = [data[1] for data in training_data]
        self.value_targets = [data[2] for data in training_data]

    def __len__(self):
        return len(self.state_vectors)

    def __getitem__(self, idx):
        state = torch.as_tensor(self.state_vectors[idx], dtype=torch.float32)
        policy_target = torch.as_tensor(self.policy_targets[idx], dtype=torch.float32)
        value_target = torch.as_tensor(self.value_targets[idx], dtype=torch.float32)
        return state, policy_target, value_target


def prepare_data(training_data, batch_size=16):
    dataset = AlphaZeroDataset(training_data)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader


def extract_tensors(model):
    """
    Remove the tensors from a PyTorch model, convert them to NumPy
    arrays, and return the stripped model and tensors.
    """
    tensors = []
    for _, module in model.named_modules():
        # Store the tensors in Python dictionaries
        params = {
            name: torch.clone(param).detach().numpy()
            for name, param in module.named_parameters(recurse=False)
        }
        buffers = {
            name: torch.clone(buf).detach().numpy()
            for name, buf in module.named_buffers(recurse=False)
        }
        tensors.append({"params": params, "buffers": buffers})

    # Make a copy of the original model and strip all tensors and
    # buffers out of the copy.
    m_copy = copy.deepcopy(model)
    for _, module in m_copy.named_modules():
        for name in [name for name, _ in module.named_parameters(recurse=False)] + [
            name for name, _ in module.named_buffers(recurse=False)
        ]:
            setattr(module, name, None)

    # Make sure the copy is configured for inference.
    m_copy.train(False)
    return m_copy, tensors


def replace_tensors(model, tensors):
    """
    Restore the tensors that extract_tensors() stripped out of a
    PyTorch model.
    """
    modules = [module for _, module in model.named_modules()]
    for module, tensor_dict in zip(modules, tensors):
        # There are separate APIs to set parameters and buffers.
        for name, array in tensor_dict["params"].items():
            module.register_parameter(name, torch.nn.Parameter(torch.as_tensor(array)))
        for name, array in tensor_dict["buffers"].items():
            module.register_buffer(name, torch.as_tensor(array))


### Need memeory buffer (batch-size of 128) only after say 4 episodes to fill the buffer
### Forget value-head and only pure MCTS for uct formula
### Try pretraining
