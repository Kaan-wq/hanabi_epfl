"""AlphaZero training: policy cross-entropy plus weighted value MSE, on prioritized replay."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn

from hanabi_search.config import NetworkConfig, TrainingConfig
from hanabi_search.exceptions import NonFiniteLossError
from hanabi_search.networks import AlphaZeroNetwork
from hanabi_search.training.replay_buffer import PrioritizedReplayBuffer


@dataclass(frozen=True, slots=True)
class LossStats:
    """Losses of one training step.

    Attributes:
        policy_loss: Weighted cross-entropy between visit policy and network policy.
        value_loss: Weighted squared error of the value head (unscaled).
        total_loss: `policy_loss + value_loss_weight * value_loss`.
        value_loss_weight: Weight of the value loss in `total_loss`.
    """

    policy_loss: float
    value_loss: float
    total_loss: float
    value_loss_weight: float

    @property
    def policy_share(self) -> float:
        """Fraction of the total loss coming from the policy term."""
        return self.policy_loss / self.total_loss

    @property
    def value_share(self) -> float:
        """Fraction of the total loss coming from the weighted value term."""
        return self.value_loss_weight * self.value_loss / self.total_loss


def build_network(
    config: NetworkConfig, num_actions: int, observation_size: int, device: torch.device
) -> AlphaZeroNetwork:
    """Creates the network on `device`, loading `config.checkpoint` if set."""
    network = AlphaZeroNetwork(num_actions, observation_size, config.hidden_sizes).to(device)
    if config.checkpoint is not None:
        network.load_state_dict(
            torch.load(config.checkpoint, map_location=device, weights_only=True)
        )
    return network


class AlphaZeroTrainer:
    """Trains a network from a prioritized replay buffer.

    Args:
        network: Network to train (shared with the playing agents).
        buffer: Source of training transitions.
        config: Optimization hyperparameters.
        device: Device of `network`.
    """

    def __init__(
        self,
        network: AlphaZeroNetwork,
        buffer: PrioritizedReplayBuffer,
        config: TrainingConfig,
        device: torch.device,
    ) -> None:
        self.network = network
        self.buffer = buffer
        self.config = config
        self.device = device
        self.optimizer = torch.optim.AdamW(
            network.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
        )

    def train_step(self) -> LossStats | None:
        """Runs one optimizer step on a sampled batch.

        Returns:
            The losses, or `None` if the buffer is still smaller than a batch.

        Raises:
            NonFiniteLossError: If the loss is NaN or infinite.
        """
        config = self.config
        if len(self.buffer) < config.batch_size:
            return None
        batch = self.buffer.sample(config.batch_size)
        transitions = batch.transitions
        device = self.device
        observations = torch.from_numpy(np.stack([t.observation for t in transitions])).to(device)
        target_policies = torch.from_numpy(np.stack([t.policy for t in transitions])).to(device)
        target_values = torch.tensor(
            [t.value for t in transitions], dtype=torch.float32, device=device
        )
        weights = torch.from_numpy(batch.weights).to(device)  # (batch,)

        self.network.train()
        self.optimizer.zero_grad()
        logits, values = self.network(observations)  # (batch, num_actions), (batch, 1)
        values = values.squeeze(-1)  # (batch,)
        policy_errors = -(target_policies * nn.functional.log_softmax(logits, dim=1)).sum(dim=1)
        policy_loss = (policy_errors * weights).mean()
        value_loss = (
            nn.functional.mse_loss(values, target_values, reduction="none") * weights
        ).mean()
        loss = policy_loss + config.value_loss_weight * value_loss
        if not torch.isfinite(loss):
            raise NonFiniteLossError(
                f"loss={loss.item()} (policy={policy_loss.item()}, value={value_loss.item()})"
            )
        torch.autograd.backward(loss)
        self.optimizer.step()
        self.network.eval()

        priorities = policy_errors + config.value_loss_weight * (values - target_values).abs()
        self.buffer.update_priorities(batch.indices, priorities.detach().cpu().numpy())
        return LossStats(
            policy_loss=policy_loss.item(),
            value_loss=value_loss.item(),
            total_loss=loss.item(),
            value_loss_weight=config.value_loss_weight,
        )

    def save(self, path: Path) -> None:
        """Saves the network's state dict to `path`."""
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.network.state_dict(), path)
