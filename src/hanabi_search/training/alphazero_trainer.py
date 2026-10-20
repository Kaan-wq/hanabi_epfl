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
        # Fused: one kernel for all parameters, without a host sync per parameter.
        self.optimizer = torch.optim.AdamW(
            network.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
            fused=True,
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
        # One host-to-device copy: (batch, observation_size + num_actions + 2).
        packed = np.concatenate(
            [
                np.stack([t.observation for t in transitions]),
                np.stack([t.policy for t in transitions]),
                np.array([[t.value] for t in transitions], dtype=np.float32),
                batch.weights[:, np.newaxis],
            ],
            axis=1,
        )
        inputs = torch.from_numpy(packed).to(self.device)
        observation_size = transitions[0].observation.shape[0]
        observations = inputs[:, :observation_size]
        target_policies = inputs[:, observation_size:-2]  # (batch, num_actions)
        target_values = inputs[:, -2]  # (batch,)
        weights = inputs[:, -1]  # (batch,)

        self.network.train()
        self.optimizer.zero_grad(set_to_none=True)
        logits, values = self.network(observations)  # (batch, num_actions), (batch, 1)
        values = values.squeeze(-1)  # (batch,)
        policy_errors = -(target_policies * nn.functional.log_softmax(logits, dim=1)).sum(dim=1)
        policy_loss = (policy_errors * weights).mean()
        value_loss = (
            nn.functional.mse_loss(values, target_values, reduction="none") * weights
        ).mean()
        loss = policy_loss + config.value_loss_weight * value_loss
        priorities = policy_errors + config.value_loss_weight * (values - target_values).abs()
        # One device-to-host copy (the step's only sync): priorities, then the 3 losses.
        host = (
            torch.cat([priorities, torch.stack([policy_loss, value_loss, loss])])
            .detach()
            .cpu()
            .numpy()
        )
        policy_value, value_value, total_value = (float(x) for x in host[-3:])
        if not np.isfinite(total_value):
            self.network.eval()
            raise NonFiniteLossError(
                f"loss={total_value} (policy={policy_value}, value={value_value})"
            )
        torch.autograd.backward(loss)
        self.optimizer.step()
        self.network.eval()

        self.buffer.update_priorities(batch.indices, host[:-3])
        return LossStats(
            policy_loss=policy_value,
            value_loss=value_value,
            total_loss=total_value,
            value_loss_weight=config.value_loss_weight,
        )

    def save(self, path: Path) -> None:
        """Saves the network's state dict to `path`."""
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.network.state_dict(), path)
