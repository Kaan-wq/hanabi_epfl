"""Policy/value network for AlphaZero."""

from collections.abc import Sequence

import numpy as np
import numpy.typing as npt
import torch
from torch import nn


class AlphaZeroNetwork(nn.Module):
    """MLP with a shared backbone, a policy head and a value head.

    Submodule names (`backbone`, `policy_head`, `value_head`) are part of the
    checkpoint format; renaming them breaks loading saved state dicts.

    Args:
        num_actions: Size of the action space.
        observation_size: Length of the encoded observation.
        hidden_sizes: Widths of the backbone layers.
    """

    def __init__(
        self, num_actions: int, observation_size: int, hidden_sizes: Sequence[int] = (512, 256, 256)
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        previous = observation_size
        for size in hidden_sizes:
            layers.extend([nn.Linear(previous, size), nn.GELU()])
            previous = size
        self.backbone = nn.Sequential(*layers)
        self.policy_head = nn.Sequential(
            nn.Linear(previous, previous), nn.GELU(), nn.Linear(previous, num_actions)
        )
        self.value_head = nn.Sequential(
            nn.Linear(previous, previous // 2),
            nn.GELU(),
            nn.Linear(previous // 2, 1),
            nn.Tanh(),  # Values in [-1, 1].
        )
        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, observations: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Computes policy logits and values.

        Args:
            observations: Encoded observations, shape (batch, observation_size).

        Returns:
            Policy logits, shape (batch, num_actions), and values in [-1, 1],
            shape (batch, 1).
        """
        features = self.backbone(observations)
        return self.policy_head(features), self.value_head(features)

    @torch.no_grad()
    def predict(
        self, observations: npt.NDArray[np.float32]
    ) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
        """Evaluates a batch on the network's device, for search.

        Args:
            observations: Encoded observations, shape (batch, observation_size).

        Returns:
            Move probabilities, shape (batch, num_actions), and values in [-1, 1],
            shape (batch,).
        """
        device = next(self.parameters()).device
        logits, values = self(torch.from_numpy(observations).to(device))
        # One device-to-host copy for both outputs.
        outputs = torch.cat([torch.softmax(logits, dim=1), values], dim=1).cpu().numpy()
        return outputs[:, :-1], outputs[:, -1]
