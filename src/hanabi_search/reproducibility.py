"""Seeding and device selection."""

import logging
import random
from typing import Literal

import numpy as np
import torch

from hanabi_search.exceptions import ConfigError

logger = logging.getLogger(__name__)


def seed_everything(seed: int) -> None:
    """Seeds Python's `random`, NumPy and PyTorch (e.g. for a network's initial weights).

    Games do not depend on these: the engine is seeded through the game's `seed`
    parameter, and agents draw from the game's own generator.
    """
    random.seed(seed)
    np.random.seed(seed)  # noqa: NPY002 -- seeds any library still using it.
    torch.manual_seed(seed)


def select_device(name: Literal["cpu", "mps"]) -> torch.device:
    """Returns the requested device.

    Raises:
        ConfigError: If MPS is requested but not available.
    """
    if name == "mps" and not torch.backends.mps.is_available():
        raise ConfigError("Device 'mps' requested but MPS is not available on this machine")
    logger.info("Using device %s", name)
    return torch.device(name)
