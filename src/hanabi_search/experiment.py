"""Wiring an `ExperimentConfig` into agents, networks, run directories and training loops."""

import json
import logging
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch

from hanabi_search.agents import Agent, AgentResources, build_agents
from hanabi_search.config import (
    AlphaZeroAgentConfig,
    ExperimentConfig,
    NetworkConfig,
    SelfPlayConfig,
)
from hanabi_search.env import OBSERVATION_ENCODER_VERSION, HanabiEnv, StepResult
from hanabi_search.evaluation import play_games
from hanabi_search.exceptions import ConfigError
from hanabi_search.logging_utils import setup_logging
from hanabi_search.networks import AlphaZeroNetwork
from hanabi_search.reproducibility import select_device
from hanabi_search.training import (
    AlphaZeroTrainer,
    LossStats,
    PrioritizedReplayBuffer,
    TransitionSink,
    build_network,
)

logger = logging.getLogger(__name__)

RUNS_ROOT = Path("runs")
CHECKPOINT_FILE = "model.pt"
CHECKPOINT_METADATA_FILE = "checkpoint.json"


def create_run_dir(config: ExperimentConfig, root: Path = RUNS_ROOT) -> Path:
    """Creates `root/<name>/<timestamp>_<seed>/`, writes the resolved config and logs there."""
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    run_dir = root / config.name / f"{timestamp}_{config.seed}"
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "config.yaml").write_text(config.to_yaml())
    setup_logging(log_file=run_dir / "run.log")
    logger.info("Run directory %s (seed %d)", run_dir, config.seed)
    return run_dir


def uses_network(config: ExperimentConfig) -> bool:
    """Returns whether any seat is an AlphaZero agent."""
    return any(isinstance(agent, AlphaZeroAgentConfig) for agent in config.agents)


def make_network(config: ExperimentConfig, device: torch.device) -> AlphaZeroNetwork:
    """Builds the experiment's network, sized for its game."""
    env = HanabiEnv(config.game.to_params())
    return build_network(
        config.network, env.num_moves(), env.vectorized_observation_shape()[0], device
    )


def save_checkpoint(network: AlphaZeroNetwork, run_dir: Path, config: ExperimentConfig) -> Path:
    """Saves the network with the metadata needed to reuse it (config, encoder version)."""
    path = run_dir / CHECKPOINT_FILE
    torch.save(network.state_dict(), path)
    metadata = {
        "observation_encoder": OBSERVATION_ENCODER_VERSION,
        "game": config.game.model_dump(mode="json"),
        "network": config.network.model_dump(mode="json"),
        "experiment": config.name,
        "seed": config.seed,
    }
    (run_dir / CHECKPOINT_METADATA_FILE).write_text(json.dumps(metadata, indent=2))
    return path


@dataclass(slots=True)
class Setup:
    """Everything needed to play an experiment's games.

    Attributes:
        agents: One agent per seat.
        network: The AlphaZero network, if any seat uses one.
        device: Device of `network`.
    """

    agents: list[Agent]
    network: AlphaZeroNetwork | None
    device: torch.device


def set_up(
    config: ExperimentConfig,
    checkpoint: Path | None = None,
    transition_sink: TransitionSink | None = None,
    self_play: SelfPlayConfig | None = None,
    ask: Callable[[str], str] | None = None,
    show: Callable[[str], None] | None = None,
) -> Setup:
    """Builds the device, network (if needed) and agents of an experiment.

    Args:
        config: The experiment.
        checkpoint: Network weights overriding `config.network.checkpoint`.
        transition_sink: Where search agents send training data.
        self_play: Exploration for AlphaZero agents; only for training games.
        ask: Input function for human seats.
        show: Output function for human seats.
    """
    device = select_device(config.device)
    network = None
    if uses_network(config):
        network_config = config.network
        if checkpoint is not None:
            network_config = NetworkConfig(
                hidden_sizes=network_config.hidden_sizes, checkpoint=checkpoint
            )
        network = make_network(config.model_copy(update={"network": network_config}), device)
        network.eval()
    resources = AgentResources(network, transition_sink, ask, show, self_play)
    return Setup(build_agents(config.seat_agents(), config.game, resources), network, device)


def train_alphazero(config: ExperimentConfig, run_dir: Path) -> Path:
    """Self-play AlphaZero training; saves the network after every episode.

    Episode `i` is played with seed `config.seed + i`. Training metrics are appended
    to `run_dir/metrics.jsonl`.

    Returns:
        Path of the saved network.

    Raises:
        ConfigError: If the experiment has no training section or no AlphaZero seat.
    """
    training = config.training
    if training is None or not uses_network(config):
        raise ConfigError("Training needs a `training` section and at least one AlphaZero agent")
    buffer_config = training.replay_buffer
    buffer = PrioritizedReplayBuffer(
        capacity=buffer_config.capacity,
        alpha=buffer_config.alpha,
        beta=buffer_config.beta,
        rng=np.random.default_rng(config.seed),
        save_path=run_dir / "transitions.jsonl" if buffer_config.save_transitions else None,
    )
    setup = set_up(config, transition_sink=buffer.add, self_play=training.self_play)
    assert setup.network is not None
    trainer = AlphaZeroTrainer(setup.network, buffer, training, setup.device)
    metrics_path = run_dir / "metrics.jsonl"
    checkpoint = run_dir / CHECKPOINT_FILE

    def train_after_move(_env: HanabiEnv, _result: StepResult) -> None:
        trainer.train_step()

    on_move = train_after_move if training.train_every_step else None
    for episode in range(training.num_episodes):
        seed = config.seed + episode
        (result,) = play_games(config.game, setup.agents, [seed], on_move)
        stats = trainer.train_step()
        checkpoint = save_checkpoint(setup.network, run_dir, config)
        _log_episode(metrics_path, episode, result.score, len(buffer), stats)
    return checkpoint


def _log_episode(
    metrics_path: Path, episode: int, score: int, buffer_size: int, stats: LossStats | None
) -> None:
    record: dict[str, float | int | None] = {
        "episode": episode,
        "score": score,
        "buffer_size": buffer_size,
        "policy_loss": None if stats is None else stats.policy_loss,
        "value_loss": None if stats is None else stats.value_loss,
        "total_loss": None if stats is None else stats.total_loss,
    }
    with metrics_path.open("a") as file:
        file.write(json.dumps(record) + "\n")
    if stats is None:
        logger.info(
            "Episode %d: score %d (buffer %d, not training yet)", episode, score, buffer_size
        )
    else:
        logger.info(
            "Episode %d: score %d, loss %.4f (policy %.4f, value %.4f), buffer %d",
            episode,
            score,
            stats.total_loss,
            stats.policy_loss,
            stats.value_loss,
            buffer_size,
        )
