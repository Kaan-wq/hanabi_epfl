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
from hanabi_search.evaluation import GameResult, play_games_batched
from hanabi_search.exceptions import ConfigError
from hanabi_search.logging_utils import setup_logging
from hanabi_search.networks import AlphaZeroNetwork
from hanabi_search.reproducibility import seed_everything, select_device
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
        tables: Agents of each concurrent game, one per seat (no agent is shared).
        network: The AlphaZero network, if any seat uses one.
        device: Device of `network`.
    """

    tables: list[list[Agent]]
    network: AlphaZeroNetwork | None
    device: torch.device

    @property
    def agents(self) -> list[Agent]:
        """The first table's agents, for playing one game at a time."""
        return self.tables[0]


def set_up(
    config: ExperimentConfig,
    checkpoint: Path | None = None,
    transition_sink: TransitionSink | None = None,
    self_play: SelfPlayConfig | None = None,
    ask: Callable[[str], str] | None = None,
    show: Callable[[str], None] | None = None,
    num_tables: int = 1,
) -> Setup:
    """Builds the device, network (if needed) and agents of an experiment.

    Args:
        config: The experiment.
        checkpoint: Network weights overriding `config.network.checkpoint`.
        transition_sink: Where search agents send training data.
        self_play: Exploration for AlphaZero agents; only for training games.
        ask: Input function for human seats.
        show: Output function for human seats.
        num_tables: Number of independent sets of agents, one per concurrent game.

    Raises:
        ConfigError: If `num_tables` is below 1.
    """
    if num_tables < 1:
        raise ConfigError(f"num_tables must be at least 1, got {num_tables}")
    device = select_device(config.device)
    seed_everything(config.seed)  # Initial weights of a network without checkpoint.
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
    tables = [build_agents(config.seat_agents(), config.game, resources) for _ in range(num_tables)]
    return Setup(tables, network, device)


def train_alphazero(config: ExperimentConfig, run_dir: Path) -> Path:
    """Self-play AlphaZero training; saves the network after every episode.

    Episodes are played `training.parallel_games` at a time, with batched network
    evaluations; episode `i` uses seed `config.seed + i`. The network trains while
    they are played, so games in progress see updated weights. Training metrics are
    appended to `run_dir/metrics.jsonl`, one line per finished game.

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
    setup = set_up(
        config,
        transition_sink=buffer.add,
        self_play=training.self_play,
        num_tables=training.parallel_games,
    )
    network = setup.network
    assert network is not None
    trainer = AlphaZeroTrainer(network, buffer, training, setup.device)
    metrics_path = run_dir / "metrics.jsonl"
    pending_steps = 0.0
    finished_games = 0

    def train_after_move(_env: HanabiEnv, _result: StepResult) -> None:
        nonlocal pending_steps
        pending_steps += training.train_steps_per_move
        while pending_steps >= 1:
            trainer.train_step()
            pending_steps -= 1

    def after_game(result: GameResult) -> None:
        nonlocal finished_games
        stats = trainer.train_step()
        save_checkpoint(network, run_dir, config)
        _log_episode(metrics_path, finished_games, result, len(buffer), stats)
        finished_games += 1

    seeds = [config.seed + episode for episode in range(training.num_episodes)]
    play_games_batched(config.game, setup.tables, seeds, network, train_after_move, after_game)
    return run_dir / CHECKPOINT_FILE


def _log_episode(
    metrics_path: Path, episode: int, result: GameResult, buffer_size: int, stats: LossStats | None
) -> None:
    score = result.score
    record: dict[str, float | int | None] = {
        "episode": episode,
        "seed": result.seed,
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
