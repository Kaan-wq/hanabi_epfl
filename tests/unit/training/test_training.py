import json
import math
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pytest
import torch

from hanabi_search.config import TrainingConfig
from hanabi_search.exceptions import NonFiniteLossError
from hanabi_search.networks import AlphaZeroNetwork
from hanabi_search.training import (
    AlphaZeroTrainer,
    PrioritizedReplayBuffer,
    Transition,
    value_target,
    visit_policy,
)

OBSERVATION_SIZE = 12
NUM_ACTIONS = 5


def _transitions(count: int, seed: int = 0) -> list[Transition]:
    rng = np.random.default_rng(seed)
    result = []
    for _ in range(count):
        policy = rng.random(NUM_ACTIONS).astype(np.float32)
        result.append(
            Transition(
                observation=rng.integers(0, 2, OBSERVATION_SIZE).astype(np.float32),
                policy=policy / policy.sum(),
                value=float(rng.uniform(-1, 1)),
            )
        )
    return result


def _buffer(capacity: int = 64, save_path: Path | None = None) -> PrioritizedReplayBuffer:
    return PrioritizedReplayBuffer(capacity, 0.6, 0.4, np.random.default_rng(0), save_path)


def test_value_target_maps_scores_to_unit_interval() -> None:
    assert value_target(0, 25) == -1
    assert value_target(25, 25) == 1
    assert value_target(5, 10) == 0


def test_visit_policy_normalizes_and_falls_back_to_uniform() -> None:
    assert visit_policy({1: 3, 3: 1}, 4).tolist() == [0, 0.75, 0, 0.25]
    assert visit_policy({}, 4).tolist() == [0.25] * 4


def test_buffer_overwrites_oldest_when_full() -> None:
    buffer = _buffer(capacity=4)
    buffer.add(_transitions(6))
    assert len(buffer) == 4


def test_buffer_sample_and_priorities() -> None:
    buffer = _buffer()
    buffer.add(_transitions(32))
    batch = buffer.sample(16)
    assert len(batch.transitions) == 16
    assert batch.indices.shape == batch.weights.shape == (16,)
    assert np.all((batch.weights > 0) & (batch.weights <= 1))
    beta = buffer.beta
    buffer.update_priorities(batch.indices, np.ones(16, dtype=np.float32))
    assert buffer.beta > beta


def _priority_buffer(alpha: float) -> PrioritizedReplayBuffer:
    return PrioritizedReplayBuffer(16, alpha, 0.5, np.random.default_rng(0))


def _draws(
    buffer: PrioritizedReplayBuffer, count: int
) -> tuple[npt.NDArray[np.int64], dict[int, float]]:
    """Samples `count` batches of `len(buffer)`; returns all indices and the weight per index.

    Weights are normalized by each batch's maximum, so only weights from batches that
    contain every index are recorded.
    """
    indices: list[npt.NDArray[np.int64]] = []
    weights: dict[int, float] = {}
    for _ in range(count):
        batch = buffer.sample(len(buffer))
        indices.append(batch.indices)
        if len(set(batch.indices.tolist())) == len(buffer):
            weights = {int(i): float(w) for i, w in zip(batch.indices, batch.weights, strict=True)}
    return np.concatenate(indices), weights


def test_buffer_applies_alpha_once() -> None:
    buffer = _priority_buffer(alpha=0.5)
    buffer.add(_transitions(2))
    buffer.update_priorities(np.array([0, 1]), np.array([1.0, 4.0], dtype=np.float32))
    # Probabilities are proportional to [1, 4] ** 0.5 = [1, 2], so sample 1 is drawn twice
    # as often and its importance weight is 2 ** -beta times that of sample 0.
    indices, weights = _draws(buffer, 2000)
    assert np.mean(indices == 1) == pytest.approx(2 / 3, abs=0.03)
    assert weights[1] / weights[0] == pytest.approx(2**-buffer.beta, rel=1e-4)


def test_buffer_gives_new_transitions_the_max_raw_priority() -> None:
    buffer = _priority_buffer(alpha=0.5)
    buffer.add(_transitions(2))
    buffer.update_priorities(np.array([0, 1]), np.array([9.0, 1.0], dtype=np.float32))
    buffer.add(_transitions(1, seed=1))
    indices, weights = _draws(buffer, 2000)
    # Sample 2 is as likely as sample 0 (raw priority 9), so their weights match.
    assert np.mean(indices == 2) == pytest.approx(np.mean(indices == 0), abs=0.03)
    assert weights[2] / weights[0] == pytest.approx(1.0, rel=1e-4)


def test_buffer_sample_larger_than_contents_raises() -> None:
    buffer = _buffer()
    buffer.add(_transitions(3))
    with pytest.raises(ValueError, match="Cannot sample"):
        buffer.sample(4)


def test_buffer_saves_transitions(tmp_path: Path) -> None:
    path = tmp_path / "transitions.jsonl"
    buffer = _buffer(save_path=path)
    buffer.add(_transitions(3))
    records = [json.loads(line) for line in path.read_text().splitlines()]
    assert len(records) == 3
    assert len(records[0]["observation"]) == OBSERVATION_SIZE


def _trainer(config: TrainingConfig) -> AlphaZeroTrainer:
    torch.manual_seed(0)
    network = AlphaZeroNetwork(NUM_ACTIONS, OBSERVATION_SIZE, (16, 8))
    buffer = _buffer()
    buffer.add(_transitions(48))
    return AlphaZeroTrainer(network, buffer, config, torch.device("cpu"))


def test_training_steps_give_finite_decreasing_loss() -> None:
    trainer = _trainer(TrainingConfig(batch_size=16, learning_rate=1e-2))
    losses = [trainer.train_step() for _ in range(30)]
    assert all(stats is not None and math.isfinite(stats.total_loss) for stats in losses)
    first, last = losses[0], losses[-1]
    assert first is not None and last is not None
    assert last.total_loss < first.total_loss
    assert first.policy_share + first.value_share == pytest.approx(1.0)


def test_training_waits_for_a_full_batch() -> None:
    trainer = _trainer(TrainingConfig(batch_size=128))
    assert trainer.train_step() is None


def test_non_finite_loss_raises() -> None:
    trainer = _trainer(TrainingConfig(batch_size=16))
    with torch.no_grad():
        for parameter in trainer.network.parameters():
            parameter.fill_(float("nan"))
    with pytest.raises(NonFiniteLossError):
        trainer.train_step()


def test_checkpoint_round_trip(tmp_path: Path) -> None:
    trainer = _trainer(TrainingConfig(batch_size=16))
    trainer.save(tmp_path / "model.pt")
    restored = AlphaZeroNetwork(NUM_ACTIONS, OBSERVATION_SIZE, (16, 8))
    restored.load_state_dict(torch.load(tmp_path / "model.pt", weights_only=True))
    for a, b in zip(restored.parameters(), trainer.network.parameters(), strict=True):
        assert torch.equal(a, b)
