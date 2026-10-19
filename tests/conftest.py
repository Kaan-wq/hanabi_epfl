from collections.abc import Iterator

import pytest

from hanabi_search.config import GameConfig
from hanabi_search.env import HanabiEnv, HanabiState
from tests.support import make_env


@pytest.fixture
def game() -> GameConfig:
    return GameConfig(players=2)


@pytest.fixture
def env(game: GameConfig) -> HanabiEnv:
    return make_env(game, seed=7)


@pytest.fixture
def state(env: HanabiEnv) -> HanabiState:
    return env.state


@pytest.fixture
def ray_session() -> Iterator[None]:
    ray = pytest.importorskip("ray")
    from hanabi_search.agents.mcts.parallel import ensure_ray  # noqa: PLC0415

    # AlphaZero workers each reserve 4 CPUs; Ray treats these as scheduling labels only.
    ensure_ray(num_cpus=8)
    yield
    ray.shutdown()
