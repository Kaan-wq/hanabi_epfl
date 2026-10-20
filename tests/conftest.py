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
