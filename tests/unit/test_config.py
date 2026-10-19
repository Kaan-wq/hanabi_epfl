from pathlib import Path

import pytest

from hanabi_search.config import (
    ExperimentConfig,
    GameConfig,
    MCTSAgentConfig,
    RuleBasedAgentConfig,
)
from hanabi_search.exceptions import ConfigError
from tests.support import REPO_ROOT

CONFIG_FILES = sorted((REPO_ROOT / "configs").glob("*.yaml"))


@pytest.mark.parametrize("path", CONFIG_FILES, ids=lambda p: p.name)
def test_shipped_configs_are_valid(path: Path) -> None:
    config = ExperimentConfig.from_yaml(path)
    assert len(config.seat_agents()) == config.game.players
    assert ExperimentConfig.model_validate_json(config.model_dump_json()) == config


def test_game_params_match_engine_names() -> None:
    params = GameConfig(players=3, hand_size=4).to_params(seed=9)
    assert params == {
        "colors": 5,
        "ranks": 5,
        "players": 3,
        "max_information_tokens": 8,
        "max_life_tokens": 3,
        "observation_type": 1,
        "random_start_player": True,
        "hand_size": 4,
        "seed": 9,
    }
    assert "seed" not in GameConfig().to_params()


def test_single_agent_fills_every_seat() -> None:
    config = ExperimentConfig(name="x", game=GameConfig(players=4), agents=(MCTSAgentConfig(),))
    assert len(config.seat_agents()) == 4


def test_wrong_number_of_agents_is_rejected() -> None:
    with pytest.raises(ValueError, match="1 or 3 entries"):
        ExperimentConfig(
            name="x",
            game=GameConfig(players=3),
            agents=(RuleBasedAgentConfig(), RuleBasedAgentConfig()),
        )


def test_invalid_yaml_raises_config_error(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text("name: x\nagents:\n  - kind: telepathy\n")
    with pytest.raises(ConfigError):
        ExperimentConfig.from_yaml(path)
    with pytest.raises(ConfigError):
        ExperimentConfig.from_yaml(tmp_path / "missing.yaml")


def test_yaml_round_trip(tmp_path: Path) -> None:
    config = ExperimentConfig.from_yaml(REPO_ROOT / "configs" / "alphazero_2p.yaml")
    path = tmp_path / "resolved.yaml"
    path.write_text(config.to_yaml())
    assert ExperimentConfig.from_yaml(path) == config
