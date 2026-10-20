import pytest

from hanabi_search.agents import AlphaZeroAgent
from hanabi_search.config import AlphaZeroAgentConfig, ExperimentConfig, NetworkConfig
from hanabi_search.exceptions import ConfigError
from hanabi_search.experiment import set_up

CONFIG = ExperimentConfig(
    name="x", agents=(AlphaZeroAgentConfig(),), network=NetworkConfig(hidden_sizes=(16,))
)


def test_set_up_builds_independent_tables_sharing_one_network() -> None:
    setup = set_up(CONFIG, num_tables=3)
    assert len(setup.tables) == 3
    agents = [agent for table in setup.tables for agent in table]
    assert len({id(agent) for agent in agents}) == 3 * CONFIG.game.players
    assert all(isinstance(a, AlphaZeroAgent) and a.network is setup.network for a in agents)
    assert setup.agents is setup.tables[0]


def test_set_up_needs_a_table() -> None:
    with pytest.raises(ConfigError, match="num_tables"):
        set_up(CONFIG, num_tables=0)
