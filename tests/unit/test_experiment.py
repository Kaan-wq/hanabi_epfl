import pytest

from hanabi_search.agents import AlphaZeroAgent
from hanabi_search.config import (
    AlphaZeroAgentConfig,
    EvaluationConfig,
    ExperimentConfig,
    MCTSAgentConfig,
    NetworkConfig,
)
from hanabi_search.exceptions import ConfigError
from hanabi_search.experiment import evaluate_experiment, set_up

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


@pytest.mark.parametrize(
    "agent", [AlphaZeroAgentConfig(rollouts=4), MCTSAgentConfig(rollouts=4)], ids=["az", "mcts"]
)
def test_evaluation_does_not_depend_on_the_number_of_processes(
    agent: AlphaZeroAgentConfig | MCTSAgentConfig,
) -> None:
    summaries = []
    for num_processes in (1, 3):
        config = CONFIG.model_copy(
            update={
                "agents": (agent,),
                "evaluation": EvaluationConfig(
                    num_games=6, seed=50, parallel_games=2, num_processes=num_processes
                ),
            }
        )
        summaries.append(evaluate_experiment(config))
    assert summaries[0] == summaries[1]
    assert summaries[0].num_games == 6
