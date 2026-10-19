import pytest
import torch

from hanabi_search.agents import ParallelAlphaZeroAgent, ParallelMCTSAgent
from hanabi_search.agents.alphazero.parallel import extract_tensors, replace_tensors
from hanabi_search.config import AlphaZeroAgentConfig, GameConfig, MCTSAgentConfig
from hanabi_search.env import HanabiEnv, HanabiMove
from hanabi_search.networks import AlphaZeroNetwork
from tests.support import current_observation


def _network(env: HanabiEnv) -> AlphaZeroNetwork:
    torch.manual_seed(0)
    return AlphaZeroNetwork(env.num_moves(), env.vectorized_observation_shape()[0], (16,))


def test_weight_sharing_round_trip(env: HanabiEnv) -> None:
    network = _network(env)
    skeleton, tensors = extract_tensors(network)
    replace_tensors(skeleton, tensors)
    inputs = torch.rand(3, env.vectorized_observation_shape()[0])
    for restored, original in zip(skeleton(inputs), network(inputs), strict=True):
        assert torch.allclose(restored, original)


@pytest.mark.slow
@pytest.mark.usefixtures("ray_session")
@pytest.mark.filterwarnings("ignore::ResourceWarning")  # Raised inside Ray's own process handling.
@pytest.mark.filterwarnings("ignore::pytest.PytestUnraisableExceptionWarning")
def test_parallel_mcts_returns_legal_move(env: HanabiEnv) -> None:
    observation = current_observation(env)
    config = MCTSAgentConfig(rollouts=8, num_workers=2)
    agent = ParallelMCTSAgent(observation["current_player"], GameConfig(), config)
    move = agent.act(observation, env.state)
    assert isinstance(move, HanabiMove)
    assert env.state.move_is_legal(move)


@pytest.mark.slow
@pytest.mark.usefixtures("ray_session")
@pytest.mark.filterwarnings("ignore::ResourceWarning")  # Raised inside Ray's own process handling.
@pytest.mark.filterwarnings("ignore::pytest.PytestUnraisableExceptionWarning")
def test_parallel_alphazero_returns_legal_move(env: HanabiEnv) -> None:
    observation = current_observation(env)
    config = AlphaZeroAgentConfig(rollouts=8, num_workers=2)
    agent = ParallelAlphaZeroAgent(
        observation["current_player"], GameConfig(), config, _network(env), torch.device("cpu")
    )
    move = agent.act(observation, env.state)
    assert isinstance(move, HanabiMove)
    assert env.state.move_is_legal(move)
