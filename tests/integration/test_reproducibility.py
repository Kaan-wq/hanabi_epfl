import random

import numpy as np
import pytest
import torch

from hanabi_search.agents import RULE_BASED_AGENTS, AlphaZeroAgent, MCTSAgent, ParallelMCTSAgent
from hanabi_search.config import AlphaZeroAgentConfig, GameConfig, MCTSAgentConfig, SelfPlayConfig
from hanabi_search.env import HanabiEnv
from hanabi_search.evaluation import evaluate, play_game, play_games
from hanabi_search.networks import AlphaZeroNetwork
from hanabi_search.training import Transition
from tests.support import PLAYER_COUNTS


def _mcts_table(game: GameConfig) -> list[MCTSAgent]:
    return [MCTSAgent(seat, game, MCTSAgentConfig(rollouts=4)) for seat in range(game.players)]


def test_same_seeds_give_identical_mcts_games() -> None:
    game = GameConfig(players=2)
    first = play_games(game, _mcts_table(game), [10, 11])
    second = play_games(game, _mcts_table(game), [10, 11])
    assert [r.moves for r in first] == [r.moves for r in second]


@pytest.mark.parametrize("players", PLAYER_COUNTS)
def test_games_ignore_the_global_random_state(players: int) -> None:
    # Agents draw only from the game's own generator, so interleaved games cannot
    # disturb each other.
    game = GameConfig(players=players)
    results = []
    for global_seed in (1, 2):
        random.seed(global_seed)
        env = HanabiEnv(game.to_params(4))
        results.append(play_game(env, _mcts_table(game), 4).moves)
    assert results[0] == results[1]


def _self_play(game: GameConfig, seeds: list[int]) -> tuple[list[list[str]], list[Transition]]:
    """Plays AlphaZero self-play games (with exploration) on `seeds`."""
    env = HanabiEnv(game.to_params())
    torch.manual_seed(0)
    network = AlphaZeroNetwork(env.num_moves(), env.vectorized_observation_shape()[0], (16,))
    collected: list[Transition] = []
    agents = [
        AlphaZeroAgent(
            seat,
            game,
            AlphaZeroAgentConfig(rollouts=4),
            network,
            torch.device("cpu"),
            collected.extend,
            SelfPlayConfig(),
        )
        for seat in range(game.players)
    ]
    return [r.moves for r in play_games(game, agents, seeds)], collected


@pytest.mark.parametrize("players", PLAYER_COUNTS)
def test_same_seeds_give_identical_alphazero_self_play(players: int) -> None:
    game = GameConfig(players=players)
    first_moves, first_data = _self_play(game, [5])
    second_moves, second_data = _self_play(game, [5])
    assert first_moves == second_moves
    assert len(first_data) == len(second_data)
    for a, b in zip(first_data, second_data, strict=True):
        assert np.array_equal(a.observation, b.observation)
        assert np.array_equal(a.policy, b.policy)
        assert a.value == b.value
    other_moves, _ = _self_play(game, [6])
    assert other_moves != first_moves


@pytest.mark.slow
@pytest.mark.usefixtures("ray_session")
@pytest.mark.filterwarnings("ignore::ResourceWarning")  # Raised inside Ray's own process handling.
@pytest.mark.filterwarnings("ignore::pytest.PytestUnraisableExceptionWarning")
def test_same_seeds_give_identical_parallel_mcts_games() -> None:
    game = GameConfig(players=2)
    config = MCTSAgentConfig(rollouts=40, num_workers=2)
    agents = [ParallelMCTSAgent(seat, game, config) for seat in range(2)]
    first = play_games(game, agents, [10])
    second = play_games(game, agents, [10])
    assert first[0].moves == second[0].moves


def test_game_outcome_does_not_depend_on_play_order() -> None:
    # Each game reseeds everything, so a game's result is a function of its seed alone.
    game = GameConfig(players=3)
    agents = [RULE_BASED_AGENTS["Piers"](seat, game) for seat in range(3)]
    forward = {r.seed: r.moves for r in play_games(game, agents, [1, 2, 3])}
    backward = {r.seed: r.moves for r in play_games(game, agents, [3, 2, 1])}
    assert forward == backward


def test_evaluate_reports_requested_number_of_games() -> None:
    game = GameConfig(players=4)
    agents = [RULE_BASED_AGENTS["VanDenBergh"](seat, game) for seat in range(4)]
    summary = evaluate(game, agents, num_games=20, first_seed=500)
    assert summary.num_games == 20
    assert summary.num_players == 4
    assert 0 <= summary.mean <= 25
