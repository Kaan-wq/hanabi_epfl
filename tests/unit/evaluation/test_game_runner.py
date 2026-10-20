import gc

import pytest
import torch

from hanabi_search.agents import RULE_BASED_AGENTS, Agent, AlphaZeroAgent
from hanabi_search.config import AlphaZeroAgentConfig, GameConfig
from hanabi_search.env import HanabiEnv
from hanabi_search.evaluation import GameSession, play_games, play_games_batched
from hanabi_search.exceptions import ConfigError, HanabiSearchError
from hanabi_search.networks import AlphaZeroNetwork
from hanabi_search.training import Transition
from tests.support import PLAYER_COUNTS

SEEDS = [11, 12, 13, 14, 15]


def _network(game: GameConfig) -> AlphaZeroNetwork:
    env = HanabiEnv(game.to_params())
    torch.manual_seed(0)
    return AlphaZeroNetwork(env.num_moves(), env.vectorized_observation_shape()[0], (32, 16)).eval()


def _alphazero_table(
    game: GameConfig, network: AlphaZeroNetwork, sink: list[Transition] | None = None
) -> list[Agent]:
    return [
        AlphaZeroAgent(
            seat,
            game,
            AlphaZeroAgentConfig(rollouts=6),
            network,
            None if sink is None else sink.extend,
        )
        for seat in range(game.players)
    ]


@pytest.mark.parametrize("players", PLAYER_COUNTS)
def test_batched_games_match_sequential_games(players: int) -> None:
    game = GameConfig(players=players)
    network = _network(game)
    sequential = play_games(game, _alphazero_table(game, network), SEEDS)
    batched = play_games_batched(
        game, [_alphazero_table(game, network) for _ in range(3)], SEEDS, network
    )
    assert [r.seed for r in batched] == SEEDS
    assert [r.moves for r in batched] == [r.moves for r in sequential]


def test_batched_games_with_more_tables_than_seeds() -> None:
    game = GameConfig(players=2)
    network = _network(game)
    tables = [_alphazero_table(game, network) for _ in range(4)]
    assert [r.seed for r in play_games_batched(game, tables, [7, 8], network)] == [7, 8]


def test_batched_games_collect_every_decision() -> None:
    game = GameConfig(players=2)
    network = _network(game)
    collected: list[Transition] = []
    tables = [_alphazero_table(game, network, collected) for _ in range(2)]
    results = play_games_batched(game, tables, SEEDS, network)
    assert len(collected) == sum(len(r.moves) for r in results)


def test_mixed_tables_only_batch_network_seats() -> None:
    game = GameConfig(players=2)
    network = _network(game)

    def table() -> list[Agent]:
        return [
            AlphaZeroAgent(0, game, AlphaZeroAgentConfig(rollouts=4), network),
            RULE_BASED_AGENTS["VanDenBergh"](1, game),
        ]

    batched = play_games_batched(game, [table(), table()], SEEDS, network)
    assert [r.moves for r in batched] == [r.moves for r in play_games(game, table(), SEEDS)]


def test_games_restore_the_gc_thresholds() -> None:
    game = GameConfig(players=2)
    network = _network(game)
    thresholds = gc.get_threshold()
    during: list[int] = []
    table = _alphazero_table(game, network)
    play_games_batched(game, [table], [7], network, lambda *_: during.append(gc.get_threshold()[0]))
    assert min(during) >= 10_000
    assert gc.get_threshold() == thresholds


def test_batched_games_need_a_table() -> None:
    game = GameConfig(players=2)
    with pytest.raises(ConfigError, match="at least one table"):
        play_games_batched(game, [], SEEDS, _network(game))


def test_session_requires_the_pending_evaluation() -> None:
    game = GameConfig(players=2)
    session = GameSession(HanabiEnv(game.to_params(3)), _alphazero_table(game, _network(game)), 3)
    assert session.advance() is not None
    with pytest.raises(HanabiSearchError, match="pending request"):
        session.advance()
