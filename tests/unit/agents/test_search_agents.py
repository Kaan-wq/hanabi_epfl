import pytest
import torch

from hanabi_search.agents import RULE_BASED_AGENTS, AlphaZeroAgent, MCTSAgent
from hanabi_search.config import AlphaZeroAgentConfig, GameConfig, MCTSAgentConfig
from hanabi_search.env import HanabiEnv, HanabiMove
from hanabi_search.evaluation import play_games
from hanabi_search.networks import AlphaZeroNetwork
from hanabi_search.training import Transition
from tests.support import SMALL_GAME, current_observation


def _network(game: GameConfig) -> AlphaZeroNetwork:
    env = HanabiEnv(game.to_params())
    torch.manual_seed(0)
    return AlphaZeroNetwork(env.num_moves(), env.vectorized_observation_shape()[0], (32, 16))


@pytest.mark.parametrize("use_rules", [True, False])
def test_mcts_plays_a_game_and_collects_data(use_rules: bool) -> None:
    game = GameConfig(players=2)
    collected: list[Transition] = []
    mcts = MCTSAgent(
        0,
        game,
        MCTSAgentConfig(rollouts=4, use_rules=use_rules, collect_data=True),
        collected.extend,
    )
    partner = RULE_BASED_AGENTS["VanDenBergh"](1, game)
    (result,) = play_games(game, [mcts, partner], [3])
    assert 0 <= result.score <= 25
    num_moves = HanabiEnv(game.to_params()).num_moves()
    assert collected
    for transition in collected:
        assert transition.policy.shape == (num_moves,)
        assert transition.policy.sum() == pytest.approx(1.0)
        assert transition.value == pytest.approx(2 * result.score / 25 - 1)


def test_mcts_plays_reduced_game() -> None:
    agents = [MCTSAgent(seat, SMALL_GAME, MCTSAgentConfig(rollouts=3)) for seat in range(2)]
    (result,) = play_games(SMALL_GAME, agents, [1])
    assert 0 <= result.score <= SMALL_GAME.max_score


def test_mcts_returns_a_legal_move(env: HanabiEnv) -> None:
    observation = current_observation(env)
    agent = MCTSAgent(observation["current_player"], GameConfig(), MCTSAgentConfig(rollouts=5))
    move = agent.act(observation, env.state)
    assert isinstance(move, HanabiMove)
    assert env.state.move_is_legal(move)
    assert sum(agent.root_visits().values()) > 0


def test_mcts_without_rules_is_waiting_when_not_its_turn(env: HanabiEnv) -> None:
    observations = env.observe_all()
    waiting = 1 - observations["current_player"]
    agent = MCTSAgent(waiting, GameConfig(), MCTSAgentConfig(rollouts=2))
    assert agent.act(observations["player_observations"][waiting], env.state) is None


def test_mcts_choose_move_on_terminal_node_raises(env: HanabiEnv) -> None:
    agent = MCTSAgent(0, GameConfig(), MCTSAgentConfig(rollouts=1))
    while not env.state.is_terminal():
        env.step(env.state.legal_moves()[0])
    agent.reset(env.state)
    agent.root_node.focused_state = env.state
    with pytest.raises(RuntimeError, match="terminal"):
        agent.choose_move(agent.root_node)


@pytest.mark.parametrize("game", [GameConfig(players=2), SMALL_GAME])
def test_alphazero_plays_and_records_every_decision(game: GameConfig) -> None:
    collected: list[Transition] = []
    network = _network(game)
    agents = [
        AlphaZeroAgent(
            seat,
            game,
            AlphaZeroAgentConfig(rollouts=4),
            network,
            torch.device("cpu"),
            collected.extend,
        )
        for seat in range(game.players)
    ]
    (result,) = play_games(game, agents, [2])
    assert len(collected) == len(result.moves)
    assert all(-1 <= t.value <= 1 for t in collected)
