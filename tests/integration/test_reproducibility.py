from hanabi_search.agents import RULE_BASED_AGENTS, MCTSAgent
from hanabi_search.config import GameConfig, MCTSAgentConfig
from hanabi_search.evaluation import evaluate, play_games


def _mcts_table(game: GameConfig) -> list[MCTSAgent]:
    return [MCTSAgent(seat, game, MCTSAgentConfig(rollouts=4)) for seat in range(game.players)]


def test_same_seeds_give_identical_mcts_games() -> None:
    game = GameConfig(players=2)
    first = play_games(game, _mcts_table(game), [10, 11])
    second = play_games(game, _mcts_table(game), [10, 11])
    assert [r.moves for r in first] == [r.moves for r in second]


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
