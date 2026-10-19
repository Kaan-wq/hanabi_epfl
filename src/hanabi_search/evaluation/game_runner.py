"""Playing games between agents."""

import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from hanabi_search.agents import Agent
from hanabi_search.config import GameConfig
from hanabi_search.env import HanabiEndOfGameType, HanabiEnv, StepResult
from hanabi_search.evaluation.metrics import ScoreSummary
from hanabi_search.exceptions import HanabiSearchError
from hanabi_search.reproducibility import seed_everything

logger = logging.getLogger(__name__)

# Called after every move with the environment and the step's result.
type MoveCallback = Callable[[HanabiEnv, StepResult], None]


@dataclass(frozen=True, slots=True)
class GameResult:
    """Outcome of one game.

    Attributes:
        seed: Seed the game was played with.
        score: Final score.
        end_status: Why the game ended.
        moves: Every move, as the engine prints it.
    """

    seed: int
    score: int
    end_status: HanabiEndOfGameType
    moves: list[str]


def play_game(
    env: HanabiEnv, agents: Sequence[Agent], seed: int, on_move: MoveCallback | None = None
) -> GameResult:
    """Plays one game to the end, then calls `end_episode` on every agent.

    Args:
        env: Environment, already configured with the game's seed.
        agents: One agent per seat.
        seed: Seed the environment was created with (recorded in the result).
        on_move: Optional callback after every move.

    Raises:
        HanabiSearchError: If the acting agent returns no action.
    """
    observations = env.reset()
    moves: list[str] = []
    done = False
    while not done:
        player = observations["current_player"]
        action = agents[player].act(observations["player_observations"][player], env.state)
        if action is None:
            raise HanabiSearchError(f"Agent in seat {player} returned no action on its turn")
        move = env.to_move(action)
        moves.append(str(move))
        result = env.step(move)
        observations, done = result.observations, result.done
        if on_move is not None:
            on_move(env, result)
    score = env.score()
    for agent in agents:
        agent.end_episode(score)
    return GameResult(seed, score, env.state.end_of_game_status(), moves)


def play_games(
    game: GameConfig,
    agents: Sequence[Agent],
    seeds: Sequence[int],
    on_move: MoveCallback | None = None,
) -> list[GameResult]:
    """Plays one game per seed. Each game reseeds Python, NumPy, PyTorch and the engine.

    Args:
        game: Game variant.
        agents: One agent per seat, reused across games.
        seeds: Seed of each game.
        on_move: Optional callback after every move.
    """
    results = []
    for seed in seeds:
        seed_everything(seed)
        result = play_game(HanabiEnv(game.to_params(seed)), agents, seed, on_move)
        logger.debug("Game seed=%d score=%d (%s)", seed, result.score, result.end_status.name)
        results.append(result)
    return results


def evaluate(
    game: GameConfig, agents: Sequence[Agent], num_games: int, first_seed: int
) -> ScoreSummary:
    """Plays `num_games` games on seeds `first_seed, first_seed + 1, ...` and summarizes scores."""
    results = play_games(game, agents, range(first_seed, first_seed + num_games))
    return ScoreSummary.from_scores(
        [result.score for result in results], game.players, game.max_score
    )
