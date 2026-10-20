"""Playing games between agents, optionally many at once with batched network evaluations."""

import gc
import logging
from collections import deque
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager

import numpy as np

from hanabi_search.agents import Agent
from hanabi_search.agents.base import Evaluation, EvaluationRequest
from hanabi_search.config import GameConfig
from hanabi_search.env import HanabiEnv
from hanabi_search.evaluation.game_session import GameResult, GameSession, MoveCallback
from hanabi_search.evaluation.metrics import ScoreSummary
from hanabi_search.exceptions import ConfigError
from hanabi_search.networks import AlphaZeroNetwork

logger = logging.getLogger(__name__)

# Young-generation GC threshold while games are played (Python's default is 2,000).
_SEARCH_GC_THRESHOLD = 10_000


@contextmanager
def _infrequent_garbage_collection() -> Iterator[None]:
    """Raises the young-generation GC threshold for the duration of the games.

    Search trees allocate many small lists that live for a whole move; at the
    default threshold the collector rescans them constantly (about 8% of search
    time at 4 players, measured on Python 3.14).
    """
    thresholds = gc.get_threshold()
    gc.set_threshold(max(thresholds[0], _SEARCH_GC_THRESHOLD), *thresholds[1:])
    try:
        yield
    finally:
        gc.set_threshold(*thresholds)


@_infrequent_garbage_collection()
def play_game(
    env: HanabiEnv, agents: Sequence[Agent], seed: int, on_move: MoveCallback | None = None
) -> GameResult:
    """Plays one game to the end; agents evaluate their own positions one at a time.

    Args:
        env: Environment, created with the game's seed.
        agents: One agent per seat.
        seed: Seed the environment was created with (recorded in the result).
        on_move: Optional callback after every move.

    Raises:
        HanabiSearchError: If the acting agent returns no action.
    """
    session = GameSession(env, agents, seed, on_move)
    request = session.advance()
    while request is not None:
        request = session.advance(session.acting_agent.evaluate(request))
    assert session.result is not None
    return session.result


def play_games(
    game: GameConfig,
    agents: Sequence[Agent],
    seeds: Sequence[int],
    on_move: MoveCallback | None = None,
) -> list[GameResult]:
    """Plays one game per seed, one after the other, with the same agents.

    Args:
        game: Game variant.
        agents: One agent per seat, reused across games.
        seeds: Seed of each game.
        on_move: Optional callback after every move.
    """
    results = []
    for seed in seeds:
        result = play_game(HanabiEnv(game.to_params(seed)), agents, seed, on_move)
        logger.debug("Game seed=%d score=%d (%s)", seed, result.score, result.end_status.name)
        results.append(result)
    return results


@_infrequent_garbage_collection()
def play_games_batched(
    game: GameConfig,
    tables: Sequence[Sequence[Agent]],
    seeds: Sequence[int],
    network: AlphaZeroNetwork,
    on_move: MoveCallback | None = None,
    on_game_end: Callable[[GameResult], None] | None = None,
) -> list[GameResult]:
    """Plays one game per seed, `len(tables)` at a time, batching network evaluations.

    Every game runs until its acting agent needs a position evaluated; the pending
    positions of all games then go through `network` in one forward pass. A table
    whose game ends starts the next seed. Each game depends only on its seed and on
    the network's weights while it is played.

    Args:
        game: Game variant.
        tables: Agents of each concurrent game (one per seat); agents must not be
            shared between tables.
        seeds: Seed of each game.
        network: Answers every agent's evaluation requests.
        on_move: Optional callback after every move of any game.
        on_game_end: Optional callback when a game ends, in the order games end.

    Returns:
        One result per seed, in the order of `seeds`.

    Raises:
        ConfigError: If `tables` is empty.
    """
    if not tables:
        raise ConfigError("play_games_batched needs at least one table")
    queue = deque(enumerate(seeds))
    results: list[GameResult | None] = [None] * len(seeds)
    # Games waiting for an evaluation: (index in seeds, session, request).
    waiting: list[tuple[int, GameSession, EvaluationRequest]] = []

    def finish(index: int, session: GameSession) -> None:
        assert session.result is not None
        results[index] = session.result
        if on_game_end is not None:
            on_game_end(session.result)

    def start_next_game(table: Sequence[Agent]) -> None:
        while queue:
            index, seed = queue.popleft()
            session = GameSession(HanabiEnv(game.to_params(seed)), table, seed, on_move)
            request = session.advance()
            if request is not None:
                waiting.append((index, session, request))
                return
            finish(index, session)

    for table in tables:
        start_next_game(table)
    while waiting:
        batch = waiting
        waiting = []
        policies, values = network.predict(np.stack([request for _, _, request in batch]))
        for row, (index, session, _) in enumerate(batch):
            evaluation: Evaluation = (policies[row], float(values[row]))
            request = session.advance(evaluation)
            if request is not None:
                waiting.append((index, session, request))
            else:
                finish(index, session)
                start_next_game(session.agents)
    assert all(result is not None for result in results)
    return [result for result in results if result is not None]


def evaluate(
    game: GameConfig, agents: Sequence[Agent], num_games: int, first_seed: int
) -> ScoreSummary:
    """Plays `num_games` games on seeds `first_seed, first_seed + 1, ...` and summarizes scores."""
    results = play_games(game, agents, range(first_seed, first_seed + num_games))
    return ScoreSummary.from_scores(
        [result.score for result in results], game.players, game.max_score
    )
