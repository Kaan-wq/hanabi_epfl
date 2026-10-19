"""Shared helpers and constants for tests."""

import random
from pathlib import Path

from hanabi_search.config import GameConfig
from hanabi_search.env import HanabiEnv, ObservationDict

REPO_ROOT = Path(__file__).resolve().parents[1]
PLAYER_COUNTS = (2, 3, 4, 5)

# A reduced game, to catch code that assumes the full game.
SMALL_GAME = GameConfig(
    players=2, colors=2, ranks=5, hand_size=2, max_information_tokens=3, max_life_tokens=1
)


def make_env(game: GameConfig, seed: int = 0) -> HanabiEnv:
    env = HanabiEnv(game.to_params(seed))
    env.reset()
    return env


def current_observation(env: HanabiEnv) -> ObservationDict:
    observations = env.observe_all()
    return observations["player_observations"][observations["current_player"]]


def advance_randomly(env: HanabiEnv, num_moves: int, seed: int = 0) -> None:
    """Plays random legal moves, avoiding plays (which can end the game early)."""
    rng = random.Random(seed)
    for _ in range(num_moves):
        if env.state.is_terminal():
            return
        moves = [m for m in env.state.legal_moves() if m.type().name != "PLAY"]
        env.step(rng.choice(moves or env.state.legal_moves()))
