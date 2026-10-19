"""Play games between the agents of an experiment config and report the scores.

Usage:
    uv run python scripts/play.py --config configs/vandenbergh_selfplay_2p.yaml --num-games 100
"""

import argparse
import logging
from pathlib import Path

from hanabi_search.config import ExperimentConfig, HumanAgentConfig
from hanabi_search.env import HanabiEnv, StepResult
from hanabi_search.evaluation import ScoreSummary, play_games
from hanabi_search.experiment import set_up
from hanabi_search.logging_utils import setup_logging


def parse_args() -> argparse.Namespace:
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--config", type=Path, required=True, help="Experiment YAML file")
    parser.add_argument("--num-games", type=int, default=10, help="Number of games to play")
    parser.add_argument("--seed", type=int, help="Seed of the first game (default: config seed)")
    parser.add_argument("--checkpoint", type=Path, help="Network weights for AlphaZero seats")
    parser.add_argument("--show-moves", action="store_true", help="Print every move")
    return parser.parse_args()


def print_move(env: HanabiEnv, result: StepResult) -> None:
    """Prints the move just made and the score."""
    last = env.state.move_history()[-1]
    print(f"  {last}  -> score {result.reward}")


def main() -> None:
    """Plays the games and prints their scores."""
    args = parse_args()
    setup_logging(logging.INFO)
    config = ExperimentConfig.from_yaml(args.config)
    has_human = any(isinstance(agent, HumanAgentConfig) for agent in config.agents)
    setup = set_up(config, args.checkpoint, ask=input, show=print)
    first_seed = config.seed if args.seed is None else args.seed
    on_move = print_move if args.show_moves or has_human else None

    scores = []
    for result in play_games(
        config.game, setup.agents, range(first_seed, first_seed + args.num_games), on_move
    ):
        print(f"seed {result.seed}: score {result.score} ({result.end_status.name})")
        scores.append(result.score)
    print(ScoreSummary.from_scores(scores, config.game.players, config.game.max_score))


if __name__ == "__main__":
    main()
