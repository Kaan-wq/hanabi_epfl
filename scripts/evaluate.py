"""Evaluate an experiment's agents on its fixed, held-out evaluation seeds.

Results are written to runs/<experiment>/<timestamp>_<seed>/evaluation.json.

Usage:
    uv run python scripts/evaluate.py --config configs/alphazero_2p.yaml \
        --checkpoint runs/alphazero_2p/<run>/model.pt --num-games 1000
"""

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from hanabi_search.config import ExperimentConfig
from hanabi_search.experiment import create_run_dir, evaluate_experiment


def parse_args() -> argparse.Namespace:
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--config", type=Path, required=True, help="Experiment YAML file")
    parser.add_argument("--checkpoint", type=Path, help="Network weights for AlphaZero seats")
    parser.add_argument("--num-games", type=int, help="Override evaluation.num_games")
    return parser.parse_args()


def main() -> None:
    """Evaluates the agents and saves the summary."""
    args = parse_args()
    config = ExperimentConfig.from_yaml(args.config)
    run_dir = create_run_dir(config)
    summary = evaluate_experiment(config, args.checkpoint, args.num_games)
    print(summary)
    record = {
        **asdict(summary),
        "first_seed": config.evaluation.seed,
        "checkpoint": None if args.checkpoint is None else str(args.checkpoint),
    }
    (run_dir / "evaluation.json").write_text(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
