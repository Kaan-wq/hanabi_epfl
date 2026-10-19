"""Train AlphaZero by self-play, as defined by an experiment config.

Writes the resolved config, logs, training metrics and the network to
runs/<experiment>/<timestamp>_<seed>/.

Usage:
    uv run python scripts/train.py --config configs/alphazero_2p.yaml
"""

import argparse
from pathlib import Path

from hanabi_search.config import ExperimentConfig
from hanabi_search.experiment import create_run_dir, train_alphazero


def parse_args() -> argparse.Namespace:
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--config", type=Path, required=True, help="Experiment YAML file")
    return parser.parse_args()


def main() -> None:
    """Trains and saves the network."""
    args = parse_args()
    config = ExperimentConfig.from_yaml(args.config)
    run_dir = create_run_dir(config)
    checkpoint = train_alphazero(config, run_dir)
    print(f"Saved network to {checkpoint}")


if __name__ == "__main__":
    main()
