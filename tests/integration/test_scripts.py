import json
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from tests.support import REPO_ROOT


def _run(script: str, *args: str, cwd: Path) -> str:
    completed = subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / script), *args],
        cwd=cwd,
        capture_output=True,
        text=True,
        check=False,
        timeout=600,
    )
    assert completed.returncode == 0, completed.stderr
    return completed.stdout


def test_play_script(tmp_path: Path) -> None:
    config = REPO_ROOT / "configs" / "vandenbergh_selfplay_2p.yaml"
    output = _run("play.py", "--config", str(config), "--num-games", "3", cwd=tmp_path)
    assert output.count("score") == 3
    assert "n=3" in output


def test_mcts_play_script(tmp_path: Path) -> None:
    config = yaml.safe_load((REPO_ROOT / "configs" / "mcts_vs_vandenbergh_3p.yaml").read_text())
    config["agents"][0]["rollouts"] = 3
    path = tmp_path / "mcts.yaml"
    path.write_text(yaml.safe_dump(config))
    output = _run("play.py", "--config", str(path), "--num-games", "1", cwd=tmp_path)
    assert "n=1" in output


@pytest.mark.slow
def test_train_then_evaluate(tmp_path: Path) -> None:
    config = {
        "name": "tiny_alphazero",
        "seed": 3,
        "game": {"players": 2},
        "agents": [{"kind": "alphazero", "rollouts": 2}],
        "network": {"hidden_sizes": [16]},
        "training": {"num_episodes": 3, "batch_size": 8, "parallel_games": 2},
        "evaluation": {"num_games": 2, "seed": 100},
    }
    config_path = tmp_path / "tiny.yaml"
    config_path.write_text(yaml.safe_dump(config))

    _run("train.py", "--config", str(config_path), cwd=tmp_path)
    (run_dir,) = (tmp_path / "runs" / "tiny_alphazero").iterdir()
    for name in ("config.yaml", "run.log", "model.pt", "checkpoint.json", "transitions.jsonl"):
        assert (run_dir / name).is_file(), name
    metrics = [json.loads(line) for line in (run_dir / "metrics.jsonl").read_text().splitlines()]
    assert [m["episode"] for m in metrics] == [0, 1, 2]
    assert sorted(m["seed"] for m in metrics) == [3, 4, 5]
    assert json.loads((run_dir / "checkpoint.json").read_text())["observation_encoder"]

    checkpoint = run_dir / "model.pt"
    output = _run(
        "evaluate.py", "--config", str(config_path), "--checkpoint", str(checkpoint), cwd=tmp_path
    )
    assert "n=2" in output
