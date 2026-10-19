import math

import pytest

from hanabi_search.evaluation import ScoreSummary


def test_summary_statistics() -> None:
    summary = ScoreSummary.from_scores([20, 25, 15, 25], num_players=2, max_score=25)
    assert summary.mean == pytest.approx(21.25)
    sample_std = math.sqrt(sum((s - 21.25) ** 2 for s in [20, 25, 15, 25]) / 3)
    assert summary.std_error == pytest.approx(sample_std / 2)
    assert summary.perfect_rate == 0.5
    assert "n=4" in str(summary)
    assert "2 players" in str(summary)


def test_single_game_has_zero_error() -> None:
    assert ScoreSummary.from_scores([7], 3, 25).std_error == 0


def test_empty_scores_raise() -> None:
    with pytest.raises(ValueError, match="zero games"):
        ScoreSummary.from_scores([], 2, 25)
