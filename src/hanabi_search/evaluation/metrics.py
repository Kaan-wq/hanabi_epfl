"""Score statistics over many games."""

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Self


@dataclass(frozen=True, slots=True)
class ScoreSummary:
    """Mean score with its standard error, and the perfect-game rate.

    Attributes:
        num_players: Players per game.
        num_games: Number of games.
        mean: Mean final score.
        std_error: Standard error of the mean (sample standard deviation / sqrt(n)).
        perfect_rate: Fraction of games reaching `max_score`.
        max_score: Score of a perfect game.
    """

    num_players: int
    num_games: int
    mean: float
    std_error: float
    perfect_rate: float
    max_score: int

    @classmethod
    def from_scores(cls, scores: Sequence[int], num_players: int, max_score: int) -> Self:
        """Summarizes final scores.

        Raises:
            ValueError: If `scores` is empty.
        """
        if not scores:
            raise ValueError("Cannot summarize zero games")
        n = len(scores)
        mean = sum(scores) / n
        variance = sum((s - mean) ** 2 for s in scores) / (n - 1) if n > 1 else 0.0
        return cls(
            num_players=num_players,
            num_games=n,
            mean=mean,
            std_error=math.sqrt(variance / n),
            perfect_rate=sum(s == max_score for s in scores) / n,
            max_score=max_score,
        )

    def __str__(self) -> str:
        return (
            f"{self.mean:.2f} ± {self.std_error:.2f} (n={self.num_games}, "
            f"{self.num_players} players, perfect {self.perfect_rate:.1%})"
        )
