"""One game in progress, paused whenever the acting agent needs a network evaluation."""

import random
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from hanabi_search.agents import Agent
from hanabi_search.agents.base import ActSteps, Evaluation, EvaluationRequest
from hanabi_search.env import Action, HanabiEndOfGameType, HanabiEnv, StepResult
from hanabi_search.exceptions import HanabiSearchError

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


class GameSession:
    """Plays one game, handing every evaluation request of the acting agent to the caller.

    Every agent gets the game's random generator, `random.Random(seed)`, so the game
    depends on its seed alone, however sessions are interleaved. When the game ends,
    `end_episode` is called on every agent and `result` is set.

    Args:
        env: Environment created with the game's seed; it is reset here.
        agents: One agent per seat, not used by any other session at the same time.
        seed: Seed `env` was created with (recorded in the result).
        on_move: Optional callback after every move.
    """

    def __init__(
        self,
        env: HanabiEnv,
        agents: Sequence[Agent],
        seed: int,
        on_move: MoveCallback | None = None,
    ) -> None:
        self.env = env
        self.agents = agents
        self.seed = seed
        self.result: GameResult | None = None
        self._on_move = on_move
        self._observations = env.reset()
        self._moves: list[str] = []
        self._steps: ActSteps | None = None
        rng = random.Random(seed)
        for agent in agents:
            agent.begin_episode(rng)

    @property
    def acting_agent(self) -> Agent:
        """The agent whose turn it is."""
        return self.agents[self._observations["current_player"]]

    def advance(self, evaluation: Evaluation | None = None) -> EvaluationRequest | None:
        """Plays until the acting agent requests an evaluation or the game ends.

        Args:
            evaluation: The answer to the request returned by the previous call;
                `None` on the first call.

        Returns:
            The next evaluation request, or `None` once the game is over.

        Raises:
            HanabiSearchError: If an agent returns no action on its turn, or if a
                pending request is not answered.
        """
        while self.result is None:
            try:
                if self._steps is None:
                    player = self._observations["current_player"]
                    self._steps = self.agents[player].act_steps(
                        self._observations["player_observations"][player], self.env.state
                    )
                    return next(self._steps)
                if evaluation is None:
                    raise HanabiSearchError("advance needs the evaluation of the pending request")
                return self._steps.send(evaluation)
            except StopIteration as finished:
                self._steps = None
                evaluation = None
                self._play(finished.value)
        return None

    def _play(self, action: Action | None) -> None:
        player = self._observations["current_player"]
        if action is None:
            raise HanabiSearchError(f"Agent in seat {player} returned no action on its turn")
        env = self.env
        move = env.to_move(action)
        self._moves.append(str(move))
        step = env.step(move)
        self._observations = step.observations
        if self._on_move is not None:
            self._on_move(env, step)
        if step.done:
            score = env.score()
            for agent in self.agents:
                agent.end_episode(score)
            self.result = GameResult(self.seed, score, env.state.end_of_game_status(), self._moves)
