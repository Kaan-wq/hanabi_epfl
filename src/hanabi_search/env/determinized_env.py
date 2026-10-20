"""Environment for MCTS rollouts that re-determinizes hidden hands as play proceeds.

Implements re-determinizing information-set MCTS: before another player acts,
their hand is resampled from what *they* know, so the searching player cannot
exploit hidden information. After they act, their real hand is restored as
closely as possible.
"""

import enum
import random
from collections.abc import Sequence

from hanabi_search.env._pyhanabi import engine_message, ffi, lib
from hanabi_search.env.game import GameParams
from hanabi_search.env.hanabi_env import HanabiEnv, StepResult
from hanabi_search.env.hand_sampler import resample_hand
from hanabi_search.env.observation_types import Action
from hanabi_search.exceptions import EngineCallError, IllegalMoveError


class DeterminizationType(enum.StrEnum):
    """How other players' hands are handled during rollouts."""

    RESTORE = "restore"  # Resample before they act, restore after.
    REPLACE = "replace"  # Resample before they act, never restore.
    NONE = "none"  # Never resample (the searcher sees true hands).


# The engine's codes for each determinization (hand_sampler.h, `Determinization`).
_DETERMINIZATION_CODES = {
    DeterminizationType.RESTORE: 0,
    DeterminizationType.REPLACE: 1,
    DeterminizationType.NONE: 2,
}


class RolloutReward(enum.StrEnum):
    """Reward returned by `DeterminizedEnv.reward`."""

    SCORE = "score"  # Game score: 0 once all lives are lost.
    PROGRESS = "progress"  # Cards on the fireworks, even after losing all lives.


class DeterminizedEnv(HanabiEnv):
    """A `HanabiEnv` whose `step` re-determinizes other players' hands.

    The caller sets `state` (usually a copy of the real game state) before stepping.

    Args:
        params: Game parameters, matching the real game.
        searching_player: Absolute id of the player running the search.
        determinization: How other players' hands are handled.
        rollout_reward: What `reward()` returns.
        rng: Source of the sampler's seeds; set it to the game's generator so
            rollouts are reproducible.
    """

    def __init__(
        self,
        params: GameParams,
        searching_player: int,
        determinization: DeterminizationType = DeterminizationType.RESTORE,
        rollout_reward: RolloutReward = RolloutReward.SCORE,
        rng: random.Random | None = None,
    ) -> None:
        super().__init__(params)
        self.rng = rng or random.Random()
        self.searching_player = searching_player
        self.determinization = determinization
        self.rollout_reward = rollout_reward
        # The hand of the player about to act, before it was resampled, kept
        # by the engine between calls (count -1: nothing remembered yet).
        self._remembered = ffi.new("pyhanabi_card_t[]", self.game.hand_size())
        self._num_remembered = ffi.new("int32_t*", -1)

    def step(self, action: Action) -> StepResult:
        """Applies an action like `advance`, then returns every player's observation."""
        self.advance(action)
        return StepResult(self.observe_all(), self.reward(), self.state.is_terminal())

    def advance(self, action: Action) -> None:
        """Applies an action, deals, then restores/resamples hands as configured.

        Builds no observations, unlike `step`: rollouts call `observe` for the
        one player who acts next.

        Raises:
            IllegalMoveError: If the action is not legal (or the game is over).
        """
        move = self.to_move(action)
        if not self.play_moves([self.game.get_move_uid(move)]):
            raise IllegalMoveError(f"Illegal move {move} in state:\n{self.state}")

    def play_moves(self, move_uids: Sequence[int]) -> int:
        """Applies moves in order as `advance` does, in a single engine call.

        Stops early at the first illegal move or once the game is over.

        Args:
            move_uids: Ids of the moves (see `HanabiGame.get_move_uid`).

        Returns:
            How many moves were applied.

        Raises:
            EngineCallError: If a move id is out of range or a hand cannot be swapped.
        """
        applied = int(
            lib.pyhanabi_state_play_determinized(
                self.state.c_state,
                move_uids,
                len(move_uids),
                self.searching_player,
                _DETERMINIZATION_CODES[self.determinization],
                self._remembered,
                self._num_remembered,
                self.rng.getrandbits(64),
            )
        )
        if applied < 0:
            raise EngineCallError(engine_message(f"Cannot play the moves {list(move_uids)}"))
        return applied

    def reward(self) -> int:
        """Returns the rollout reward selected by `rollout_reward`."""
        if self.rollout_reward == RolloutReward.PROGRESS:
            return self.progress()
        return self.score()

    def replace_hand(self, player: int) -> None:
        """Replaces `player`'s hand with one sampled from what they know."""
        resample_hand(self.state, player, self.rng.getrandbits(64))
