"""Environment for MCTS rollouts that re-determinizes hidden hands as play proceeds.

Implements re-determinizing information-set MCTS: before another player acts,
their hand is resampled from what *they* know, so the searching player cannot
exploit hidden information. After they act, their real hand is restored as
closely as possible.
"""

import enum
import random

from hanabi_search.env._pyhanabi import CData, card_values, engine_message, ffi, lib
from hanabi_search.env.game import GameParams
from hanabi_search.env.hanabi_env import HanabiEnv, StepResult
from hanabi_search.env.hand_sampler import resample_hand
from hanabi_search.env.moves import HanabiMoveType
from hanabi_search.env.observation_types import Action
from hanabi_search.exceptions import EngineCallError, HanabiSearchError

_CARD_MOVES = (HanabiMoveType.DISCARD, HanabiMoveType.PLAY)


class DeterminizationType(enum.StrEnum):
    """How other players' hands are handled during rollouts."""

    RESTORE = "restore"  # Resample before they act, restore after.
    REPLACE = "replace"  # Resample before they act, never restore.
    NONE = "none"  # Never resample (the searcher sees true hands).


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
        # The hand of the player about to act, before it was resampled, kept in
        # engine card buffers: no Python card objects on this hot path.
        self._remembered = ffi.new("pyhanabi_card_t[]", self.game.hand_size())
        self._num_remembered = -1  # Nothing remembered yet.
        self._acted_card = ffi.new("pyhanabi_card_t*")

    def step(self, action: Action) -> StepResult:
        """Applies an action like `advance`, then returns every player's observation."""
        self.advance(action)
        return StepResult(self.observe_all(), self.reward(), self.state.is_terminal())

    def advance(self, action: Action) -> None:
        """Applies an action, deals, then restores/resamples hands as configured.

        Builds no observations, unlike `step`: rollouts call `observe` for the
        one player who acts next.
        """
        move = self.to_move(action)
        state = self.state
        acting_player = state.cur_player()
        acted_color = acted_rank = -1
        if move.type() in _CARD_MOVES:
            self._read_card(acting_player, move.card_index(), self._acted_card)
            acted_color, acted_rank = card_values(self._acted_card)

        state.apply_move(move)
        self._deal_pending_cards()

        if (
            self.determinization == DeterminizationType.RESTORE
            and acting_player != self.searching_player
        ):
            self._restore_hand(acting_player, acted_color, acted_rank, move.card_index())

        next_player = state.cur_player()
        if (
            self.determinization != DeterminizationType.NONE
            and next_player != self.searching_player
        ):
            self._remember_hand(next_player)
            self.replace_hand(next_player)

    def reward(self) -> int:
        """Returns the rollout reward selected by `rollout_reward`."""
        if self.rollout_reward == RolloutReward.PROGRESS:
            return self.progress()
        return self.score()

    def replace_hand(self, player: int) -> None:
        """Replaces `player`'s hand with one sampled from what they know."""
        resample_hand(self.state, player, self.rng.getrandbits(64))

    def _read_card(self, player: int, index: int, card: CData) -> None:
        if not lib.StateGetHandCard(self.state.c_state, player, index, card):
            raise EngineCallError(engine_message(f"No card {index} in the hand of {player}"))

    def _remember_hand(self, player: int) -> None:
        count = lib.StateGetHandSize(self.state.c_state, player)
        remembered = self._remembered
        for index in range(count):
            self._read_card(player, index, remembered + index)
        self._num_remembered = count

    def _restore_hand(
        self, player: int, removed_color: int, removed_rank: int, removed_index: int
    ) -> None:
        """Restores `player`'s hand to the remembered one (see `hand_sampler.restore_hand`)."""
        if self._num_remembered < 0:
            raise HanabiSearchError("restore_hand called before any hand was remembered")
        if not lib.pyhanabi_state_restore_hand(
            self.state.c_state,
            player,
            self._remembered,
            self._num_remembered,
            removed_color,
            removed_rank,
            removed_index,
            self.rng.getrandbits(64),
        ):
            raise EngineCallError(engine_message(f"Cannot restore the hand of player {player}"))
