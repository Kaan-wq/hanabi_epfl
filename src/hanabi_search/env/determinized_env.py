"""Environment for MCTS rollouts that re-determinizes hidden hands as play proceeds.

Implements re-determinizing information-set MCTS: before another player acts,
their hand is resampled from what *they* know, so the searching player cannot
exploit hidden information. After they act, their real hand is restored as
closely as possible.
"""

import enum
import random

from hanabi_search.env.cards import HanabiCard
from hanabi_search.env.game import GameParams
from hanabi_search.env.hanabi_env import HanabiEnv, StepResult
from hanabi_search.env.hand_sampler import resample_hand, restore_hand
from hanabi_search.env.moves import HanabiMoveType
from hanabi_search.env.observation_types import Action
from hanabi_search.exceptions import HanabiSearchError


def _sampling_seed() -> int:
    # Drawn from Python's generator, so seeding `random` (as each game does)
    # makes rollouts reproducible.
    return random.getrandbits(64)


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
    """

    def __init__(
        self,
        params: GameParams,
        searching_player: int,
        determinization: DeterminizationType = DeterminizationType.RESTORE,
        rollout_reward: RolloutReward = RolloutReward.SCORE,
    ) -> None:
        super().__init__(params)
        self.searching_player = searching_player
        self.determinization = determinization
        self.rollout_reward = rollout_reward
        self._remembered_hand: list[HanabiCard] | None = None

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
        acted_card = None
        if move.type() in (HanabiMoveType.DISCARD, HanabiMoveType.PLAY):
            acted_card = state.player_hand(acting_player)[move.card_index()]

        state.apply_move(move)
        self._deal_pending_cards()

        if (
            self.determinization == DeterminizationType.RESTORE
            and acting_player != self.searching_player
        ):
            if self._remembered_hand is None:
                raise HanabiSearchError("restore_hand called before any hand was remembered")
            self.restore_hand(acting_player, self._remembered_hand, acted_card, move.card_index())

        next_player = state.cur_player()
        if (
            self.determinization != DeterminizationType.NONE
            and next_player != self.searching_player
        ):
            self._remembered_hand = state.player_hand(next_player)
            self.replace_hand(next_player)

    def reward(self) -> int:
        """Returns the rollout reward selected by `rollout_reward`."""
        if self.rollout_reward == RolloutReward.PROGRESS:
            return self.progress()
        return self.score()

    def replace_hand(self, player: int) -> None:
        """Replaces `player`'s hand with one sampled from what they know."""
        resample_hand(self.state, player, _sampling_seed())

    def restore_hand(
        self,
        player: int,
        remembered_hand: list[HanabiCard],
        removed_card: HanabiCard | None = None,
        removed_card_index: int = -1,
    ) -> None:
        """Restores `player`'s hand as closely as possible to `remembered_hand`.

        Args:
            player: Absolute id of the player.
            remembered_hand: Hand to match, usually remembered before re-determinizing.
            removed_card: Card just played or discarded; resolves conflicts within the hand.
            removed_card_index: Position of that card in `remembered_hand`; skipped.
        """
        restore_hand(
            self.state,
            player,
            remembered_hand,
            removed_card,
            removed_card_index,
            _sampling_seed(),
        )
