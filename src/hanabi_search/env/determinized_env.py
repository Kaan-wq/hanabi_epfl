"""Environment for MCTS rollouts that re-determinizes hidden hands as play proceeds.

Implements re-determinizing information-set MCTS: before another player acts,
their hand is resampled from what *they* know, so the searching player cannot
exploit hidden information. After they act, their real hand is restored as
closely as possible.
"""

import enum
import random

from hanabi_search.env.card_sampler import CardSampler, DeckShape
from hanabi_search.env.cards import HanabiCard
from hanabi_search.env.game import GameParams
from hanabi_search.env.hanabi_env import HanabiEnv, StepResult
from hanabi_search.env.moves import HanabiMove, HanabiMoveType
from hanabi_search.env.observation_types import Action
from hanabi_search.exceptions import HanabiSearchError


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
        self.sampler = CardSampler(DeckShape.from_game(self.game))
        self._remembered_hand: list[HanabiCard] | None = None

    def step(self, action: Action) -> StepResult:
        """Applies an action, deals, then restores/resamples hands as configured."""
        move = self.to_move(action)
        state = self.state
        acting_player = state.cur_player()
        acted_card = None
        if move.type() in (HanabiMoveType.DISCARD, HanabiMoveType.PLAY):
            acted_card = state.player_hands()[acting_player][move.card_index()]

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
            self._remembered_hand = state.player_hands()[next_player]
            self.replace_hand(next_player)

        observations = self.observe_all()
        return StepResult(observations, self.reward(), state.is_terminal())

    def reward(self) -> int:
        """Returns the rollout reward selected by `rollout_reward`."""
        if self.rollout_reward == RolloutReward.PROGRESS:
            return self.progress()
        return self.score()

    def return_hand(self, player: int) -> None:
        """Returns all of `player`'s cards to the deck, keeping their card knowledge."""
        for _ in range(len(self.state.player_hands()[player])):
            # Cards shift left after each return, so always return position 0.
            self.state.apply_move(HanabiMove.get_return_move(card_index=0, player=player))

    def replace_hand(self, player: int) -> None:
        """Replaces `player`'s hand with one sampled from what they know."""
        state = self.state
        hand_size = len(state.player_hands()[player])
        card_knowledge = state.observation(player).card_knowledge()[0]
        self.return_hand(player)
        new_hand = self.sampler.sample_hand(
            player,
            hand_size,
            state.player_hands(),
            state.discard_pile(),
            state.fireworks(),
            card_knowledge,
        )
        for card_index, card in enumerate(new_hand):
            self._deal(card_index, player, card)

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
        state = self.state
        card_knowledge = state.observation(player).card_knowledge()[0]
        hand_size = len(state.player_hands()[player])
        self.return_hand(player)

        card_index = 0
        for remembered_index, remembered_card in enumerate(remembered_hand):
            if remembered_index == removed_card_index:
                continue
            card: HanabiCard | None = remembered_card
            if removed_card and remembered_card == removed_card:
                additional_cards = [
                    remembered_hand[i]
                    for i in range(remembered_index + 1, len(remembered_hand))
                    if i != removed_card_index
                ]
                valid_cards = self.sampler.valid_cards(
                    player,
                    card_index,
                    state.player_hands(),
                    state.discard_pile(),
                    state.fireworks(),
                    card_knowledge,
                    additional_cards,
                )
                if not any(valid == remembered_card for valid in valid_cards):
                    if valid_cards:
                        card = random.choice(valid_cards)
                    else:
                        state.remove_knowledge(player, card_index)
                        card = self.sampler.sample_card(
                            player,
                            card_index,
                            state.player_hands(),
                            state.discard_pile(),
                            state.fireworks(),
                            None,
                            additional_cards,
                        )
            self._deal(card_index, player, card)
            card_index += 1

        if hand_size > len(state.player_hands()[player]):
            card = self.sampler.sample_card(
                player,
                card_index,
                state.player_hands(),
                state.discard_pile(),
                state.fireworks(),
                card_knowledge,
            )
            self._deal(card_index, player, card)

    def _deal(self, card_index: int, player: int, card: HanabiCard | None) -> None:
        if card is None:
            raise HanabiSearchError(
                f"No consistent card to deal to player {player} at {card_index}"
            )
        move = HanabiMove.get_deal_specific_move(card_index, player, card.color(), card.rank())
        self.state.apply_move(move)
