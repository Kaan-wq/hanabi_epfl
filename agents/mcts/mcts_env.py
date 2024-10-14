from enum import IntEnum
import random

from agents.mcts.cython.sampler.mcts_sampler import MCTS_Sampler
from pyhanabi import HanabiMove, HanabiMoveType, AgentObservationType, CHANCE_PLAYER_ID, try_cdef, try_load
from rl_env import HanabiEnv


class DetermineType(IntEnum):
    """Move types, consistent with hanabi_lib/hanabi_move.h."""

    RESTORE = 0
    REPLACE = 1
    NONE = 2


class ScoreType(IntEnum):
    """Move types, consistent with hanabi_lib/hanabi_move.h."""

    SCORE = 0
    REGRET = 1
    PROGRESS = 2


class MCTS_Env(HanabiEnv):
    def __init__(self, config):
        self.mcts_player = config["mcts_player"]
        self.determine_type = config["determine_type"]
        self.score_type = config["score_type"]
        self.remember_hand = None
        self.sampler = MCTS_Sampler()
        super().__init__(config)

    def step(self, action):
        if isinstance(action, dict):
            move = self._build_move(action)
        elif isinstance(action, int):
            move = self.game.get_move(action)
        elif isinstance(action, HanabiMove):
            move = action
        else:
            raise ValueError("Expected action as dict or int, got: {}".format(action))

        actioned_card = None
        action_player = self.state.cur_player()

        if (
            move.type() == HanabiMoveType.DISCARD
            or move.type() == HanabiMoveType.PLAY
        ):
            actioned_card = self.state.player_hands()[self.state.cur_player()][
                move.card_index()
            ]

        self.state.apply_move(move)

        while self.state.cur_player() == CHANCE_PLAYER_ID:
            self.state.deal_random_card()

        if (
            self.determine_type == DetermineType.RESTORE
            and action_player != self.mcts_player
        ):
            self.restore_hand(
                action_player, self.remember_hand, actioned_card, move.card_index()
            )

        if self.determine_type != DetermineType.NONE and self.state.cur_player() != self.mcts_player:
            self.remember_hand = self.state.player_hands()[self.state.cur_player()]
            self.replace_hand(self.state.cur_player())

        observations = self._make_observation_all_players()
        reward = self.reward()
        done = self.state.is_terminal()
        info = {}

        return (observations, reward, done, info)

    def reward(self):
        """Custom reward function for use during RIS-MCTS rollouts
        This is therefore not the same as the overall game score
        """

        if self.score_type == ScoreType.PROGRESS:
            return self.progress()
        else:
            return self.score()

    def return_hand(self, player):
        """Return all cards from a player's hand to the deck
        Note: a return move retains card knowledge in each spot"""

        hand_size = len(self.state.player_hands()[player])
        for _ in range(hand_size):
            # Return card in the furthest left position, cards always shift left
            return_move = HanabiMove.get_return_move(card_index=0, player=player)
            self.state.apply_move(return_move)

    def replace_hand(self, player):
        '''Replace a player's hand with a new one sampled from the deck. Card knowledge is retained.'''

        hand_size = len(self.state.player_hands()[player])
        temp_observation = self.state.observation(player)
        card_knowledge = temp_observation.card_knowledge()[0]
        self.return_hand(player)
        replacement_hand = self.sampler.sample_hand(player, 
                                                    hand_size, 
                                                    self.state.player_hands(), 
                                                    self.state.discard_pile(), 
                                                    self.state.fireworks(), 
                                                    card_knowledge)

        for card_index in range(len(replacement_hand)):
            card = replacement_hand[card_index]
            deal_specific_move = HanabiMove.get_deal_specific_move(
                card_index, player, card.color(), card.rank()
            )
            self.state.apply_move(deal_specific_move)

    def restore_hand(
        self, player, remember_hand, removed_card=None, removed_card_index=-1
    ):
        """As best as possible, restore player's current hand as closely as possible to a remembered one
        remember_hand: A hand to match (usually remembered before a redeterminisation)
        removed_card: This card was played or discarded. Used to resolve intra-hand conflict.
        removed_card_index: This card was played or discarded. Used to skip over it.
        """

        temp_observation = self.state.observation(player)
        card_knowledge = temp_observation.card_knowledge()[0]
        hand_size = len(self.state.player_hands()[player])
        self.return_hand(player)

        card_index = 0

        for remember_card_index in range(len(remember_hand)):
            if remember_card_index == removed_card_index:
                continue

            card = remember_hand[remember_card_index]

            if removed_card and card == removed_card:
                additional_cards = [
                    remember_hand[i]
                    for i in range(remember_card_index + 1, len(remember_hand))
                    if i != removed_card_index
                ]
                valid_cards = self.sampler.valid_cards(
                    player,
                    card_index,
                    self.state.player_hands(),
                    self.state.discard_pile(),
                    self.state.fireworks(),
                    card_knowledge,
                    additional_cards,
                )

                if not any(c == card for c in valid_cards):
                    if len(valid_cards) > 0:
                        card = random.choice(valid_cards)
                    else:
                        self.state.remove_knowledge(player, card_index)
                        card = self.sampler.sample_card(
                            player,
                            card_index,
                            self.state.player_hands(),
                            self.state.discard_pile(),
                            self.state.fireworks(),
                            None,
                            additional_cards,
                        )

            deal_specific_move = HanabiMove.get_deal_specific_move(
                card_index, player, card.color(), card.rank()
            )
            self.state.apply_move(deal_specific_move)
            card_index += 1

        if hand_size > len(self.state.player_hands()[player]):
            card = self.sampler.sample_card(
                player,
                card_index,
                self.state.player_hands(),
                self.state.discard_pile(),
                self.state.fireworks(),
                card_knowledge,
            )
            deal_specific_move = HanabiMove.get_deal_specific_move(
                card_index, player, card.color(), card.rank()
            )
            self.state.apply_move(deal_specific_move)


def make(
    environment_name="Hanabi-Full",
    num_players=2,
    mcts_player=0,
    determine_type=0,
    score_type=0,
    pyhanabi_path=None,
):
    """Make an environment.

    Args:
      environment_name: str, Name of the environment to instantiate.
      num_players: int, Number of players in this game.
      pyhanabi_path: str, absolute path to header files for c code linkage.

    Returns:
      env: An `Environment` object.

    Raises:
      ValueError: Unknown environment name.
    """

    if pyhanabi_path is not None:
        prefixes = (pyhanabi_path,)
        assert try_cdef(prefixes=prefixes), "cdef failed to load"
        assert try_load(prefixes=prefixes), "library failed to load"

    if (
        environment_name == "Hanabi-Full"
        or environment_name == "Hanabi-Full-CardKnowledge"
    ):
        return MCTS_Env(
            config={
                "colors": 5,
                "ranks": 5,
                "players": num_players,
                "mcts_player": mcts_player,
                "determine_type": determine_type,
                "score_type": score_type,
                "max_information_tokens": 8,
                "max_life_tokens": 3,
                "observation_type": AgentObservationType.CARD_KNOWLEDGE.value,
            }
        )
    else:
        raise ValueError("Unknown environment {}".format(environment_name))
