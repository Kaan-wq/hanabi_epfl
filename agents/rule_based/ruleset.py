# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Michael Brooks (MBlogs) Declaration: This code was sourced at https://github.com/rocanaan/hanabi-ad-hoc-learning
# Edits were made for bug fixes and additions.

import random
import numpy as np
import pyhanabi

colors = ['R', 'Y', 'G', 'W', 'B']
num_in_deck_by_rank = [3, 2, 2, 2, 1]  # Rank is zero-based

def playable_card(card, fireworks):
    if isinstance(card, pyhanabi.HanabiCard):
        card = {'color': colors[card.color()], 'rank': card.rank()}
    if card['color'] is None or card['rank'] is None:
        return False
    return card['rank'] == fireworks[card['color']]

def useless_card(card, fireworks, max_fireworks):
    if isinstance(card, pyhanabi.HanabiCard):
        card = {'color': colors[card.color()], 'rank': card.rank()}
    color = card['color']
    rank = card['rank']
    if rank < fireworks[color]:
        return True
    if rank >= max_fireworks[color]:
        return True
    return False

def get_plausible_cards(observation, player_offset, hand_index):
    card_knowledge = observation['pyhanabi'].card_knowledge()[player_offset]
    hidden_card = card_knowledge[hand_index]
    plausible_colors = [c for c in range(5) if hidden_card.color_plausible(c)]
    plausible_ranks = [r for r in range(5) if hidden_card.rank_plausible(r)]
    plausible_cards = [
        pyhanabi.HanabiCard(color, rank)
        for color in plausible_colors
        for rank in plausible_ranks
    ]
    return plausible_cards

def get_observed_card_counts(observation):
    card_counts = {}
    num_players = observation['num_players']
    current_player = observation['current_player_offset']
    # Count cards in other players' hands
    for player_offset in range(num_players):
        if player_offset != current_player:
            their_hand = observation['observed_hands'][player_offset]
            for card in their_hand:
                key = (colors.index(card['color']), card['rank'])
                card_counts[key] = card_counts.get(key, 0) + 1
    # Count discarded cards
    for card in observation['discard_pile']:
        key = (colors.index(card['color']), card['rank'])
        card_counts[key] = card_counts.get(key, 0) + 1
    # Count cards in fireworks
    for color, rank in observation['fireworks'].items():
        color_index = colors.index(color)
        for r in range(rank):
            key = (color_index, r)
            card_counts[key] = card_counts.get(key, 0) + 1
    return card_counts

def get_remaining_counts(observation):
    total_cards = {
        (color_index, rank_index): num_in_deck_by_rank[rank_index]
        for color_index in range(5)
        for rank_index in range(5)
    }
    observed_counts = get_observed_card_counts(observation)
    for key, count in observed_counts.items():
        total_cards[key] -= count
    return total_cards

def get_card_playability(observation, player_offset=0):
    remaining_counts = get_remaining_counts(observation)
    fireworks = observation['fireworks']
    my_hand_size = len(observation['observed_hands'][player_offset])
    playability_array = np.zeros(my_hand_size)
    for hand_index in range(my_hand_size):
        total_possibilities = 0
        playable_possibilities = 0
        plausible_cards = get_plausible_cards(observation, player_offset, hand_index)
        for plausible in plausible_cards:
            key = (plausible.color(), plausible.rank())
            remaining_count = remaining_counts.get(key, 0)
            if remaining_count <= 0:
                continue
            total_possibilities += remaining_count
            if playable_card(plausible, fireworks):
                playable_possibilities += remaining_count
        if total_possibilities > 0:
            playability_array[hand_index] = playable_possibilities / total_possibilities
        else:
            playability_array[hand_index] = 0
    return playability_array

def get_probability_useless(observation, player_offset=0):
    remaining_counts = get_remaining_counts(observation)
    fireworks = observation['fireworks']
    max_fireworks = get_max_fireworks(observation)
    my_hand_size = len(observation['observed_hands'][player_offset])
    probability_useless = np.zeros(my_hand_size)
    for hand_index in range(my_hand_size):
        total_possibilities = 0
        useless_possibilities = 0
        plausible_cards = get_plausible_cards(observation, player_offset, hand_index)
        for plausible in plausible_cards:
            key = (plausible.color(), plausible.rank())
            remaining_count = remaining_counts.get(key, 0)
            if remaining_count <= 0:
                continue
            total_possibilities += remaining_count
            if useless_card(plausible, fireworks, max_fireworks):
                useless_possibilities += remaining_count
        if total_possibilities > 0:
            probability_useless[hand_index] = useless_possibilities / total_possibilities
        else:
            probability_useless[hand_index] = 0
    return probability_useless

def get_probability_notcritical(observation, player_offset=0):
    remaining_counts = get_remaining_counts(observation)
    fireworks = observation['fireworks']
    max_fireworks = get_max_fireworks(observation)
    my_hand_size = len(observation['observed_hands'][player_offset])
    probability_notcritical = np.zeros(my_hand_size)
    for hand_index in range(my_hand_size):
        total_possibilities = 0
        notcritical_possibilities = 0
        plausible_cards = get_plausible_cards(observation, player_offset, hand_index)
        for plausible in plausible_cards:
            key = (plausible.color(), plausible.rank())
            remaining_count = remaining_counts.get(key, 0)
            if remaining_count <= 0:
                continue
            total_possibilities += remaining_count
            is_useless = useless_card(plausible, fireworks, max_fireworks)
            is_not_critical = is_useless or remaining_count > 1
            if is_not_critical:
                notcritical_possibilities += remaining_count
        if total_possibilities > 0:
            probability_notcritical[hand_index] = notcritical_possibilities / total_possibilities
        else:
            probability_notcritical[hand_index] = 0
    return probability_notcritical

def get_max_fireworks(observation):
    discarded_counts = {}
    max_fireworks = {color: 5 for color in colors}
    for card in observation['discard_pile']:
        color = card['color']
        rank = card['rank']
        key = (color, rank)
        discarded_counts[key] = discarded_counts.get(key, 0) + 1
    for (color, rank), count in discarded_counts.items():
        if count >= num_in_deck_by_rank[rank]:
            if max_fireworks[color] > rank:
                max_fireworks[color] = rank
    return max_fireworks

class Ruleset():
    @staticmethod
    def playable_now_convention(observation):
        fireworks = observation["fireworks"]
        history_last_moves = observation["pyhanabi"].last_moves()
        history_moves = [h for h in history_last_moves if h.move().to_dict()["action_type"] not in ["DEAL", "RETURN", "DEAL_SPECIFIC"]]
        if not history_moves:
            return None
        history_move = history_moves[0]
        card_info_revealed = history_move.card_info_newly_revealed()
        action = history_move.move().to_dict()
        if card_info_revealed:
            if action["target_offset"] == 1:
                hand_index = card_info_revealed[0]
                if len(card_info_revealed) == 1:
                    plausible_cards = get_plausible_cards(observation, 0, hand_index)
                    if any(playable_card(plausible_card, fireworks) for plausible_card in plausible_cards):
                        return {'action_type': 'PLAY', 'card_index': hand_index}
        return None

    @staticmethod
    def discard_oldest_first(observation):
        if observation['information_tokens'] < 8:
            return {'action_type': 'DISCARD', 'card_index': 0}
        return None

    @staticmethod
    def osawa_discard(observation):
        if observation['information_tokens'] == 8:
            return None
        fireworks = observation['fireworks']
        max_fireworks = get_max_fireworks(observation)
        for card_index, card in enumerate(observation['card_knowledge'][0]):
            color = card['color']
            rank = card['rank']
            if color is not None:
                if fireworks[color] == 5:
                    return {'action_type': 'DISCARD', 'card_index': card_index}
                if rank is not None:
                    if rank < fireworks[color] or rank >= max_fireworks[color]:
                        return {'action_type': 'DISCARD', 'card_index': card_index}
            if rank is not None:
                if rank < min(fireworks.values()):
                    return {'action_type': 'DISCARD', 'card_index': card_index}
        # Discard card with highest probability of being useless
        probability_useless = get_probability_useless(observation)
        card_index = np.argmax(probability_useless)
        if probability_useless[card_index] > 0:
            return {'action_type': 'DISCARD', 'card_index': card_index}
        return None

    @staticmethod
    def tell_unknown(observation):
        PLAYER_OFFSET = 1
        if observation['information_tokens'] > 0:
            their_hand = observation['observed_hands'][PLAYER_OFFSET]
            their_knowledge = observation['card_knowledge'][PLAYER_OFFSET]
            for index, card in enumerate(their_knowledge):
                if card['color'] is None:
                    return {'action_type': 'REVEAL_COLOR', 'color': their_hand[index]['color'], 'target_offset': PLAYER_OFFSET}
                if card['rank'] is None:
                    return {'action_type': 'REVEAL_RANK', 'rank': their_hand[index]['rank'], 'target_offset': PLAYER_OFFSET}
        return None

    @staticmethod
    def discard_randomly(observation):
        if observation['information_tokens'] < 8:
            hand_size = len(observation['observed_hands'][0])
            discard_index = random.randint(0, hand_size - 1)
            return {'action_type': 'DISCARD', 'card_index': discard_index}
        return None

    @staticmethod
    def play_safe_card(observation):
        PLAYER_OFFSET = 0
        fireworks = observation['fireworks']
        for card_index in range(len(observation['card_knowledge'][PLAYER_OFFSET])):
            plausible_cards = get_plausible_cards(observation, PLAYER_OFFSET, card_index)
            if plausible_cards and all(playable_card(plausible, fireworks) for plausible in plausible_cards):
                return {'action_type': 'PLAY', 'card_index': card_index}
        return None

    @staticmethod
    def play_if_certain(observation):
        PLAYER_OFFSET = 0
        fireworks = observation['fireworks']
        for card_index, card in enumerate(observation['card_knowledge'][PLAYER_OFFSET]):
            color = card['color']
            rank = card['rank']
            if color is not None and rank is not None:
                if rank == fireworks[color]:
                    return {'action_type': 'PLAY', 'card_index': card_index}
        return None

    @staticmethod
    def tell_playable_card_outer(observation):
        fireworks = observation['fireworks']
        if observation['information_tokens'] > 0:
            for player_offset in range(1, observation['num_players']):
                player_hand = observation['observed_hands'][player_offset]
                player_hints = observation['card_knowledge'][player_offset]
                for index, (card, hint) in enumerate(zip(player_hand, player_hints)):
                    if playable_card(card, fireworks):
                        if hint['rank'] is None:
                            return {'action_type': 'REVEAL_RANK', 'rank': card['rank'], 'target_offset': player_offset}
                        elif hint['color'] is None:
                            return {'action_type': 'REVEAL_COLOR', 'color': card['color'], 'target_offset': player_offset}
        return None

    @staticmethod
    def tell_anyone_useful_card(observation):
        return Ruleset.tell_playable_card_outer(observation)

    @staticmethod
    def tell_most_information_factory(consider_hints=False):
        def tell_most_information(observation):
            if observation['information_tokens'] > 0:
                max_affected = -1
                best_action = None
                for player_offset in range(1, observation['num_players']):
                    player_hand = observation['observed_hands'][player_offset]
                    player_hints = observation['card_knowledge'][player_offset]
                    for card, hint in zip(player_hand, player_hints):
                        affected_colors = sum(1 for c in player_hand if c['color'] == card['color'])
                        affected_ranks = sum(1 for c in player_hand if c['rank'] == card['rank'])
                        if consider_hints:
                            if hint['color'] is not None:
                                affected_colors = 0
                            if hint['rank'] is not None:
                                affected_ranks = 0
                        if affected_colors > max_affected:
                            max_affected = affected_colors
                            best_action = {'action_type': 'REVEAL_COLOR', 'color': card['color'], 'target_offset': player_offset}
                        if affected_ranks > max_affected:
                            max_affected = affected_ranks
                            best_action = {'action_type': 'REVEAL_RANK', 'rank': card['rank'], 'target_offset': player_offset}
                return best_action
            return None
        return tell_most_information

    @staticmethod
    def tell_dispensable_factory(min_information_tokens=8):
        def tell_dispensable(observation):
            if observation['information_tokens'] < min_information_tokens and observation['information_tokens'] > 0:
                fireworks = observation['fireworks']
                for player_offset in range(1, observation['num_players']):
                    player_hand = observation['observed_hands'][player_offset]
                    player_hints = observation['card_knowledge'][player_offset]
                    for index, (card, hint) in enumerate(zip(player_hand, player_hints)):
                        color = card['color']
                        rank = card['rank']
                        known_color = hint['color']
                        known_rank = hint['rank']
                        if known_color is None and fireworks[color] == 5:
                            return {'action_type': 'REVEAL_COLOR', 'color': color, 'target_offset': player_offset}
                        if known_rank is None and rank < min(fireworks.values()):
                            return {'action_type': 'REVEAL_RANK', 'rank': rank, 'target_offset': player_offset}
                        if rank < fireworks[color]:
                            if known_color is None and known_rank is not None:
                                return {'action_type': 'REVEAL_COLOR', 'color': color, 'target_offset': player_offset}
                            if known_color is not None and known_rank is None:
                                return {'action_type': 'REVEAL_RANK', 'rank': rank, 'target_offset': player_offset}
            return None
        return tell_dispensable

    @staticmethod
    def tell_anyone_useless_card(observation):
        fireworks = observation['fireworks']
        max_fireworks = get_max_fireworks(observation)
        if observation['information_tokens'] > 0:
            for player_offset in range(1, observation['num_players']):
                player_hand = observation['observed_hands'][player_offset]
                player_hints = observation['card_knowledge'][player_offset]
                for index, (card, hint) in enumerate(zip(player_hand, player_hints)):
                    if useless_card(card, fireworks, max_fireworks):
                        if hint['color'] is None:
                            return {'action_type': 'REVEAL_COLOR', 'color': card['color'], 'target_offset': player_offset}
                        if hint['rank'] is None:
                            return {'action_type': 'REVEAL_RANK', 'rank': card['rank'], 'target_offset': player_offset}
        return None

    @staticmethod
    def tell_playable_card(observation):
        fireworks = observation['fireworks']
        if observation['information_tokens'] > 0:
            for player_offset in range(1, observation['num_players']):
                player_hand = observation['observed_hands'][player_offset]
                for card in player_hand:
                    if playable_card(card, fireworks):
                        if random.choice([True, False]):
                            return {'action_type': 'REVEAL_RANK', 'rank': card['rank'], 'target_offset': player_offset}
                        else:
                            return {'action_type': 'REVEAL_COLOR', 'color': card['color'], 'target_offset': player_offset}
        return None

    @staticmethod
    def legal_random(observation):
        if observation['current_player_offset'] == 0:
            return random.choice(observation['legal_moves'])
        else:
            return None

    @staticmethod
    def play_probably_safe_factory(threshold=0.95, require_extra_lives=False):
        def play_probably_safe_threshold(observation):
            playability_vector = get_card_playability(observation)
            card_index = np.argmax(playability_vector)
            if not require_extra_lives or observation['life_tokens'] > 1:
                if playability_vector[card_index] >= threshold:
                    return {'action_type': 'PLAY', 'card_index': card_index}
            return None
        return play_probably_safe_threshold

    @staticmethod
    def discard_probably_useless_factory(threshold=0.75):
        def discard_probably_useless_threshold(observation):
            if observation['information_tokens'] < 8:
                probability_useless = get_probability_useless(observation)
                card_index = np.argmax(probability_useless)
                if probability_useless[card_index] >= threshold:
                    return {'action_type': 'DISCARD', 'card_index': card_index}
            return None
        return discard_probably_useless_threshold

    @staticmethod
    def hail_mary(observation):
        if observation['deck_size'] == 0 and observation['life_tokens'] > 1:
            return Ruleset.play_probably_safe_factory(0.0)(observation)
        return None

    @staticmethod
    def discard_probably_notcritical_factory(threshold=0.75):
        def discard_probably_notcritical_threshold(observation):
            if observation['information_tokens'] < 8:
                probability_notcritical = get_probability_notcritical(observation)
                card_index = np.argmax(probability_notcritical)
                if probability_notcritical[card_index] >= threshold:
                    return {'action_type': 'DISCARD', 'card_index': card_index}
            return None
        return discard_probably_notcritical_threshold

    @staticmethod
    def complete_tell_useful(observation):
        fireworks = observation['fireworks']
        if observation['information_tokens'] > 0:
            for player_offset in range(1, observation['num_players']):
                player_hand = observation['observed_hands'][player_offset]
                player_hints = observation['card_knowledge'][player_offset]
                for index, (card, hint) in enumerate(zip(player_hand, player_hints)):
                    if playable_card(card, fireworks):
                        if hint['color'] is None and hint['rank'] is not None:
                            return {'action_type': 'REVEAL_COLOR', 'color': card['color'], 'target_offset': player_offset}
                        if hint['rank'] is None and hint['color'] is not None:
                            return {'action_type': 'REVEAL_RANK', 'rank': card['rank'], 'target_offset': player_offset}
        return None

    @staticmethod
    def complete_tell_dispensable(observation):
        fireworks = observation['fireworks']
        max_fireworks = get_max_fireworks(observation)
        if observation['information_tokens'] > 0:
            for player_offset in range(1, observation['num_players']):
                player_hand = observation['observed_hands'][player_offset]
                player_hints = observation['card_knowledge'][player_offset]
                for index, (card, hint) in enumerate(zip(player_hand, player_hints)):
                    if useless_card(card, fireworks, max_fireworks):
                        if hint['color'] is None and hint['rank'] is not None:
                            return {'action_type': 'REVEAL_COLOR', 'color': card['color'], 'target_offset': player_offset}
                        if hint['rank'] is None and hint['color'] is not None:
                            return {'action_type': 'REVEAL_RANK', 'rank': card['rank'], 'target_offset': player_offset}
        return None

    @staticmethod
    def complete_tell_unplayable(observation):
        fireworks = observation['fireworks']
        max_fireworks = get_max_fireworks(observation)
        if observation['information_tokens'] > 0:
            for player_offset in range(1, observation['num_players']):
                player_hand = observation['observed_hands'][player_offset]
                player_hints = observation['card_knowledge'][player_offset]
                for index, (card, hint) in enumerate(zip(player_hand, player_hints)):
                    if not useless_card(card, fireworks, max_fireworks) and not playable_card(card, fireworks):
                        if hint['color'] is None and hint['rank'] is not None:
                            return {'action_type': 'REVEAL_COLOR', 'color': card['color'], 'target_offset': player_offset}
                        if hint['rank'] is None and hint['color'] is not None:
                            return {'action_type': 'REVEAL_RANK', 'rank': card['rank'], 'target_offset': player_offset}
        return None

    @staticmethod
    def play_probably_safe_late_factory(threshold=0.4, deck_size=5):
        def play_probably_safe_late(observation):
            if observation['deck_size'] <= deck_size:
                return Ruleset.play_probably_safe_factory(threshold)(observation)
            return None
        return play_probably_safe_late

    @staticmethod
    def discard_most_confident(observation):
        if observation['information_tokens'] == 8:
            return None
        action = Ruleset.osawa_discard(observation)
        if action is None:
            action = Ruleset.discard_probably_notcritical_factory(0)(observation)
        if action is None:
            action = Ruleset.discard_oldest_first(observation)
        return action