"""Hand-written rules: each maps an observation to an action, or `None` if it does not apply.

Rule-based agents try their rules in order and take the first action. MCTS uses
rules to restrict which moves it expands.

Based on the ruleset from https://github.com/rocanaan/hanabi-ad-hoc-learning, with
bug fixes and additions by Michael Brooks (MBlogs).
"""

import random
from collections.abc import Callable

import numpy as np

from hanabi_search.agents.rule_based.card_counting import (
    get_card_playability,
    get_max_fireworks,
    get_probability_notcritical,
    get_probability_useless,
    plausible_card_mask,
    playable_card,
    playable_cards,
    useless_card,
)
from hanabi_search.env import HanabiMoveType, MoveDict, ObservationDict

type Rule = Callable[[ObservationDict], MoveDict | None]

# Rules that hint a specific partner always target the next player.
_NEXT_PLAYER = 1
_SKIPPED_HISTORY_TYPES = frozenset(
    {HanabiMoveType.DEAL, HanabiMoveType.RETURN, HanabiMoveType.DEAL_SPECIFIC}
)


def _reveal_color(color: str | None, target_offset: int) -> MoveDict:
    return {"action_type": "REVEAL_COLOR", "color": color, "target_offset": target_offset}


def _reveal_rank(rank: int, target_offset: int) -> MoveDict:
    return {"action_type": "REVEAL_RANK", "rank": rank, "target_offset": target_offset}


def _play(card_index: int) -> MoveDict:
    return {"action_type": "PLAY", "card_index": card_index}


def _discard(card_index: int) -> MoveDict:
    return {"action_type": "DISCARD", "card_index": card_index}


def _can_discard(observation: ObservationDict) -> bool:
    return observation["information_tokens"] < observation["max_information_tokens"]


def _complete_partial_hint(
    observation: ObservationDict, wants_card: Callable[[int, int], bool]
) -> MoveDict | None:
    """Completes a half-known card (color xor rank hinted) the first time `wants_card` holds.

    `wants_card(player_offset, hand_index)` selects which cards are worth hinting.
    """
    if observation["information_tokens"] <= 0:
        return None
    for offset in range(1, observation["num_players"]):
        hand = observation["observed_hands"][offset]
        hints = observation["card_knowledge"][offset]
        for index, (card, hint) in enumerate(zip(hand, hints, strict=True)):
            if not wants_card(offset, index):
                continue
            if hint["color"] is None and hint["rank"] is not None:
                return _reveal_color(card["color"], offset)
            if hint["rank"] is None and hint["color"] is not None:
                return _reveal_rank(card["rank"], offset)
    return None


def playable_now_convention(observation: ObservationDict) -> MoveDict | None:
    """Plays a card if the last hint singled it out and it may be playable."""
    history = [
        item
        for item in observation["pyhanabi"].last_moves()
        if item.move().type() not in _SKIPPED_HISTORY_TYPES
    ]
    if not history:
        return None
    last = history[0]
    revealed = last.card_info_newly_revealed()
    action = last.move().to_dict()
    if revealed and action["target_offset"] == _NEXT_PLAYER and len(revealed) == 1:
        hand_index = revealed[0]
        plausible = plausible_card_mask(observation, 0, hand_index)
        if (plausible & playable_cards(observation)).any():
            return _play(hand_index)
    return None


def discard_oldest_first(observation: ObservationDict) -> MoveDict | None:
    """Discards the oldest card when discarding is allowed."""
    return _discard(0) if _can_discard(observation) else None


def osawa_discard(observation: ObservationDict) -> MoveDict | None:
    """Discards a card known to be useless; otherwise the one most likely useless."""
    if not _can_discard(observation):
        return None
    fireworks = observation["fireworks"]
    max_fireworks = get_max_fireworks(observation)
    num_ranks = observation["num_ranks"]
    for card_index, card in enumerate(observation["card_knowledge"][0]):
        color = card["color"]
        rank = card["rank"]
        if color is not None:
            if fireworks[color] == num_ranks:
                return _discard(card_index)
            if rank is not None and (rank < fireworks[color] or rank >= max_fireworks[color]):
                return _discard(card_index)
        if rank is not None and rank < min(fireworks.values()):
            return _discard(card_index)
    probability_useless = get_probability_useless(observation)
    card_index = int(np.argmax(probability_useless))
    return _discard(card_index) if probability_useless[card_index] > 0 else None


def tell_unknown(observation: ObservationDict) -> MoveDict | None:
    """Hints the next player about the first attribute they do not know."""
    if observation["information_tokens"] <= 0:
        return None
    hand = observation["observed_hands"][_NEXT_PLAYER]
    for index, hint in enumerate(observation["card_knowledge"][_NEXT_PLAYER]):
        if hint["color"] is None:
            return _reveal_color(hand[index]["color"], _NEXT_PLAYER)
        if hint["rank"] is None:
            return _reveal_rank(hand[index]["rank"], _NEXT_PLAYER)
    return None


def tell_randomly(observation: ObservationDict) -> MoveDict | None:
    """Hints the color or rank of a random card of the next player."""
    if observation["information_tokens"] <= 0:
        return None
    card = random.choice(observation["observed_hands"][_NEXT_PLAYER])
    if random.randint(0, 1) == 0:
        return _reveal_rank(card["rank"], _NEXT_PLAYER)
    return _reveal_color(card["color"], _NEXT_PLAYER)


def discard_randomly(observation: ObservationDict) -> MoveDict | None:
    """Discards a random card when discarding is allowed."""
    if not _can_discard(observation):
        return None
    hand_size = len(observation["observed_hands"][0])
    return _discard(random.randint(0, hand_size - 1))


def play_safe_card(observation: ObservationDict) -> MoveDict | None:
    """Plays a card that is playable whatever its hidden identity."""
    playable = playable_cards(observation)
    for card_index in range(len(observation["card_knowledge"][0])):
        plausible = plausible_card_mask(observation, 0, card_index)
        if plausible.any() and not (plausible & ~playable).any():
            return _play(card_index)
    return None


def play_if_certain(observation: ObservationDict) -> MoveDict | None:
    """Plays a card whose color and rank were both hinted and that is playable."""
    fireworks = observation["fireworks"]
    for card_index, card in enumerate(observation["card_knowledge"][0]):
        color = card["color"]
        if color is not None and card["rank"] is not None and card["rank"] == fireworks[color]:
            return _play(card_index)
    return None


def tell_playable_card_outer(observation: ObservationDict) -> MoveDict | None:
    """Hints the missing attribute of the first playable card in any partner's hand."""
    if observation["information_tokens"] <= 0:
        return None
    fireworks = observation["fireworks"]
    for offset in range(1, observation["num_players"]):
        hand = observation["observed_hands"][offset]
        hints = observation["card_knowledge"][offset]
        for card, hint in zip(hand, hints, strict=True):
            if playable_card(card, fireworks):
                if hint["rank"] is None:
                    return _reveal_rank(card["rank"], offset)
                if hint["color"] is None:
                    return _reveal_color(card["color"], offset)
    return None


def tell_anyone_useful_card(observation: ObservationDict) -> MoveDict | None:
    """Alias of `tell_playable_card_outer`."""
    return tell_playable_card_outer(observation)


def tell_most_information_factory(consider_hints: bool = False) -> Rule:
    """Returns a rule giving the hint that touches the most cards.

    Args:
        consider_hints: If true, an attribute already hinted on a card counts for nothing.
    """

    def tell_most_information(observation: ObservationDict) -> MoveDict | None:
        if observation["information_tokens"] <= 0:
            return None
        max_affected = -1
        best_action = None
        for offset in range(1, observation["num_players"]):
            hand = observation["observed_hands"][offset]
            hints = observation["card_knowledge"][offset]
            for card, hint in zip(hand, hints, strict=True):
                affected_colors = sum(1 for c in hand if c["color"] == card["color"])
                affected_ranks = sum(1 for c in hand if c["rank"] == card["rank"])
                if consider_hints:
                    if hint["color"] is not None:
                        affected_colors = 0
                    if hint["rank"] is not None:
                        affected_ranks = 0
                if affected_colors > max_affected:
                    max_affected = affected_colors
                    best_action = _reveal_color(card["color"], offset)
                if affected_ranks > max_affected:
                    max_affected = affected_ranks
                    best_action = _reveal_rank(card["rank"], offset)
        return best_action

    return tell_most_information


def tell_dispensable_factory(min_information_tokens: int | None = None) -> Rule:
    """Returns a rule hinting cards that can safely be discarded.

    Args:
        min_information_tokens: Only hint while fewer tokens than this remain;
            `None` means the game's maximum.
    """

    def tell_dispensable(observation: ObservationDict) -> MoveDict | None:
        threshold = (
            observation["max_information_tokens"]
            if min_information_tokens is None
            else min_information_tokens
        )
        if not 0 < observation["information_tokens"] < threshold:
            return None
        fireworks = observation["fireworks"]
        num_ranks = observation["num_ranks"]
        for offset in range(1, observation["num_players"]):
            hand = observation["observed_hands"][offset]
            hints = observation["card_knowledge"][offset]
            for card, hint in zip(hand, hints, strict=True):
                color, rank = card["color"], card["rank"]
                assert color is not None
                known_color, known_rank = hint["color"], hint["rank"]
                if known_color is None and fireworks[color] == num_ranks:
                    return _reveal_color(color, offset)
                if known_rank is None and rank < min(fireworks.values()):
                    return _reveal_rank(rank, offset)
                if rank < fireworks[color]:
                    if known_color is None and known_rank is not None:
                        return _reveal_color(color, offset)
                    if known_color is not None and known_rank is None:
                        return _reveal_rank(rank, offset)
        return None

    return tell_dispensable


def tell_anyone_useless_card(observation: ObservationDict) -> MoveDict | None:
    """Hints an attribute of the first useless card in any partner's hand."""
    if observation["information_tokens"] <= 0:
        return None
    fireworks = observation["fireworks"]
    max_fireworks = get_max_fireworks(observation)
    for offset in range(1, observation["num_players"]):
        hand = observation["observed_hands"][offset]
        hints = observation["card_knowledge"][offset]
        for card, hint in zip(hand, hints, strict=True):
            if useless_card(card, fireworks, max_fireworks):
                if hint["color"] is None:
                    return _reveal_color(card["color"], offset)
                if hint["rank"] is None:
                    return _reveal_rank(card["rank"], offset)
    return None


def tell_playable_card(observation: ObservationDict) -> MoveDict | None:
    """Hints a random attribute of the first playable card in any partner's hand."""
    if observation["information_tokens"] <= 0:
        return None
    fireworks = observation["fireworks"]
    for offset in range(1, observation["num_players"]):
        for card in observation["observed_hands"][offset]:
            if playable_card(card, fireworks):
                if random.choice([True, False]):
                    return _reveal_rank(card["rank"], offset)
                return _reveal_color(card["color"], offset)
    return None


def legal_random(observation: ObservationDict) -> MoveDict | None:
    """Picks a uniformly random legal move on the agent's turn."""
    if observation["current_player_offset"] != 0:
        return None
    return random.choice(observation["legal_moves"])


def play_probably_safe_factory(threshold: float = 0.95, require_extra_lives: bool = False) -> Rule:
    """Returns a rule playing the most-likely-playable card if P(playable) >= `threshold`.

    Args:
        threshold: Minimum probability that the card is playable.
        require_extra_lives: Only play while more than one life remains.
    """

    def play_probably_safe(observation: ObservationDict) -> MoveDict | None:
        playability = get_card_playability(observation)
        card_index = int(np.argmax(playability))
        if require_extra_lives and observation["life_tokens"] <= 1:
            return None
        return _play(card_index) if playability[card_index] >= threshold else None

    return play_probably_safe


def discard_probably_useless_factory(threshold: float = 0.75) -> Rule:
    """Returns a rule discarding the most-likely-useless card if P(useless) >= `threshold`."""

    def discard_probably_useless(observation: ObservationDict) -> MoveDict | None:
        if not _can_discard(observation):
            return None
        probability_useless = get_probability_useless(observation)
        card_index = int(np.argmax(probability_useless))
        return _discard(card_index) if probability_useless[card_index] >= threshold else None

    return discard_probably_useless


def hail_mary(observation: ObservationDict) -> MoveDict | None:
    """With an empty deck and lives to spare, plays the most-likely-playable card."""
    if observation["deck_size"] == 0 and observation["life_tokens"] > 1:
        return play_probably_safe_factory(0.0)(observation)
    return None


def discard_probably_notcritical_factory(threshold: float = 0.75) -> Rule:
    """Returns a rule discarding the card most likely not critical if P >= `threshold`."""

    def discard_probably_notcritical(observation: ObservationDict) -> MoveDict | None:
        if not _can_discard(observation):
            return None
        probability_notcritical = get_probability_notcritical(observation)
        card_index = int(np.argmax(probability_notcritical))
        if probability_notcritical[card_index] >= threshold:
            return _discard(card_index)
        return None

    return discard_probably_notcritical


def complete_tell_useful(observation: ObservationDict) -> MoveDict | None:
    """Completes a half-known playable card in a partner's hand."""
    fireworks = observation["fireworks"]
    hands = observation["observed_hands"]
    return _complete_partial_hint(
        observation, lambda offset, index: playable_card(hands[offset][index], fireworks)
    )


def complete_tell_dispensable(observation: ObservationDict) -> MoveDict | None:
    """Completes a half-known useless card in a partner's hand."""
    fireworks = observation["fireworks"]
    max_fireworks = get_max_fireworks(observation)
    hands = observation["observed_hands"]
    return _complete_partial_hint(
        observation,
        lambda offset, index: useless_card(hands[offset][index], fireworks, max_fireworks),
    )


def complete_tell_unplayable(observation: ObservationDict) -> MoveDict | None:
    """Completes a half-known card that is neither playable nor useless."""
    fireworks = observation["fireworks"]
    max_fireworks = get_max_fireworks(observation)
    hands = observation["observed_hands"]

    def unplayable_but_useful(offset: int, index: int) -> bool:
        card = hands[offset][index]
        return not useless_card(card, fireworks, max_fireworks) and not playable_card(
            card, fireworks
        )

    return _complete_partial_hint(observation, unplayable_but_useful)


def play_probably_safe_late_factory(threshold: float = 0.4, deck_size: int = 5) -> Rule:
    """Returns `play_probably_safe_factory(threshold)` restricted to deck size <= `deck_size`."""

    def play_probably_safe_late(observation: ObservationDict) -> MoveDict | None:
        if observation["deck_size"] <= deck_size:
            return play_probably_safe_factory(threshold)(observation)
        return None

    return play_probably_safe_late


def discard_most_confident(observation: ObservationDict) -> MoveDict | None:
    """Discards by `osawa_discard`, else the least critical card, else the oldest."""
    if not _can_discard(observation):
        return None
    return (
        osawa_discard(observation)
        or discard_probably_notcritical_factory(0)(observation)
        or discard_oldest_first(observation)
    )
