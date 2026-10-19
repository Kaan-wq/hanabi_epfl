"""Card counting over dict observations: which hidden cards are plausible, playable or useless.

Based on the ruleset from https://github.com/rocanaan/hanabi-ad-hoc-learning, with
bug fixes and additions by Michael Brooks (MBlogs).
"""

from collections.abc import Callable

import numpy as np
import numpy.typing as npt

from hanabi_search.env import COLOR_CHAR, CardDict, HanabiCard, HanabiCardKnowledge, ObservationDict

type Probabilities = npt.NDArray[np.float64]


def colors_in_play(observation: ObservationDict) -> tuple[str, ...]:
    """Returns the color characters used in this game."""
    return COLOR_CHAR[: observation["num_colors"]]


def _as_card_dict(card: CardDict | HanabiCard) -> CardDict:
    if isinstance(card, HanabiCard):
        return {"color": COLOR_CHAR[card.color()], "rank": card.rank()}
    return card


def playable_card(card: CardDict | HanabiCard, fireworks: dict[str, int]) -> bool:
    """Returns whether `card` can be played on `fireworks` right now."""
    card = _as_card_dict(card)
    if card["color"] is None:
        return False
    return card["rank"] == fireworks[card["color"]]


def useless_card(
    card: CardDict | HanabiCard, fireworks: dict[str, int], max_fireworks: dict[str, int]
) -> bool:
    """Returns whether `card` can never be played (already played, or unreachable)."""
    card = _as_card_dict(card)
    color = card["color"]
    assert color is not None
    return card["rank"] < fireworks[color] or card["rank"] >= max_fireworks[color]


def plausible_cards(
    observation: ObservationDict, knowledge: HanabiCardKnowledge
) -> list[HanabiCard]:
    """Returns every (color, rank) that hints have not ruled out for a hidden card."""
    colors = [c for c in range(observation["num_colors"]) if knowledge.color_plausible(c)]
    ranks = [r for r in range(observation["num_ranks"]) if knowledge.rank_plausible(r)]
    return [HanabiCard(color, rank) for color in colors for rank in ranks]


def get_plausible_cards(
    observation: ObservationDict, player_offset: int, hand_index: int
) -> list[HanabiCard]:
    """Returns the plausible cards for one card in the hand of `player_offset`."""
    knowledge = observation["pyhanabi"].card_knowledge()[player_offset][hand_index]
    return plausible_cards(observation, knowledge)


def get_observed_card_counts(observation: ObservationDict) -> dict[tuple[int, int], int]:
    """Counts visible cards (others' hands, discards, fireworks) by (color, rank)."""
    counts: dict[tuple[int, int], int] = {}
    for offset in range(observation["num_players"]):
        if offset == observation["current_player_offset"]:
            continue
        for card in observation["observed_hands"][offset]:
            assert card["color"] is not None
            key = (COLOR_CHAR.index(card["color"]), card["rank"])
            counts[key] = counts.get(key, 0) + 1
    for card in observation["discard_pile"]:
        assert card["color"] is not None
        key = (COLOR_CHAR.index(card["color"]), card["rank"])
        counts[key] = counts.get(key, 0) + 1
    for color, height in observation["fireworks"].items():
        color_index = COLOR_CHAR.index(color)
        for rank in range(height):
            counts[color_index, rank] = counts.get((color_index, rank), 0) + 1
    return counts


def get_remaining_counts(observation: ObservationDict) -> dict[tuple[int, int], int]:
    """Counts cards not visible to the observer (deck plus hidden hands) by (color, rank)."""
    cards_per_rank = observation["cards_per_rank"]
    remaining = {
        (color, rank): cards_per_rank[rank]
        for color in range(observation["num_colors"])
        for rank in range(observation["num_ranks"])
    }
    for key, count in get_observed_card_counts(observation).items():
        remaining[key] -= count
    return remaining


def get_max_fireworks(observation: ObservationDict) -> dict[str, int]:
    """Returns the highest reachable firework per color, given the discards."""
    max_fireworks = dict.fromkeys(colors_in_play(observation), observation["num_ranks"])
    discarded: dict[tuple[str, int], int] = {}
    for card in observation["discard_pile"]:
        assert card["color"] is not None
        key = (card["color"], card["rank"])
        discarded[key] = discarded.get(key, 0) + 1
    for (color, rank), count in discarded.items():
        if count >= observation["cards_per_rank"][rank] and max_fireworks[color] > rank:
            max_fireworks[color] = rank
    return max_fireworks


def _hand_probabilities(
    observation: ObservationDict,
    player_offset: int,
    is_match: Callable[[HanabiCard, int], bool],
) -> Probabilities:
    """For each card in a hand: P(card satisfies `is_match`), weighting by remaining copies.

    `is_match(card, remaining_count)` decides whether a plausible card counts.
    """
    remaining_counts = get_remaining_counts(observation)
    hand_knowledge = observation["pyhanabi"].card_knowledge()[player_offset]
    hand_size = len(observation["observed_hands"][player_offset])
    probabilities = np.zeros(hand_size)
    for hand_index in range(hand_size):
        total = 0
        matching = 0
        for card in plausible_cards(observation, hand_knowledge[hand_index]):
            remaining = remaining_counts.get((card.color(), card.rank()), 0)
            if remaining <= 0:
                continue
            total += remaining
            if is_match(card, remaining):
                matching += remaining
        probabilities[hand_index] = matching / total if total > 0 else 0
    return probabilities


def get_card_playability(observation: ObservationDict, player_offset: int = 0) -> Probabilities:
    """Returns, per card in the hand, the probability that it is playable now."""
    fireworks = observation["fireworks"]
    return _hand_probabilities(
        observation, player_offset, lambda card, _: playable_card(card, fireworks)
    )


def get_probability_useless(observation: ObservationDict, player_offset: int = 0) -> Probabilities:
    """Returns, per card in the hand, the probability that it can never be played."""
    fireworks = observation["fireworks"]
    max_fireworks = get_max_fireworks(observation)
    return _hand_probabilities(
        observation, player_offset, lambda card, _: useless_card(card, fireworks, max_fireworks)
    )


def get_probability_notcritical(
    observation: ObservationDict, player_offset: int = 0
) -> Probabilities:
    """Returns, per card in the hand, the probability that discarding it loses nothing.

    A card is not critical if it is useless or another copy remains.
    """
    fireworks = observation["fireworks"]
    max_fireworks = get_max_fireworks(observation)
    return _hand_probabilities(
        observation,
        player_offset,
        lambda card, remaining: useless_card(card, fireworks, max_fireworks) or remaining > 1,
    )
