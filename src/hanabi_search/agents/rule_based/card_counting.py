"""Card counting over dict observations: which hidden cards are plausible, playable or useless.

Based on the ruleset from https://github.com/rocanaan/hanabi-ad-hoc-learning, with
bug fixes and additions by Michael Brooks (MBlogs).
"""

import numpy as np
import numpy.typing as npt

from hanabi_search.env import COLOR_CHAR, CardDict, ObservationDict

type Probabilities = npt.NDArray[np.float64]
# One entry per card identity, shape (num_colors, num_ranks).
type CardMask = npt.NDArray[np.bool_]


def colors_in_play(observation: ObservationDict) -> tuple[str, ...]:
    """Returns the color characters used in this game."""
    return COLOR_CHAR[: observation["num_colors"]]


def playable_card(card: CardDict, fireworks: dict[str, int]) -> bool:
    """Returns whether `card` can be played on `fireworks` right now."""
    if card["color"] is None:
        return False
    return card["rank"] == fireworks[card["color"]]


def useless_card(card: CardDict, fireworks: dict[str, int], max_fireworks: dict[str, int]) -> bool:
    """Returns whether `card` can never be played (already played, or unreachable)."""
    color = card["color"]
    assert color is not None
    return card["rank"] < fireworks[color] or card["rank"] >= max_fireworks[color]


def plausible_card_mask(
    observation: ObservationDict, player_offset: int, hand_index: int
) -> CardMask:
    """Returns which cards hints still allow for one card in the hand of `player_offset`."""
    mask: CardMask = observation["pyhanabi"].plausible_cards()[player_offset, hand_index]
    return mask


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


def _firework_heights(observation: ObservationDict) -> npt.NDArray[np.int64]:
    fireworks = observation["fireworks"]
    return np.array([fireworks[color] for color in colors_in_play(observation)])


def playable_cards(observation: ObservationDict) -> CardMask:
    """Returns which cards can be played on the fireworks right now."""
    ranks = np.arange(observation["num_ranks"])
    return np.equal(ranks, _firework_heights(observation)[:, None])


def useless_cards(observation: ObservationDict) -> CardMask:
    """Returns which cards can never be played (already played, or unreachable)."""
    ranks = np.arange(observation["num_ranks"])
    max_fireworks = get_max_fireworks(observation)
    max_heights = np.array([max_fireworks[color] for color in colors_in_play(observation)])
    return (ranks < _firework_heights(observation)[:, None]) | (ranks >= max_heights[:, None])


def _hand_probabilities(
    observation: ObservationDict, player_offset: int, matches: CardMask
) -> Probabilities:
    """For each card in a hand: P(card is in `matches`), weighting by unseen copies.

    Each card the hints allow counts once per copy the observer cannot see.
    """
    pyhanabi = observation["pyhanabi"]
    hand_size = len(observation["observed_hands"][player_offset])
    plausible = pyhanabi.plausible_cards()[player_offset, :hand_size]  # (hand, colors, ranks)
    weights = plausible * np.maximum(pyhanabi.unseen_card_counts(), 0)
    total = weights.sum(axis=(1, 2))
    matching = (weights * matches).sum(axis=(1, 2))
    return np.divide(matching, total, out=np.zeros(hand_size), where=total > 0)


def get_card_playability(observation: ObservationDict, player_offset: int = 0) -> Probabilities:
    """Returns, per card in the hand, the probability that it is playable now."""
    return _hand_probabilities(observation, player_offset, playable_cards(observation))


def get_probability_useless(observation: ObservationDict, player_offset: int = 0) -> Probabilities:
    """Returns, per card in the hand, the probability that it can never be played."""
    return _hand_probabilities(observation, player_offset, useless_cards(observation))


def get_probability_notcritical(
    observation: ObservationDict, player_offset: int = 0
) -> Probabilities:
    """Returns, per card in the hand, the probability that discarding it loses nothing.

    A card is not critical if it is useless or another unseen copy remains.
    """
    unseen = observation["pyhanabi"].unseen_card_counts()
    return _hand_probabilities(
        observation, player_offset, useless_cards(observation) | (unseen > 1)
    )
