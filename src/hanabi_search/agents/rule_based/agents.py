"""Rule-based agents from the literature, built from `ruleset` rules.

Originally from https://github.com/rocanaan/hanabi-ad-hoc-learning.
"""

from collections.abc import Callable, Sequence
from typing import override

from hanabi_search.agents.base import Agent
from hanabi_search.agents.rule_based import ruleset
from hanabi_search.agents.rule_based.ruleset import Rule
from hanabi_search.config import GameConfig, RuleBasedPolicy
from hanabi_search.env import Action, HanabiState, MoveDict, ObservationDict


class RuleBasedAgent(Agent):
    """Takes the first action produced by its rules; a random legal move if none apply.

    Args:
        player_id: Absolute seat of this agent.
        game: The game being played.
        rules: Rules to try, in priority order.
    """

    def __init__(self, player_id: int, game: GameConfig, rules: Sequence[Rule]) -> None:
        super().__init__(player_id, game)
        self.rules = tuple(rules)

    def choose(self, observation: ObservationDict) -> MoveDict | None:
        """Returns the action for `observation`, or `None` if it is not this agent's turn."""
        if observation["current_player_offset"] != 0:
            return None
        for rule in self.rules:
            action = rule(observation, self.rng)
            if action is not None:
                return action
        return ruleset.legal_random(observation, self.rng)

    @override
    def act(self, observation: ObservationDict, state: HanabiState) -> Action | None:
        return self.choose(observation)


class LegalRandomAgent(RuleBasedAgent):
    """Plays a uniformly random legal move."""

    def __init__(self, player_id: int, game: GameConfig) -> None:
        super().__init__(player_id, game, [ruleset.legal_random])


class VanDenBerghAgent(RuleBasedAgent):
    """High-performing rule-based agent (Van den Bergh et al.).

    `act` runs the engine's port of these rules (engine/search/van_den_bergh.h);
    `choose` runs them in Python, the reference the port is tested against.
    """

    def __init__(self, player_id: int, game: GameConfig) -> None:
        super().__init__(
            player_id,
            game,
            [
                ruleset.play_probably_safe_factory(0.6, True),
                ruleset.play_safe_card,
                ruleset.discard_probably_useless_factory(0.99),
                ruleset.tell_anyone_useful_card,
                ruleset.tell_anyone_useless_card,
                ruleset.tell_most_information_factory(False),
                ruleset.discard_probably_useless_factory(0),
            ],
        )

    @override
    def act(self, observation: ObservationDict, state: HanabiState) -> Action | None:
        if observation["current_player_offset"] != 0:
            return None
        uid = observation["pyhanabi"].van_den_bergh_move_uid()
        if uid is None:
            return ruleset.legal_random(observation, self.rng)
        return uid


class FlawedAgent(RuleBasedAgent):
    """Plays cards on little information and hints randomly."""

    def __init__(self, player_id: int, game: GameConfig) -> None:
        super().__init__(
            player_id,
            game,
            [
                ruleset.play_safe_card,
                ruleset.play_probably_safe_factory(0.25),
                ruleset.tell_randomly,
                ruleset.osawa_discard,
                ruleset.discard_oldest_first,
                ruleset.discard_randomly,
            ],
        )


class IGGIAgent(RuleBasedAgent):
    """Simple heuristic agent from the IGGI Hanabi framework."""

    def __init__(self, player_id: int, game: GameConfig) -> None:
        super().__init__(
            player_id,
            game,
            [
                ruleset.play_if_certain,
                ruleset.play_safe_card,
                ruleset.tell_playable_card_outer,
                ruleset.osawa_discard,
                ruleset.discard_oldest_first,
                ruleset.legal_random,
            ],
        )


class PiersAgent(RuleBasedAgent):
    """Heuristic agent that gambles on plays once the deck is empty."""

    def __init__(self, player_id: int, game: GameConfig) -> None:
        super().__init__(
            player_id,
            game,
            [
                ruleset.hail_mary,
                ruleset.play_safe_card,
                ruleset.play_probably_safe_factory(0.6, True),
                ruleset.tell_anyone_useful_card,
                ruleset.tell_dispensable_factory(3),
                ruleset.osawa_discard,
                ruleset.discard_oldest_first,
                ruleset.tell_randomly,
                ruleset.discard_randomly,
            ],
        )


class OuterAgent(RuleBasedAgent):
    """Hints playable cards and otherwise unknown attributes."""

    def __init__(self, player_id: int, game: GameConfig) -> None:
        super().__init__(
            player_id,
            game,
            [
                ruleset.play_safe_card,
                ruleset.osawa_discard,
                ruleset.tell_playable_card_outer,
                ruleset.tell_unknown,
                ruleset.discard_randomly,
            ],
        )


class InnerAgent(RuleBasedAgent):
    """Hints playable cards with a random attribute, otherwise hints randomly."""

    def __init__(self, player_id: int, game: GameConfig) -> None:
        super().__init__(
            player_id,
            game,
            [
                ruleset.play_safe_card,
                ruleset.osawa_discard,
                ruleset.tell_playable_card,
                ruleset.tell_randomly,
                ruleset.discard_randomly,
            ],
        )


class MuteAgent(RuleBasedAgent):
    """Never hints."""

    def __init__(self, player_id: int, game: GameConfig) -> None:
        super().__init__(
            player_id,
            game,
            [
                ruleset.play_probably_safe_factory(0.6, True),
                ruleset.play_safe_card,
                ruleset.discard_probably_useless_factory(0.99),
                ruleset.discard_probably_useless_factory(0),
                ruleset.play_probably_safe_factory(0, False),
            ],
        )


RULE_BASED_AGENTS: dict[RuleBasedPolicy, Callable[[int, GameConfig], RuleBasedAgent]] = {
    "VanDenBergh": VanDenBerghAgent,
    "Flawed": FlawedAgent,
    "Outer": OuterAgent,
    "Inner": InnerAgent,
    "Piers": PiersAgent,
    "IGGI": IGGIAgent,
    "LegalRandom": LegalRandomAgent,
    "Mute": MuteAgent,
}
