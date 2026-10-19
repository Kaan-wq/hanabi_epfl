"""A human player, or a human handing their seat over to a bot mid-game."""

from collections.abc import Callable
from typing import override

from hanabi_search.agents.base import Agent
from hanabi_search.agents.rule_based import RULE_BASED_AGENTS, RuleBasedAgent
from hanabi_search.config import GameConfig
from hanabi_search.env import HanabiState, MoveDict, ObservationDict

BOTS = {name.upper(): agent for name, agent in RULE_BASED_AGENTS.items()}

HELP = (
    "Actions: PLAY <i> | DISCARD <i> | REVEAL_COLOR <offset> <color> | "
    "REVEAL_RANK <offset> <rank 1-5> | BOT <name>  (cards and offsets start at 0)"
)


def parse_action(text: str) -> MoveDict | str:
    """Parses a typed action.

    Returns:
        The action, or an error message if the text is not a well-formed action.
    """
    words = text.upper().split()
    if not words:
        return HELP
    kind, args = words[0], words[1:]
    match kind, args:
        case (("PLAY" | "DISCARD"), [index]) if index.isdigit():
            return {"action_type": kind, "card_index": int(index)}
        case "REVEAL_COLOR", [offset, color] if offset.isdigit():
            return {"action_type": kind, "target_offset": int(offset), "color": color}
        case "REVEAL_RANK", [offset, rank] if offset.isdigit() and rank.isdigit():
            return {"action_type": kind, "target_offset": int(offset), "rank": int(rank) - 1}
        case _:
            return f"Cannot parse {text!r}. {HELP}"


class HumanAgent(Agent):
    """Asks a person for each move; `BOT <name>` hands the seat to a rule-based bot.

    Args:
        player_id: Absolute seat of this agent.
        game: The game being played.
        ask: Prompts the person and returns their answer (e.g. `input`).
        show: Displays text to the person (e.g. `print`).
    """

    def __init__(
        self,
        player_id: int,
        game: GameConfig,
        ask: Callable[[str], str],
        show: Callable[[str], None],
    ) -> None:
        super().__init__(player_id, game)
        self._ask = ask
        self._show = show
        self.bot: RuleBasedAgent | None = None

    @override
    def act(self, observation: ObservationDict, state: HanabiState) -> MoveDict | None:
        if observation["current_player_offset"] != 0:
            return None
        if self.bot is not None:
            return self.bot.choose(observation)

        for item in reversed(observation["pyhanabi"].last_moves()):
            self._show(str(item))
        self._show(str(observation["pyhanabi"]))
        legal = observation["legal_moves"]
        while True:
            self._show("Legal actions: " + ", ".join(" ".join(map(str, m.values())) for m in legal))
            text = self._ask("Your action (or BOT <name>): ")
            words = text.upper().split()
            if words[:1] == ["BOT"]:
                if len(words) == 2 and words[1] in BOTS:
                    self.bot = BOTS[words[1]](self.player_id, self.game)
                    return self.bot.choose(observation)
                self._show(f"Unknown bot. Choose one of {', '.join(BOTS)}")
                continue
            action = parse_action(text)
            if isinstance(action, str):
                self._show(action)
            elif action in legal:
                return action
            else:
                self._show(f"Not legal here: {action}")
