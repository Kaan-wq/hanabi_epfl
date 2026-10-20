import pytest

from hanabi_search.agents import (
    AgentResources,
    AlphaZeroAgent,
    HumanAgent,
    MCTSAgent,
    build_agent,
    build_agents,
)
from hanabi_search.agents.human import parse_action
from hanabi_search.agents.rule_based import PiersAgent
from hanabi_search.config import (
    AlphaZeroAgentConfig,
    GameConfig,
    HumanAgentConfig,
    MCTSAgentConfig,
    RuleBasedAgentConfig,
)
from hanabi_search.env import HanabiEnv
from hanabi_search.exceptions import ConfigError
from hanabi_search.networks import AlphaZeroNetwork
from tests.support import current_observation


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("play 2", {"action_type": "PLAY", "card_index": 2}),
        ("DISCARD 0", {"action_type": "DISCARD", "card_index": 0}),
        ("reveal_color 1 r", {"action_type": "REVEAL_COLOR", "target_offset": 1, "color": "R"}),
        ("reveal_rank 1 3", {"action_type": "REVEAL_RANK", "target_offset": 1, "rank": 2}),
    ],
)
def test_parse_action(text: str, expected: dict[str, object]) -> None:
    assert parse_action(text) == expected


@pytest.mark.parametrize("text", ["", "play", "play x", "reveal_rank 1", "dance 3"])
def test_parse_action_rejects_malformed_input(text: str) -> None:
    assert isinstance(parse_action(text), str)


def test_human_agent_retries_until_legal(env: HanabiEnv) -> None:
    observation = current_observation(env)
    rank = observation["observed_hands"][1][0]["rank"]
    answers = iter(["nonsense", "DISCARD 0", f"REVEAL_RANK 1 {rank + 1}"])
    shown: list[str] = []
    agent = HumanAgent(
        observation["current_player"], GameConfig(), lambda _: next(answers), shown.append
    )
    action = agent.act(observation, env.state)
    assert action == {"action_type": "REVEAL_RANK", "target_offset": 1, "rank": rank}
    assert any("Not legal" in line for line in shown)  # Discarding with full tokens.


def test_human_agent_hands_over_to_a_bot(env: HanabiEnv) -> None:
    observation = current_observation(env)
    answers = iter(["BOT nobody", "BOT VANDENBERGH"])
    shown: list[str] = []
    agent = HumanAgent(
        observation["current_player"], GameConfig(), lambda _: next(answers), shown.append
    )
    action = agent.act(observation, env.state)
    assert action is not None
    assert env.state.move_is_legal(env.to_move(action))
    assert agent.bot is not None
    assert any("Unknown bot" in line for line in shown)


def test_build_agents_for_every_kind() -> None:
    game = GameConfig(players=2)
    env = HanabiEnv(game.to_params())
    network = AlphaZeroNetwork(env.num_moves(), env.vectorized_observation_shape()[0], (8,))
    resources = AgentResources(network, None, input, print)
    configs = (RuleBasedAgentConfig(policy="Piers"), MCTSAgentConfig(rollouts=1))
    assert [type(a) for a in build_agents(configs, game, resources)] == [PiersAgent, MCTSAgent]
    assert isinstance(build_agent(0, AlphaZeroAgentConfig(), game, resources), AlphaZeroAgent)
    assert isinstance(build_agent(1, HumanAgentConfig(), game, resources), HumanAgent)


def test_build_agent_requires_resources() -> None:
    with pytest.raises(ConfigError, match="network"):
        build_agent(0, AlphaZeroAgentConfig(), GameConfig(), AgentResources())
    with pytest.raises(ConfigError, match="Human"):
        build_agent(0, HumanAgentConfig(), GameConfig(), AgentResources())
