"""Builds agents from experiment configs."""

from collections.abc import Callable
from dataclasses import dataclass

import torch

from hanabi_search.agents.alphazero.agent import AlphaZeroAgent
from hanabi_search.agents.base import Agent
from hanabi_search.agents.human import HumanAgent
from hanabi_search.agents.mcts.agent import MCTSAgent
from hanabi_search.agents.rule_based import RULE_BASED_AGENTS
from hanabi_search.config import (
    AgentConfig,
    AlphaZeroAgentConfig,
    GameConfig,
    HumanAgentConfig,
    MCTSAgentConfig,
    RuleBasedAgentConfig,
    SelfPlayConfig,
)
from hanabi_search.exceptions import ConfigError
from hanabi_search.networks import AlphaZeroNetwork
from hanabi_search.training.transition import TransitionSink


@dataclass(frozen=True, slots=True)
class AgentResources:
    """Shared objects some agents need.

    Attributes:
        network: AlphaZero network, required for AlphaZero agents.
        device: Device of `network`.
        transition_sink: Where search agents send training data, if collecting.
        ask: Input function for human agents.
        show: Output function for human agents.
        self_play: Exploration for AlphaZero agents in training games; `None` when
            evaluating.
    """

    network: AlphaZeroNetwork | None = None
    device: torch.device = torch.device("cpu")  # noqa: RUF009 -- immutable value
    transition_sink: TransitionSink | None = None
    ask: Callable[[str], str] | None = None
    show: Callable[[str], None] | None = None
    self_play: SelfPlayConfig | None = None


def build_agent(
    seat: int, config: AgentConfig, game: GameConfig, resources: AgentResources
) -> Agent:
    """Creates the agent for one seat.

    Raises:
        ConfigError: If a required resource is missing.
    """
    match config:
        case RuleBasedAgentConfig(policy=policy):
            return RULE_BASED_AGENTS[policy](seat, game)
        case HumanAgentConfig():
            if resources.ask is None or resources.show is None:
                raise ConfigError("Human agents need ask/show functions")
            return HumanAgent(seat, game, resources.ask, resources.show)
        case MCTSAgentConfig():
            return MCTSAgent(seat, game, config, resources.transition_sink)
        case AlphaZeroAgentConfig():
            if resources.network is None:
                raise ConfigError("AlphaZero agents need a network")
            return AlphaZeroAgent(
                seat,
                game,
                config,
                resources.network,
                resources.device,
                resources.transition_sink,
                resources.self_play,
            )


def build_agents(
    seat_configs: tuple[AgentConfig, ...], game: GameConfig, resources: AgentResources
) -> list[Agent]:
    """Creates one agent per seat."""
    return [build_agent(seat, config, game, resources) for seat, config in enumerate(seat_configs)]
