"""Experiment configuration schema. Every run is defined by one YAML file in `configs/`."""

from pathlib import Path
from typing import Annotated, Literal, Self

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from hanabi_search.env import AgentObservationType, DeterminizationType, RolloutReward
from hanabi_search.exceptions import ConfigError


class _Config(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class GameConfig(_Config):
    """Game variant. Defaults are the full game ("Hanabi-Full")."""

    players: int = Field(default=2, ge=2, le=5)
    colors: int = Field(default=5, ge=1, le=5)
    ranks: int = Field(default=5, ge=1, le=5)
    hand_size: int | None = Field(
        default=None, ge=1, le=8, description="None: 5 cards, 4 for 4+ players (engine max: 8)"
    )
    max_information_tokens: int = Field(default=8, ge=1)
    max_life_tokens: int = Field(default=3, ge=1)
    observation_type: AgentObservationType = AgentObservationType.CARD_KNOWLEDGE
    random_start_player: bool = True

    @property
    def max_score(self) -> int:
        """Score of a perfect game."""
        return self.colors * self.ranks

    def to_params(self, seed: int | None = None) -> dict[str, int | bool | str]:
        """Returns engine parameters; `seed=None` lets the engine pick a random seed."""
        params: dict[str, int | bool | str] = {
            "colors": self.colors,
            "ranks": self.ranks,
            "players": self.players,
            "max_information_tokens": self.max_information_tokens,
            "max_life_tokens": self.max_life_tokens,
            "observation_type": int(self.observation_type),
            "random_start_player": self.random_start_player,
        }
        if self.hand_size is not None:
            params["hand_size"] = self.hand_size
        if seed is not None:
            params["seed"] = seed
        return params


type RuleBasedPolicy = Literal[
    "VanDenBergh", "Flawed", "Outer", "Inner", "Piers", "IGGI", "LegalRandom", "Mute"
]


class RuleBasedAgentConfig(_Config):
    """A rule-based agent from `agents.rule_based`."""

    kind: Literal["rule_based"] = "rule_based"
    policy: RuleBasedPolicy = "VanDenBergh"


class HumanAgentConfig(_Config):
    """A human playing through the terminal."""

    kind: Literal["human"] = "human"


class MCTSAgentConfig(_Config):
    """Re-determinizing information-set MCTS with rule-guided expansion."""

    kind: Literal["mcts"] = "mcts"
    rollouts: int = Field(default=50, ge=1, description="Rollouts per move, over all workers")
    max_simulation_steps: int = Field(default=3, ge=0)
    max_depth: int = Field(default=60, ge=1)
    exploration_weight: float = Field(default=2.5, ge=0)
    use_rules: bool = Field(default=True, description="Expand only rule moves, not all legal")
    determinization: DeterminizationType = DeterminizationType.RESTORE
    rollout_reward: RolloutReward = RolloutReward.SCORE
    num_workers: int = Field(default=1, ge=1, description=">1 runs rollouts on Ray workers")
    collect_data: bool = Field(default=False, description="Record (observation, visits) data")


class AlphaZeroAgentConfig(_Config):
    """MCTS guided by a policy/value network instead of rollouts."""

    kind: Literal["alphazero"] = "alphazero"
    rollouts: int = Field(default=100, ge=1, description="Rollouts per move, over all workers")
    max_depth: int = Field(default=60, ge=1)
    exploration_weight: float = Field(default=2.5, ge=0)
    use_rules: bool = Field(default=True, description="Expand only rule moves, not all legal")
    determinization: DeterminizationType = DeterminizationType.RESTORE
    num_workers: int = Field(default=1, ge=1, description=">1 runs rollouts on Ray workers")


type AgentConfig = Annotated[
    RuleBasedAgentConfig | HumanAgentConfig | MCTSAgentConfig | AlphaZeroAgentConfig,
    Field(discriminator="kind"),
]


class NetworkConfig(_Config):
    """AlphaZero policy/value network."""

    hidden_sizes: tuple[int, ...] = (512, 256, 256)
    checkpoint: Path | None = Field(default=None, description="state_dict to start from")


class ReplayBufferConfig(_Config):
    """Prioritized replay buffer."""

    capacity: int = Field(default=10_000, ge=1)
    alpha: float = Field(default=0.6, ge=0, description="Priority exponent")
    beta: float = Field(default=0.4, ge=0, le=1, description="Initial importance-sampling exponent")
    save_transitions: bool = Field(default=True, description="Append transitions to a JSONL file")


class SelfPlayConfig(_Config):
    """Exploration of AlphaZero agents in training games; evaluation never uses it."""

    dirichlet_alpha: float = Field(default=0.3, gt=0, description="Root noise concentration")
    dirichlet_fraction: float = Field(
        default=0.25, ge=0, le=1, description="Weight of the noise in the root priors"
    )
    temperature_moves: int = Field(
        default=10,
        ge=0,
        description="Own moves per game drawn in proportion to root visits (then most visited)",
    )


class TrainingConfig(_Config):
    """AlphaZero training."""

    num_episodes: int = Field(default=100, ge=1)
    learning_rate: float = Field(default=1e-4, gt=0)
    weight_decay: float = Field(default=1e-4, ge=0)
    batch_size: int = Field(default=128, ge=1)
    value_loss_weight: float = Field(default=10.0, ge=0)
    train_every_step: bool = Field(default=True, description="Also train after every move")
    replay_buffer: ReplayBufferConfig = ReplayBufferConfig()
    self_play: SelfPlayConfig = SelfPlayConfig()


class EvaluationConfig(_Config):
    """Evaluation on fixed, held-out seeds."""

    num_games: int = Field(default=1000, ge=1)
    seed: int = Field(default=1_000_000, description="Game i uses seed + i")


class ExperimentConfig(_Config):
    """A complete experiment: game, seats, and optionally training."""

    name: str = Field(pattern=r"^[A-Za-z0-9_.-]+$")
    seed: int = 0
    device: Literal["cpu", "mps"] = "cpu"
    game: GameConfig = GameConfig()
    agents: tuple[AgentConfig, ...] = Field(
        min_length=1, description="One per seat, or a single entry used for every seat"
    )
    network: NetworkConfig = NetworkConfig()
    training: TrainingConfig | None = None
    evaluation: EvaluationConfig = EvaluationConfig()

    @model_validator(mode="after")
    def _check_seats(self) -> Self:
        if len(self.agents) not in (1, self.game.players):
            raise ValueError(
                f"agents must have 1 or {self.game.players} entries, got {len(self.agents)}"
            )
        return self

    def seat_agents(self) -> tuple[AgentConfig, ...]:
        """Returns one agent config per seat."""
        if len(self.agents) == 1:
            return self.agents * self.game.players
        return self.agents

    @classmethod
    def from_yaml(cls, path: Path) -> Self:
        """Loads and validates an experiment file.

        Raises:
            ConfigError: If the file cannot be read or is invalid.
        """
        try:
            data = yaml.safe_load(path.read_text())
            return cls.model_validate(data)
        except (OSError, yaml.YAMLError, ValidationError) as error:
            raise ConfigError(f"Invalid config {path}: {error}") from error

    def to_yaml(self) -> str:
        """Returns the fully resolved config as YAML."""
        return yaml.safe_dump(self.model_dump(mode="json"), sort_keys=False)
