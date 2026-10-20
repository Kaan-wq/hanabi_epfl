"""MCTS with rollouts split across Ray worker processes (root parallelization)."""

from collections import defaultdict
from typing import Any, override

import ray

from hanabi_search.agents.mcts.agent import MCTSAgent
from hanabi_search.config import GameConfig, MCTSAgentConfig
from hanabi_search.env import (
    LIB_ENV_VAR,
    LIBRARY_PATH,
    HanabiMove,
    HanabiState,
    ObservationDict,
)
from hanabi_search.training.transition import TransitionSink

# Per root move (serialized), summed reward and visit count.
type RootStats = dict[str, tuple[float, int]]


# Large directories workers never read; excluded when Ray ships the project to them.
_RAY_UPLOAD_EXCLUDES = ["saved_models/", "docs/", "notebooks/", "runs/", "build/", ".cache/"]


def ensure_ray(num_cpus: int | None = None) -> None:
    """Starts a local Ray instance if none is running.

    Workers are told where the engine library is, since Ray does not ship the
    git-ignored build directory to them.

    Args:
        num_cpus: CPUs Ray may schedule; `None` detects them.
    """
    if ray.is_initialized():
        return
    ray.init(
        include_dashboard=False,
        num_cpus=num_cpus,
        runtime_env={
            "env_vars": {LIB_ENV_VAR: str(LIBRARY_PATH)},
            "excludes": _RAY_UPLOAD_EXCLUDES,
        },
    )


class _MCTSWorker:
    """Runs an independent search from a serialized state.

    Args:
        player_id: Seat of the searching player.
        game: The game being played.
        config: Search hyperparameters; `rollouts` is this worker's share.
    """

    def __init__(self, player_id: int, game: GameConfig, config: MCTSAgentConfig) -> None:
        self.agent = MCTSAgent(player_id, game, config)

    def search(self, state_json: str) -> RootStats:
        """Searches from the state and returns statistics of the root's children."""
        agent = self.agent
        state = HanabiState.from_json(state_json)
        agent.environment.state = state
        observation = agent.environment.observation_dict(state.observation(state.cur_player()))
        agent.reset(state)
        for _ in range(agent.rollouts):
            agent.run_rollout(observation)
        return {
            child.initial_move().to_json(): (agent.total_reward[child], agent.visits[child])
            for child in agent.children.get(agent.root_node, ())
        }


MCTSWorker = ray.remote(num_cpus=1)(_MCTSWorker)


class ParallelMCTSAgent(MCTSAgent):
    """`MCTSAgent` whose rollouts run on `config.num_workers` Ray workers.

    Each worker searches its own tree; root statistics are summed before choosing.

    Args:
        player_id: Absolute seat of this agent.
        game: The game being played.
        config: Search hyperparameters; `rollouts` is split evenly between workers.
        transition_sink: See `MCTSAgent`.
    """

    def __init__(
        self,
        player_id: int,
        game: GameConfig,
        config: MCTSAgentConfig,
        transition_sink: TransitionSink | None = None,
    ) -> None:
        super().__init__(player_id, game, config, transition_sink)
        ensure_ray()
        worker_config = config.model_copy(
            update={"rollouts": max(1, config.rollouts // config.num_workers), "num_workers": 1}
        )
        # Ray does not type method calls on actor handles.
        self.workers: list[Any] = [
            MCTSWorker.remote(player_id, game, worker_config) for _ in range(config.num_workers)
        ]

    @override
    def act(self, observation: ObservationDict, state: HanabiState) -> HanabiMove | None:
        if observation["current_player_offset"] != 0:
            return None
        self.reset(state)
        state_json = state.to_json()
        results: list[RootStats] = ray.get([w.search.remote(state_json) for w in self.workers])

        totals: defaultdict[str, list[float]] = defaultdict(lambda: [0.0, 0.0])
        for stats in results:
            for move_json, (reward, visits) in stats.items():
                totals[move_json][0] += reward
                totals[move_json][1] += visits

        self.root_node.focused_state = self._root_state.copy()
        if not totals:
            return self.choose_move(self.root_node)
        best_json = max(
            totals,
            key=lambda m: float("-inf") if totals[m][1] <= 1 else totals[m][0] / totals[m][1],
        )
        if self.collects_data:
            game = self.environment.game
            visit_counts = {
                game.get_move_uid(HanabiMove.from_json(m)): count
                for m, (_, count) in totals.items()
            }
            self._pending.append(self._decision(observation, visit_counts))
        return HanabiMove.from_json(best_json)
