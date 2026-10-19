"""AlphaZero with rollouts split across Ray worker processes (root parallelization)."""

import copy
from collections import defaultdict
from typing import Any, override

import numpy.typing as npt
import ray
import torch
from torch import nn

from hanabi_search.agents.alphazero.agent import AlphaZeroAgent
from hanabi_search.agents.mcts.parallel import ensure_ray
from hanabi_search.config import AlphaZeroAgentConfig, GameConfig
from hanabi_search.env import HanabiMove, HanabiState, ObservationDict
from hanabi_search.networks import AlphaZeroNetwork
from hanabi_search.training.transition import TransitionSink

# Per module: parameter and buffer arrays by name.
type ModuleTensors = dict[str, dict[str, npt.NDArray[Any]]]


def extract_tensors(model: nn.Module) -> tuple[nn.Module, list[ModuleTensors]]:
    """Splits a model into a weightless skeleton and NumPy weights.

    Ray stores NumPy arrays zero-copy, so workers can share one copy of the weights.

    Returns:
        The skeleton (in eval mode) and per-module parameter/buffer arrays.
    """
    tensors: list[ModuleTensors] = [
        {
            "params": {
                name: param.detach().cpu().numpy().copy()
                for name, param in module.named_parameters(recurse=False)
            },
            "buffers": {
                name: buf.detach().cpu().numpy().copy()
                for name, buf in module.named_buffers(recurse=False)
            },
        }
        for _, module in model.named_modules()
    ]
    skeleton = copy.deepcopy(model)
    for _, module in skeleton.named_modules():
        names = [name for name, _ in module.named_parameters(recurse=False)]
        names += [name for name, _ in module.named_buffers(recurse=False)]
        for name in names:
            setattr(module, name, None)
    skeleton.train(False)
    return skeleton, tensors


def replace_tensors(skeleton: nn.Module, tensors: list[ModuleTensors]) -> None:
    """Restores weights removed by `extract_tensors` into `skeleton`."""
    for (_, module), arrays in zip(skeleton.named_modules(), tensors, strict=True):
        for name, array in arrays["params"].items():
            module.register_parameter(name, nn.Parameter(torch.as_tensor(array)))
        for name, array in arrays["buffers"].items():
            module.register_buffer(name, torch.as_tensor(array))


class _AlphaZeroWorker:
    """Runs an independent AlphaZero search from a serialized state on CPU.

    Args:
        player_id: Seat of the searching player.
        game: The game being played.
        config: Search hyperparameters; `rollouts` is this worker's share.
        weights: Output of `extract_tensors` for the network.
    """

    def __init__(
        self,
        player_id: int,
        game: GameConfig,
        config: AlphaZeroAgentConfig,
        weights: tuple[nn.Module, list[ModuleTensors]],
    ) -> None:
        skeleton, tensors = weights
        replace_tensors(skeleton, tensors)
        assert isinstance(skeleton, AlphaZeroNetwork)
        skeleton.eval()
        self.agent = AlphaZeroAgent(player_id, game, config, skeleton, torch.device("cpu"))

    def search(self, state_json: str) -> dict[str, int]:
        """Searches from the state and returns visit counts of the root's children."""
        agent = self.agent
        state = HanabiState.from_json(state_json)
        agent.environment.state = state
        observation = agent.environment.observation_dict(state.observation(state.cur_player()))
        agent.reset(state)
        for _ in range(agent.rollouts):
            agent.run_rollout(observation)
        return {
            child.initial_move().to_json(): agent.visits[child]
            for child in agent.children.get(agent.root_node, ())
        }


AlphaZeroWorker = ray.remote(num_cpus=4)(_AlphaZeroWorker)


class ParallelAlphaZeroAgent(AlphaZeroAgent):
    """`AlphaZeroAgent` whose rollouts run on `config.num_workers` Ray workers.

    Workers get a snapshot of the network weights when the agent is created.

    Args:
        player_id: Absolute seat of this agent.
        game: The game being played.
        config: Search hyperparameters; `rollouts` is split evenly between workers.
        network: Policy/value network.
        device: Device of the network in this process.
        transition_sink: See `AlphaZeroAgent`.
    """

    def __init__(
        self,
        player_id: int,
        game: GameConfig,
        config: AlphaZeroAgentConfig,
        network: AlphaZeroNetwork,
        device: torch.device,
        transition_sink: TransitionSink | None = None,
    ) -> None:
        super().__init__(player_id, game, config, network, device, transition_sink)
        ensure_ray()
        worker_config = config.model_copy(
            update={"rollouts": max(1, config.rollouts // config.num_workers), "num_workers": 1}
        )
        weights = ray.put(extract_tensors(network))
        # Ray does not type method calls on actor handles.
        self.workers: list[Any] = [
            AlphaZeroWorker.remote(player_id, game, worker_config, weights)
            for _ in range(config.num_workers)
        ]

    @override
    def act(self, observation: ObservationDict, state: HanabiState) -> HanabiMove | None:
        if observation["current_player_offset"] != 0:
            return None
        self.reset(state)
        state_json = state.to_json()
        results: list[dict[str, int]] = ray.get([w.search.remote(state_json) for w in self.workers])

        visits: defaultdict[str, int] = defaultdict(int)
        for stats in results:
            for move_json, count in stats.items():
                visits[move_json] += count

        self.root_node.focused_state = self._root_state.copy()
        if not visits:
            return self.choose_move(self.root_node)
        best_json = max(visits, key=lambda move_json: visits[move_json])
        if self.collects_data:
            game = self.environment.game
            by_uid = {
                game.get_move_uid(HanabiMove.from_json(m)): float(c) for m, c in visits.items()
            }
            self._pending.append(self._decision(observation, by_uid))
        return HanabiMove.from_json(best_json)
