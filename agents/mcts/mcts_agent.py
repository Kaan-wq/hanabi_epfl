import math
import time
from collections import defaultdict
from queue import Queue
import threading
import pyhanabi
from agents.mcts import mcts_env
from agents.mcts.mcts_node import MCTS_Node
from agents.rule_based.rule_based_agents import (
    FlawedAgent,
    IGGIAgent,
    InnerAgent,
    LegalRandomAgent,
    MuteAgent,
    OuterAgent,
    PiersAgent,
    VanDenBerghAgent,
)
from agents.rule_based.ruleset import Ruleset
from rl_env import Agent

AGENT_CLASSES = {
    "VanDenBerghAgent": VanDenBerghAgent,
    "FlawedAgent": FlawedAgent,
    "OuterAgent": OuterAgent,
    "InnerAgent": InnerAgent,
    "PiersAgent": PiersAgent,
    "IGGIAgent": IGGIAgent,
    "LegalRandomAgent": LegalRandomAgent,
    "MuteAgent": MuteAgent,
}


class MCTS_Agent(Agent):
    """Agent based on Redeterminizing Information Set Monte Carlo Tree Search"""

    def __init__(self, config):
        """Initialize the agent."""

        self.children = dict()
        self.Q = defaultdict(int)
        self.N = defaultdict(int)
        self.root_node = None
        self.root_state = None
        self.player_id = config["player_id"]

        self.max_time_limit = 1000
        self.max_rollout_num = 50
        self.max_simulation_steps = 3
        self.max_depth = 100
        self.exploration_weight = 2.5

        self.agents = [VanDenBerghAgent(config) for _ in range(config["players"])]
        self.determine_type = mcts_env.DetermineType.RESTORE
        self.score_type = mcts_env.ScoreType.SCORE

        self.playable_now_convention = False
        self.playable_now_convention_sim = False
        self.rules = [
            Ruleset.tell_most_information_factory(True),
            Ruleset.tell_anyone_useful_card,
            Ruleset.tell_dispensable_factory(8),
            Ruleset.complete_tell_useful,
            Ruleset.complete_tell_dispensable,
            Ruleset.complete_tell_unplayable,
            Ruleset.play_probably_safe_factory(0.7, False),
            Ruleset.play_probably_safe_late_factory(0.4, 5),
            Ruleset.discard_most_confident,
        ]
        self.mcts_type = config["mcts_types"][config["player_id"]]
        self._edit_mcts_config(self.mcts_type, config)

        self.environment = mcts_env.make(
            "Hanabi-Full",
            num_players=config["players"],
            mcts_player=config["player_id"],
            determine_type=self.determine_type,
            score_type=self.score_type,
        )
        self.max_information_tokens = config.get("information_tokens", 8)

    def _edit_mcts_config(self, mcts_type, config):
        """Interpret the mcts_type character"""
        if mcts_type == "0":  # default
            pass
        else:
            print(f"'mcts_config_error {mcts_type}',")

    def act(self, observation, state):
        if observation["current_player_offset"] != 0:
            return None

        if self.playable_now_convention:
            action = Ruleset.playable_now_convention(observation)
            if action is not None:
                return action

        self.reset(state)

        rollout = 0
        start_time = time.time()
        elapsed_time = 0

        while rollout < self.max_rollout_num and elapsed_time < self.max_time_limit:
            self.environment.state = self.root_state.copy()
            self.environment.replace_hand(self.player_id)

            self.root_node.focused_state = self.environment.state
            self.environment.reset(observation)

            path, reward = self.mcts_search(self.root_node, observation)
            rollout += 1
            elapsed_time = (time.time() - start_time) * 1000

        self.root_node.focused_state = self.root_state.copy()
        best_node = self.mcts_choose(self.root_node)

        return best_node.initial_move()

    def mcts_search(self, node, observation):
        path = self.mcts_select(node)
        leaf = path[-1]
        depth = 0

        environment = self.environment
        state = environment.state
        max_depth = self.max_depth

        for move in leaf.moves:
            legal_moves = state.legal_moves()
            if move not in legal_moves:
                reward = self.environment.reward()
                self.mcts_backpropagate(path, reward)
                return path, reward

            observations, reward, done, _ = environment.step(move)
            state = environment.state
            current_player = state.cur_player()
            observation = observations["player_observations"][current_player]

            depth += 1
            if depth > max_depth:
                break

        leaf.focused_state = state

        if depth < max_depth:
            self.mcts_expand(leaf, observation)

        reward = self.mcts_simulate(leaf)
        self.mcts_backpropagate(path, reward)

        return path, reward

    def mcts_choose(self, node):
        """'Choose the best successor of node. (Choose a move from the root node)"""

        if node.is_terminal():
            raise RuntimeError(f"choose called on terminal node {node}")
        if node not in self.children:
            return node.find_random_child()

        return max(
            self.children[node],
            key=lambda n: float("-inf") if self.N[n] <= 1 else self.Q[n] / self.N[n],
        )

    def mcts_select(self, node):
        """Find an unexplored descendent of `node`"""
        path = []
        while True:
            path.append(node)
            if node not in self.children or not self.children[node]:
                return path

            unexplored = [child for child in self.children[node] if self.N[child] == 0]
            if unexplored:
                n = unexplored[0]
                path.append(n)
                return path

            node = self.uct_select(node)

    def mcts_expand(self, node, observation):
        """Expand the `node` with all children"""

        if node in self.children:
            return

        moves = node.find_children(observation)
        if moves and isinstance(moves[0], dict):
            build_move = self.environment._build_move
            moves = {build_move(action) for action in moves}
        else:
            moves = set(moves)

        self.children[node] = set(
            MCTS_Node(node.moves + (move,), self.rules) for move in moves
        )

    def mcts_simulate(self, node):
        """Run a simulation from the given node."""

        environment = self.environment
        environment.state = node.focused_state
        observations = environment._make_observation_all_players()

        agents = self.agents
        playable_now_convention_sim = self.playable_now_convention_sim

        done = node.is_terminal()
        reward = environment.reward()
        steps = 0

        while not done and steps < self.max_simulation_steps:
            player_observations = observations["player_observations"]
            current_agent = player_observations[0]["current_player"]
            observation = player_observations[current_agent]
            agent = agents[observation["current_player"]]
            current_player_action = agent.act(observation)

            if playable_now_convention_sim:
                playable_now_action = Ruleset.playable_now_convention(observation)
                if playable_now_action is not None:
                    current_player_action = playable_now_action

            observations, reward, done, _ = environment.step(current_player_action)
            steps += 1

            if done:
                break

        return reward

    def mcts_backpropagate(self, path, reward):
        """Backpropagate the result of a simulation through the tree"""
        for node in path:
            self.N[node] += 1
            self.Q[node] += reward

    def uct_select(self, node):
        "Select a child of node, balancing exploration & exploitation"

        log_N_node = math.log(self.N[node])
        exploration_weight = self.exploration_weight

        def uct(child):
            Q_child = self.Q[child]
            N_child = self.N[child]
            return (Q_child / N_child) + exploration_weight * math.sqrt(
                log_N_node / N_child
            )

        return max(self.children[node], key=uct)

    def reset(self, state):
        """Reset the agent with a new state"""

        self.player_id = state.cur_player()
        self.root_state = state.copy()
        self.root_node = MCTS_Node((), self.rules)

        self.children.clear()
        self.Q.clear()
        self.N.clear()

        self.N[self.root_node] = 0
        self.Q[self.root_node] = 0

    def _get_tree_string(self):
        """Get the tree as a string."""
        tree_string = ""
        for node, children in self.children.items():
            tree_string += f"[{node}: {self.N[node]}, {self.Q[node]}] "
        return tree_string


class MCTS_Agent_Conc(MCTS_Agent, Agent):
    def __init__(self, config):
        super().__init__(config)
        self.workers = [
            MCTS_Worker(config, self.max_time_limit, self.max_rollout_num), 
            MCTS_Worker(config, self.max_time_limit, self.max_rollout_num)
        ]

        for worker in self.workers:
            worker.start()

    def act(self, observation, state):
        if observation["current_player_offset"] != 0:
            return None
        
        print(f"\n\n\n==================== Observation of Agent {observation['current_player']} ====================\n\n{observation['pyhanabi']}\n\n\n")
        print(f"\n\n\n==================== New Observation of Agent {observation['current_player']} ====================\n\n{state.observation(observation['current_player'])}\n\n\n")
        
        state_json = state.to_json()
        state = pyhanabi.HanabiState.from_json(state_json)

        self.reset(state)

        # Use the worker to perform MCTS search
        for worker in self.workers:
            worker.task_queue.put((observation, state))

        for worker in self.workers:
            worker.task_queue.join()

        # Merge results
        worker_results = []
        for worker in self.workers:
            while not worker.result_queue.empty():
                worker_results.append(worker.result_queue.get())
        merge_results(self, worker_results)


        self.root_node.focused_state = self.root_state.copy()
        best_node = self.mcts_choose(self.root_node)

        return best_node.initial_move()

    def __del__(self):
        if hasattr(self, 'workers'):
            for worker in self.workers:
                worker.task_queue.put(None)
            for worker in self.workers:
                worker.join()



class MCTS_Worker(threading.Thread):
    def __init__(self, config, max_time_limit, max_rollout_num):
        super().__init__()
        self.agent = MCTS_Agent(config)
        self.max_time_limit = max_time_limit
        self.max_rollout_num = max_rollout_num
        self.task_queue = Queue()
        self.result_queue = Queue()

    def run(self):
        while True:
            task = self.task_queue.get()
            if task is None:
                break
            observation, state = task
            self.perform_mcts_search(observation, state)
            self.task_queue.task_done()

    def perform_mcts_search(self, observation, state):
        self.agent.reset(state)
        rollout = 0
        start_time = time.time()
        elapsed_time = 0

        while rollout < self.max_rollout_num and elapsed_time < self.max_time_limit:
            self.agent.environment.state = self.agent.root_state.copy()
            self.agent.environment.replace_hand(self.agent.player_id)

            self.agent.root_node.focused_state = self.agent.environment.state
            self.agent.environment.reset(observation)

            path, reward = self.agent.mcts_search(self.agent.root_node, observation)
            rollout += 1
            elapsed_time = (time.time() - start_time) * 1000

        self.result_queue.put((self.agent.children, self.agent.Q, self.agent.N))


def merge_results(mcts_agent, worker_results):
    for children, Q, N in worker_results:
        mcts_agent.children.update(children)
        for node, q_value in Q.items():
            mcts_agent.Q[node] += q_value
        for node, n_value in N.items():
            mcts_agent.N[node] += n_value
