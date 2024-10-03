import math
import time
from collections import defaultdict
import ray
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
        # effect of rules, effect of van_der_bergh, plot of rolouts vs score, representation for alpha-0, alphaGO-0, read papers, 

        self.max_time_limit = config.get("max_time_limit", 100000)
        self.max_rollout_num = config.get("max_rollout_num", 50)
        self.max_simulation_steps = config.get("max_simulation_steps", 3)
        self.max_depth = config.get("max_depth", 6)
        self.exploration_weight = config.get("exploration_weight", 2.5)

        self.rules = config.get("rules", [
            Ruleset.tell_most_information_factory(True),
            Ruleset.tell_anyone_useful_card,
            Ruleset.tell_dispensable_factory(8),
            Ruleset.complete_tell_useful,
            Ruleset.complete_tell_dispensable,
            Ruleset.complete_tell_unplayable,
            Ruleset.play_probably_safe_factory(0.7, False),
            Ruleset.play_probably_safe_late_factory(0.4, 5),
            Ruleset.discard_most_confident,
        ])

        self.agents = [VanDenBerghAgent(config) for _ in range(config["players"])]
        self.determine_type = mcts_env.DetermineType.RESTORE
        self.score_type = mcts_env.ScoreType.SCORE
        self.mcts_type = config["mcts_types"][config["player_id"]]

        self.environment = mcts_env.make(
            "Hanabi-Full",
            num_players=config["players"],
            mcts_player=config["player_id"],
            determine_type=self.determine_type,
            score_type=self.score_type,
        )
        self.max_information_tokens = config.get("information_tokens", 8)

    def act(self, observation, state):
        if observation["current_player_offset"] != 0:
            return None

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

        done = node.is_terminal()
        reward = environment.reward()
        steps = 0

        while not done and steps < self.max_simulation_steps:
            player_observations = observations["player_observations"]
            current_agent = player_observations[0]["current_player"]
            observation = player_observations[current_agent]
            agent = agents[observation["current_player"]]
            current_player_action = agent.act(observation)

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


class MCTS_Agent_Conc(MCTS_Agent):
    def __init__(self, config):
        super().__init__(config)
        if not ray.is_initialized():
            ray.init()

        num_workers = 8
        worker_max_time_limit = self.max_time_limit // num_workers
        worker_max_rollout_num = self.max_rollout_num // num_workers

        self.workers = [
            MCTS_Worker.remote(config, worker_max_time_limit, worker_max_rollout_num)
            for _ in range(num_workers)
        ]

    def act(self, observation, state):
        if observation["current_player_offset"] != 0:
            return None

        self.reset(state)

        # Serialize the state to JSON
        state_json = state.to_json()

        # Serialize the observation
        observation_copy = observation.copy()
        current_player = observation_copy["current_player"]
        observation_copy["pyhanabi"] = current_player

        # Use the workers to perform MCTS search
        worker_futures = [
            worker.perform_mcts_search.remote(observation_copy, state_json)
            for worker in self.workers
        ]

        # Get results from workers
        worker_results = ray.get(worker_futures)

        # Merge results
        merge_results(self, worker_results)

        self.root_node.focused_state = self.root_state.copy()
        best_node = self.mcts_choose(self.root_node)

        return best_node.initial_move()


@ray.remote(num_cpus=1)
class MCTS_Worker:
    def __init__(self, config, max_time_limit, max_rollout_num):
        self.agent = MCTS_Agent(config)
        self.max_time_limit = max_time_limit
        self.max_rollout_num = max_rollout_num

    def perform_mcts_search(self, observation, state_json):
        # Reconstruct the state and observation
        state = pyhanabi.HanabiState.from_json(state_json)
        current_player = observation['pyhanabi']
        observation['pyhanabi'] = state.observation(current_player)

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

        #print(f"Worker finished {rollout}/{self.max_rollout_num} rollouts in {elapsed_time:.2f}/{self.max_time_limit} ms")

        # Serialize the nodes before returning
        serialized_children = self.serialize_children(self.agent.children)
        serialized_Q = self.serialize_Q_or_N(self.agent.Q)
        serialized_N = self.serialize_Q_or_N(self.agent.N)

        return (serialized_children, serialized_Q, serialized_N)

    def serialize_children(self, children):
        serialized = {}
        for node, child_nodes in children.items():
            node_json = node.to_json()
            child_nodes_json = [child_node.to_json() for child_node in child_nodes]
            serialized[node_json] = child_nodes_json
        return serialized

    def serialize_Q_or_N(self, Q_or_N):
        serialized = {}
        for node, value in Q_or_N.items():
            node_json = node.to_json()
            serialized[node_json] = value
        return serialized


def merge_results(mcts_agent, worker_results):
    key_to_node = {}
    all_node_jsons = set()

    # First pass: Collect all unique node_json strings
    for serialized_children, serialized_Q, serialized_N in worker_results:
        all_node_jsons.update(serialized_children.keys())
        for child_nodes_json in serialized_children.values():
            all_node_jsons.update(child_nodes_json)
        all_node_jsons.update(serialized_Q.keys())
        all_node_jsons.update(serialized_N.keys())

    # Deserialize all unique nodes
    for node_json in all_node_jsons:
        node = MCTS_Node.from_json(node_json)
        node.rules = mcts_agent.rules
        key_to_node[node_json] = node

    # Second pass: Merge the results using the deserialized nodes
    for serialized_children, serialized_Q, serialized_N in worker_results:
        # Merge children
        for node_json, child_nodes_json in serialized_children.items():
            node = key_to_node[node_json]
            if node not in mcts_agent.children:
                mcts_agent.children[node] = set()
            for child_json in child_nodes_json:
                child_node = key_to_node[child_json]
                mcts_agent.children[node].add(child_node)
        # Merge Q values
        for node_json, q_value in serialized_Q.items():
            node = key_to_node[node_json]
            mcts_agent.Q[node] += q_value
        # Merge N values
        for node_json, n_value in serialized_N.items():
            node = key_to_node[node_json]
            mcts_agent.N[node] += n_value
