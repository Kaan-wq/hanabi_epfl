from rl_env import Agent
from collections import defaultdict
import math
import time
from agents.mcts import mcts_env
from agents.mcts.mcts_node import MCTS_Node
from agents.rule_based.ruleset import Ruleset
from agents.rule_based.rule_based_agents import VanDenBerghAgent
from agents.rule_based.rule_based_agents import OuterAgent
from agents.rule_based.rule_based_agents import InnerAgent
from agents.rule_based.rule_based_agents import PiersAgent
from agents.rule_based.rule_based_agents import IGGIAgent
from agents.rule_based.rule_based_agents import LegalRandomAgent
from agents.rule_based.rule_based_agents import FlawedAgent
from agents.rule_based.rule_based_agents import MuteAgent
import pyhanabi

AGENT_CLASSES = {'VanDenBerghAgent': VanDenBerghAgent,'FlawedAgent':FlawedAgent
                  , 'OuterAgent':OuterAgent, 'InnerAgent':InnerAgent, 'PiersAgent':PiersAgent, 'IGGIAgent':IGGIAgent
                  , 'LegalRandomAgent':LegalRandomAgent,'MuteAgent':MuteAgent}

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

    self.max_time_limit =  1000 # 1 second
    self.max_rollout_num = 50

    self.max_simulation_steps = 3
    self.agents = [VanDenBerghAgent(config) for _ in range(config["players"])]
    self.exploration_weight = 2.5
    self.max_depth = 100
    self.determine_type = mcts_env.DetermineType.RESTORE
    self.score_type = mcts_env.ScoreType.SCORE

    self.playable_now_convention = False
    self.playable_now_convention_sim = False
    self.rules =  [Ruleset.tell_most_information_factory(True)
        , Ruleset.tell_anyone_useful_card
        , Ruleset.tell_dispensable_factory(8)
        , Ruleset.complete_tell_useful
        , Ruleset.complete_tell_dispensable
        , Ruleset.complete_tell_unplayable
        , Ruleset.play_probably_safe_factory(0.7, False)
        , Ruleset.play_probably_safe_late_factory(0.4, 5)
        , Ruleset.discard_most_confident]
    self.mcts_type = config["mcts_types"][config['player_id']]
    self._edit_mcts_config(self.mcts_type, config)

    self.environment = mcts_env.make('Hanabi-Full', num_players=config["players"], mcts_player=config['player_id']
                                     ,determine_type = self.determine_type, score_type = self.score_type)
    self.max_information_tokens = config.get('information_tokens', 8)

  def _edit_mcts_config(self, mcts_type, config):
    """Interpret the mcts_type character"""
    if mcts_type == '0': #default
      pass
    elif mcts_type == '1': #regret
      self.score_type = mcts_env.ScoreType.REGRET
    elif mcts_type == '2': #c_regret
      self.score_type = mcts_env.ScoreType.REGRET
      self.playable_now_convention = True
      self.playable_now_convention_sim = True
    elif mcts_type == '3': #detnone
      self.determine_type = mcts_env.DetermineType.NONE
    elif mcts_type == '4': #detnone_rulesnone
      self.determine_type = mcts_env.DetermineType.NONE
      self.rules = None
    elif mcts_type == '5': #detnone_random_rulesnone
      self.determine_type = mcts_env.DetermineType.NONE
      self.agents = [LegalRandomAgent(config) for _ in range(config["players"])]
      self.rules = None
    elif mcts_type == '6': #detnone_regret_depth1
      self.determine_type = mcts_env.DetermineType.NONE
      self.score_type = mcts_env.ScoreType.REGRET
      self.max_depth = 1
    elif mcts_type == '7': #c
      self.playable_now_convention
      self.playable_now_convention_sim
    elif mcts_type == '8': #rulesnone
      self.rules = None
    elif mcts_type == '9': #detnone_regret
      self.determine_type = mcts_env.DetermineType.NONE
      self.score_type = mcts_env.ScoreType.REGRET
    elif mcts_type == 'a': #regret_rulesnone
      self.score_type = mcts_env.ScoreType.REGRET
      self.rules = None
    elif mcts_type == 'b': #detnone_regret_rulesnone
      self.determine_type = mcts_env.DetermineType.NONE
      self.score_type = mcts_env.ScoreType.REGRET
      self.rules = None
    elif mcts_type == 'c': #detnone_c
      self.determine_type = mcts_env.DetermineType.NONE
      self.playable_now_convention = True
      self.playable_now_convention_sim = True
    elif mcts_type == 'd': #mix_default
      self.determine_type = mcts_env.DetermineType.NONE
    elif mcts_type == 'e':  # mix_flawed
      self.determine_type = mcts_env.DetermineType.NONE
      self.agents[0] = FlawedAgent(config)
    elif mcts_type == 'f':  # mix_flawed_regret
      self.determine_type = mcts_env.DetermineType.NONE
      self.agents[0] = FlawedAgent(config)
      self.score_type = mcts_env.ScoreType.REGRET
    elif mcts_type == 'g':  # mix_flawed_depth1
      self.determine_type = mcts_env.DetermineType.NONE
      self.agents[0] = FlawedAgent(config)
      self.max_depth = 1
      self.max_simulation_steps = config["players"] - 1
    elif mcts_type == 'h':  # mix_flawed_regret_depth1
      self.determine_type = mcts_env.DetermineType.NONE
      self.agents[0] = FlawedAgent(config)
      self.score_type = mcts_env.ScoreType.REGRET
      self.max_depth = 1
      self.max_simulation_steps = config["players"] - 1
    elif mcts_type == 'i':  # mix_mute
      self.determine_type = mcts_env.DetermineType.NONE
      self.agents[0] = MuteAgent(config)
    elif mcts_type == 'j':  # mix_mute_regret
      self.determine_type = mcts_env.DetermineType.NONE
      self.agents[0] = MuteAgent(config)
      self.score_type = mcts_env.ScoreType.REGRET
    elif mcts_type == 'k':  # mix_mute_depth1
      self.determine_type = mcts_env.DetermineType.NONE
      self.agents[0] = MuteAgent(config)
      self.max_depth = 1
      self.max_simulation_steps = config["players"] - 1
    elif mcts_type == 'l':  # mix_mute_regret_depth1
      self.determine_type = mcts_env.DetermineType.NONE
      self.agents[0] = MuteAgent(config)
      self.score_type = mcts_env.ScoreType.REGRET
      self.max_depth = 1
      self.max_simulation_steps = config["players"] - 1
    elif mcts_type == 'm':  # mix_inner
      self.DetermineType = mcts_env.DetermineType.NONE
      self.agents[0] = InnerAgent(config)
    elif mcts_type == 'n':  # mix_inner_regret
      self.determine_type = mcts_env.DetermineType.NONE
      self.agents[0] = InnerAgent(config)
      self.score_type = mcts_env.ScoreType.REGRET
    elif mcts_type == 'o':  # mix_inner_depth1
      self.determine_type = mcts_env.DetermineType.NONE
      self.agents[0] = InnerAgent(config)
      self.max_depth = 1
      self.max_simulation_steps = config["players"] - 1
    elif mcts_type == 'p':  # mix_inner_regret_depth1
      self.determine_type = mcts_env.DetermineType.NONE
      self.agents[0] = InnerAgent(config)
      self.score_type = mcts_env.ScoreType.REGRET
      self.max_depth = 1
      self.max_simulation_steps = config["players"] - 1
    elif mcts_type == 'q':  # mix_random
      self.determine_type = mcts_env.DetermineType.NONE
      self.agents[0] = LegalRandomAgent(config)
    elif mcts_type == 'r':  # mix_random_regret
      self.determine_type = mcts_env.DetermineType.NONE
      self.agents[0] = LegalRandomAgent(config)
      self.score_type = mcts_env.ScoreType.REGRET
    elif mcts_type == 's':  # mix_random_depth1
      self.determine_type = mcts_env.DetermineType.NONE
      self.agents[0] = LegalRandomAgent(config)
      self.max_depth = 1
      self.max_simulation_steps = config["players"] - 1
    elif mcts_type == 't':  # mix_random_regret_depth1
      self.determine_type = mcts_env.DetermineType.NONE
      self.agents[0] = LegalRandomAgent(config)
      self.score_type = mcts_env.ScoreType.REGRET
      self.max_depth = 1
      self.max_simulation_steps = config["players"] - 1
    elif mcts_type == 'u':  # mix_vdb
      self.determine_type = mcts_env.DetermineType.NONE
      self.agents[0] = VanDenBerghAgent(config)
    elif mcts_type == 'v':  # mix_vdb_regret
      self.determine_type = mcts_env.DetermineType.NONE
      self.agents[0] = VanDenBerghAgent(config)
      self.score_type = mcts_env.ScoreType.REGRET
    elif mcts_type == 'w':  # mix_vdb_depth1
      self.determine_type = mcts_env.DetermineType.NONE
      self.agents[0] = VanDenBerghAgent(config)
      self.max_depth = 1
      self.max_simulation_steps = config["players"] - 1
    elif mcts_type == 'x':  # mix_vdb_regret_depth1
      self.determine_type = mcts_env.DetermineType.NONE
      self.agents[0] = VanDenBerghAgent(config)
      self.score_type = mcts_env.ScoreType.REGRET
      self.max_depth = 1
      self.max_simulation_steps = config["players"] - 1
    elif mcts_type == 'x': #fast test
      self.max_rollout_num = 10
      self.score_type = mcts_env.ScoreType.REGRET
      self.determine_type = mcts_env.DetermineType.NONE
      self.max_depth = 1
      self.agents[0] = FlawedAgent(config)
    elif mcts_type == 't': #test
      self.max_rollout_num = 25
      self.score_type = mcts_env.ScoreType.REGRET
    else:
      print(f"'mcts_config_error {mcts_type}',")

  def _get_mcts_config(self):
    return f"{{'max_time_limit':{self.max_time_limit}, 'max_rollout_num':{self.max_rollout_num}" \
           f",'agents':'{self.agents}', 'max_simulation_steps':{self.max_simulation_steps}, 'max_depth':{self.max_depth}" \
           f", 'determine_type':{self.determine_type}, 'score_type':{self.score_type}, 'exploration_weight':{self.exploration_weight}" \
           f",'playable_now_convention':{self.playable_now_convention},'playable_now_convention_sim':{self.playable_now_convention_sim}, 'rules':'{self.rules}'}}," \

  def __str__(self):
    return 'MCTS_Agent'+str(self.mcts_type)

  def __repr__(self):
    return str(self)

  def act(self, observation, state):
    if observation['current_player_offset'] != 0:
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

    for move in leaf.moves:
      if (not any(move == legal_move for legal_move in self.environment.state.legal_moves())):
        reward = self.environment.reward()
        self.mcts_backpropagate(path, reward)

        return path, reward
      
      observations, reward, done, unused_info = self.environment.step(move)
      observation = observations['player_observations'][self.environment.state.cur_player()]

      depth += 1
      if depth > self.max_depth:
        break

    leaf.focused_state = self.environment.state

    if not depth > self.max_depth:
      self.mcts_expand(leaf, observation)

    reward = self.mcts_simulate(leaf)
    self.mcts_backpropagate(path, reward)

    return path, reward


  def mcts_choose(self, node):
    ''''Choose the best successor of node. (Choose a move from the root node)'''

    if node.is_terminal():
      raise RuntimeError(f"choose called on terminal node {node}")
    if node not in self.children:
      print(f"mcts_choose called on an unexpanded node {node}. Choosing random child.")
      return node.find_random_child()

    return max(self.children[node], key=lambda n: float("-inf") if self.N[n] <= 1 else self.Q[n] / self.N[n])

  def mcts_select(self, node):
    '''Find an unexplored descendent of `node`'''

    path = []

    while True:
      path.append(node)
      if node not in self.children or not self.children[node]:
        return path
      
      unexplored = self.children[node] - self.children.keys()
      if unexplored:
        n = unexplored.pop()
        path.append(n)

        return path
      
      node = self.uct_select(node)

  def mcts_expand(self, node, observation):
    '''Expand the `node` with all children'''

    if node in self.children:
      return

    moves = node.find_children(observation)

    if len(moves) > 0 and isinstance(moves[0], dict):
      moves = set([self.environment._build_move(action) for action in moves])

    self.children[node] = [MCTS_Node(node.moves+(move,), self.rules) for move in moves]

  def mcts_simulate(self, node):
    '''Run a simulation from the given node'''

    self.environment.state = node.focused_state
    observations = self.environment._make_observation_all_players()

    done = node.is_terminal()
    reward = self.environment.reward()
    steps = 0

    while not done and steps < self.max_simulation_steps:
      for agent_id, agent in enumerate(self.agents):
        observation = observations['player_observations'][agent_id]

        if observation['current_player'] == agent_id:
          current_player_action = agent.act(observation)

          if self.playable_now_convention_sim:
            playable_now_action = Ruleset.playable_now_convention(observation)
            if playable_now_action is not None:
              current_player_action == playable_now_action

      observations, reward, done, unused_info = self.environment.step(current_player_action)
      steps += 1

      if not done:
        done = steps >= self.max_simulation_steps

    return reward

  def mcts_backpropagate(self, path, reward):
    '''Backpropagate the result of a simulation through the tree'''

    for node in reversed(path):
      self.N[node] += 1
      self.Q[node] += reward


  def uct_select(self, node):
    "Select a child of node, balancing exploration & exploitation"

    assert all(n in self.children for n in self.children[node])
    
    log_N_vertex = math.log(self.N[node])

    selected_child = max(
        self.children[node], 
        key=lambda n: self.Q[n] / self.N[n] + self.exploration_weight * math.sqrt(log_N_vertex / self.N[n])
    )
    
    return selected_child


  def reset(self, state):
    '''Reset the agent with a new state'''

    self.player_id = state.cur_player()
    self.root_state = state.copy()
    self.root_node = MCTS_Node((), self.rules)
    self.children = dict()
    self.Q = defaultdict(int)
    self.N = defaultdict(int)
    self.N[self.root_node] = 0
    self.Q[self.root_node] = 0

  def _get_tree_string(self):
    '''Get the tree as a string'''

    tree_string = ""
    for node, children in self.children.items():
      tree_string += f"[{node}: {self.N[node]}, {self.Q[node]}] "

    return tree_string