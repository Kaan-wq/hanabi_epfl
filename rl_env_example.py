"""A simple episode runner using the RL environment."""

from __future__ import print_function
import sys
import getopt
import time
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from rl_env import make
from agents.rule_based.rule_based_agents import VanDenBerghAgent
from agents.rule_based.rule_based_agents import OuterAgent
from agents.rule_based.rule_based_agents import InnerAgent
from agents.rule_based.rule_based_agents import PiersAgent
from agents.rule_based.rule_based_agents import IGGIAgent
from agents.rule_based.rule_based_agents import LegalRandomAgent
from agents.rule_based.rule_based_agents import FlawedAgent
from agents.rule_based.rule_based_agents import MuteAgent
from agents.mcts.mcts_agent import MCTS_Agent
from agents.mcts.mcts_agent import MCTS_Agent_Conc
from agents.human_agent import HumanAgent

AGENT_CLASSES = {
    'VanDenBerghAgent': VanDenBerghAgent,
    'FlawedAgent': FlawedAgent,
    'MCTS_Agent': MCTS_Agent,
    'MCTS_Agent_Conc': MCTS_Agent_Conc,
    'OuterAgent': OuterAgent,
    'InnerAgent': InnerAgent,
    'PiersAgent': PiersAgent,
    'IGGIAgent': IGGIAgent,
    'LegalRandomAgent': LegalRandomAgent,
    'MuteAgent': MuteAgent,
    'HumanAgent': HumanAgent
}

class Runner(object):
    """Runner class."""

    def __init__(self, flags):
        """Initialize runner."""
        self.flags = flags
        self.agent_config = {'players': flags['players'], 'player_id': 0, 'mcts_types': flags['mcts_types']}
        self.environment = make('Hanabi-Full', num_players=flags['players'])
        self.agent_classes = [AGENT_CLASSES[agent_class] for agent_class in flags['agent_classes']]

    def run(self):
        """Run episodes."""
        game_stats = []
        player_stats = []
        agents = []

        for i in range(len(self.agent_classes)):
            self.agent_config.update({'player_id': i})  # Update player_id for each agent
            agents.append(self.agent_classes[i](self.agent_config))
            player_stats.append([])

        errors = 0

        with tqdm(total=self.flags['num_episodes'], desc="Running Episodes", unit="episode") as pbar:
            for episode in range(self.flags['num_episodes']):
                done = False
                observations = self.environment.reset()

                while not done:
                    for agent_id, agent in enumerate(agents):
                        observation = observations['player_observations'][agent_id]

                        if isinstance(agent, MCTS_Agent) or isinstance(agent, MCTS_Agent_Conc):
                            action = agent.act(observation, self.environment.state)
                        else:
                            action = agent.act(observation)

                        if observation['current_player'] == agent_id:
                            assert action is not None
                            current_player_action = action
                        else:
                            assert action is None

                    observations, reward, done, unused_info = self.environment.step(current_player_action)

                game_stats.append(self.environment.game_stats())

                for i in range(len(self.agent_classes)):
                    player_stats[i].append(self.environment.player_stats(i))

                # Calculate running average score
                avg_score = sum([g['score'] for g in game_stats]) / (episode + 1)
                # Update the progress bar
                pbar.set_postfix({'Avg Score': '{0:.2f}'.format(avg_score)})
                pbar.update(1)

        scores = [g['score'] for g in game_stats]
        avg_score = np.mean(scores)
        std_dev = np.std(scores)

        print(f"\nScores: {scores}")
        print(f"Stats Keys: {list(game_stats[0].keys())}")
        print(f"Game Stats: {self.simplify_stats(game_stats)}")
        print(f"Player Stats: {[self.simplify_stats(p) for p in player_stats]}")
        print(f"Average Score: {avg_score}")
        print(f"Standard Deviation: {std_dev}")
        print(f"Errors: {errors}")

        return avg_score, std_dev

    def simplify_stats(self, stats):
        """Extract just the numbers from the stats."""
        return [list(g.values()) for g in stats]

    def print_state(self):
        self.environment.print_state()


def run_simulation_and_plot():
    """Runs multiple simulations and saves the results as a plot."""
    max_depth_values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    avg_scores = []
    std_devs = []

    for max_depth in max_depth_values:
        flags = {
            'players': 2,
            'num_episodes': 2,
            'agent': 'MCTS_Agent_Conc',
            'agents': 'MCTS_Agent_Conc',
            'mcts_types': '00',
            'max_depth': max_depth,
        }
        
        flags['agent_classes'] = [flags['agent']] * flags['players']
        
        runner = Runner(flags)
        avg_score, std_dev = runner.run()
        avg_scores.append(avg_score)
        std_devs.append(std_dev)

    # Plotting
    plt.figure(figsize=(12, 8))
    sns.set_theme(style="whitegrid")
    palette = sns.color_palette("deep", 10)

    # Plotting with error bars
    plt.errorbar(max_depth_values, avg_scores, yerr=std_devs, fmt='o', color=palette[0], ecolor='lightgray', elinewidth=2, capsize=5, label="Avg Score with Std Dev")
    plt.plot(max_depth_values, avg_scores, marker='o', color=palette[1], markersize=8, linestyle='-', linewidth=2, label="Avg Score")
    plt.grid(True, which='both', linestyle='--', linewidth=0.7, alpha=0.7)
    plt.xlabel('Max Depth', fontsize=16, fontweight='bold', labelpad=10)
    plt.ylabel('Average Score', fontsize=16, fontweight='bold', labelpad=10)
    plt.title('Effect of Max Depth on Average Score', fontsize=18, fontweight='bold', pad=15)
    plt.xticks(max_depth_values, fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend(loc='upper left', fontsize=12)

    # Save the plot to a file
    plt.tight_layout()
    plt.savefig('max_depth.png')
    print("Plot saved as 'max_depth.png'.")
    

if __name__ == "__main__":
    start_time = time.time()

    run_simulation_and_plot()

    '''
    flags = {
        'players': 3,
        'num_episodes': 1,
        'agent': 'VanDenBerghAgent',
        'agents': 'VanDenBerghAgent',
        'mcts_types': '000'
    }
    options, arguments = getopt.getopt(sys.argv[1:], '',
                                       ['players=',
                                        'num_episodes=',
                                        'agent=',
                                        'agents=',
                                        'mcts_types='])
    if arguments:
        sys.exit('usage: rl_env_example.py [options]\n'
                 '--players       number of players in the game.\n'
                 '--num_episodes  number of game episodes to run.\n'
                 '--agent         class name of single agent. Supported: {}\n'
                 '--agents        class name of agents to play with.\n'
                 '--mcts_types    000 each character is the type of the mcts agent in that position.\n'
                 ''.format(' or '.join(AGENT_CLASSES.keys())))

    for flag, value in options:
        flag = flag[2:]  # Strip leading --
        flags[flag] = type(flags[flag])(value)

    flags['agent_classes'] = [flags['agent']] + [flags['agents'] for _ in range(1, flags["players"])]

    if len(flags['agent_classes']) != flags['players']:
        sys.exit(f'Number of agent classes: {len(flags["agent_classes"])} not same as number of players: {flags["players"]}')

    runner = Runner(flags)
    runner.run()
    '''
    print(f"Total Time: {time.time() - start_time:.2f} seconds")
