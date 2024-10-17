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
from agents.mcts.mcts_agent import PMCTS_Agent
from agents.αzero.αzero_agent import AlphaZero_Agent, AlphaZeroP_Agent
from agents.human_agent import HumanAgent
from agents.αzero.αzero_network import AlphaZeroNetwork, create_loss_function, train_network
import tensorflow as tf

AGENT_CLASSES = {
    'MCTS_Agent': MCTS_Agent,
    'PMCTS_Agent': PMCTS_Agent,
    'AlphaZero_Agent': AlphaZero_Agent,
    'AlphaZeroP_Agent': AlphaZeroP_Agent,
    'VanDenBerghAgent': VanDenBerghAgent,
    'FlawedAgent': FlawedAgent,
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

        self.num_actions = self.environment.num_moves()
        self.network = AlphaZeroNetwork(self.num_actions)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        self.loss_fn = create_loss_function()
        self.training_data = []

    def run(self):
        """Run episodes."""
        scores = []
        agents = []

        for i in range(len(self.agent_classes)):
            self.agent_config.update({
                'player_id': i, 
                'num_actions': self.num_actions, 
                'network': self.network, 
                'optimizer': self.optimizer, 
                'loss_fn': self.loss_fn
            })

            agents.append(self.agent_classes[i](self.agent_config))

        errors = 0

        with tqdm(total=self.flags['num_episodes'], desc="Running Episodes", unit="episode") as pbar:
            for episode in range(self.flags['num_episodes']):
                done = False
                observations = self.environment.reset()

                while not done:
                    for agent_id, agent in enumerate(agents):
                        observation = observations['player_observations'][agent_id]

                        if isinstance(agent, MCTS_Agent) or isinstance(agent, PMCTS_Agent) or isinstance(agent, AlphaZero_Agent):
                            action = agent.act(observation, self.environment.state)
                        else:
                            action = agent.act(observation)

                        if observation['current_player'] == agent_id:
                            assert action is not None
                            current_player_action = action
                        else:
                            assert action is None

                    observations, reward, done, unused_info = self.environment.step(current_player_action)
                    
                final_score = sum(v for k,v in observation["fireworks"].items()) if observation["life_tokens"] > 0 else 0
                scores.append(final_score)
                z = final_score / 25

                for agent in agents:
                    if not isinstance(agent, AlphaZero_Agent):
                        continue
                    for i in range(len(agent.training_data)):
                        state_vector, policy_targets, _ = agent.training_data[i]
                        agent.training_data[i] = (state_vector, policy_targets, z)

                    self.training_data.extend(agent.training_data)
                    agent.training_data.clear()
                
                if self.training_data:
                    loss = train_network(self.network, self.training_data, self.optimizer, self.loss_fn)
                    self.training_data.clear()
                    print(f"Final Score: {final_score}/25\nLoss: {loss}")

                # Calculate running average score
                avg_score = sum(scores) / (episode + 1)
                # Update the progress bar
                pbar.set_postfix({'Avg Score': '{0:.2f}'.format(avg_score)})
                pbar.update(1)

        avg_score = np.mean(scores)
        std_dev = np.std(scores)
        std_error = std_dev / np.sqrt(len(scores))

        print(f"\nScores: {scores}")
        print(f"Average Score: {avg_score}")
        print(f"Standard Deviation: {std_dev}")
        print(f"Standard Error: {std_error}")
        print(f"Errors: {errors}\n")

        return avg_score, std_error


def run_simulation_and_plot():
    """Runs multiple simulations and saves the results as a plot."""
    max_depth_values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    max_rollout_values = [100, 1000, 5000, 10000, 50000, 100000]
    max_simulation_steps = [i for i in range(0, 11)]
    avg_scores = []
    std_errs = []

    for max_roll_num in max_rollout_values:
        flags = {
            'players': 2,
            'num_episodes': 10,
            'agent': 'PMCTS_Agent',
            'agents': 'PMCTS_Agent',
            'mcts_types': '00',
            'max_rollout_num': max_roll_num,
            'max_simulation_steps': 0,
            #'max_depth': 3
        }
        
        flags['agent_classes'] = [flags['agent']] * flags['players']
        
        runner = Runner(flags)
        avg_score, std_error = runner.run()
        avg_scores.append(avg_score)
        std_errs.append(std_error)

    # Plotting
    plt.figure(figsize=(12, 8))
    sns.set_theme(style="whitegrid")
    palette = sns.color_palette("deep", 10)

    # Plotting with error bars
    plt.errorbar(max_rollout_values, avg_scores, yerr=std_errs, fmt='o', color=palette[0], ecolor='lightgray', elinewidth=2, capsize=5, label="Std Error")
    plt.plot(max_rollout_values, avg_scores, marker='o', color=palette[1], markersize=8, linestyle='-', linewidth=2, label="Avg Score")
    plt.grid(True, which='both', linestyle='--', linewidth=0.7, alpha=0.7)
    plt.xlabel('Max Rollout Number', fontsize=16, fontweight='bold', labelpad=10)
    plt.ylabel('Average Score', fontsize=16, fontweight='bold', labelpad=10)
    plt.title('Effect of Max Rollout Number on Average Score', fontsize=18, fontweight='bold', pad=15)
    plt.xticks(max_rollout_values, fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend(loc='upper left', fontsize=12)

    # Save the plot to a file
    plt.tight_layout()
    plt.savefig('max_roll_num.png')
    print("Plot saved as 'max_roll_num.png'.")
    

if __name__ == "__main__":
    start_time = time.time()

    #run_simulation_and_plot()

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

    print(f"Total Time: {time.time() - start_time:.2f} seconds")

# Far more efficient, prepare presentation (rules + mcts explanation + very hard Johan), next week start alpha zero