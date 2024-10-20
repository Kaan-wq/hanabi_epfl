"""A simple episode runner using the RL environment."""

from __future__ import print_function
import sys
import getopt
import time
from tqdm import tqdm
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
from agents.alphazero.alphazero_agent import AlphaZero_Agent, AlphaZeroP_Agent
from agents.human_agent import HumanAgent
from agents.alphazero.alphazero_network import AlphaZeroNetwork, prepare_data
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

        self.training_data = []
        self.num_actions = self.environment.num_moves()
        self.obs_shape = self.environment.vectorized_observation_shape()[0]

        self.network = AlphaZeroNetwork(self.num_actions, self.obs_shape)
        self.optimizer = tf.keras.optimizers.AdamW(learning_rate=1e-4, weight_decay=1e-4)
        self.network.compile(
            optimizer=self.optimizer,
            loss={
                'policy_logits': tf.keras.losses.CategoricalCrossentropy(from_logits=True),
                'value': tf.keras.losses.MeanSquaredError()
            },
        )

    def run(self):
        """Run episodes."""
        scores = []
        agents = []

        for i in range(len(self.agent_classes)):
            self.agent_config.update({
                'player_id': i, 
                'num_actions': self.num_actions, 
                'obs_shape': self.obs_shape,
                'network': self.network
            })

            agents.append(self.agent_classes[i](self.agent_config))

        errors = 0

        with tqdm(total=self.flags['num_episodes'], desc="Running Episodes", unit="episode", ncols=200) as pbar:
            pbar.set_postfix({'Avg Score': 'N/A', 'Score': 'N/A', 'Avg Loss': 'N/A'})
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
                z = 2 * (final_score / 25) - 1

                avg_score = sum(scores) / (episode + 1)

                for agent in agents:
                    if not isinstance(agent, AlphaZero_Agent):
                        continue
                    for i in range(len(agent.training_data)):
                        state_vector, policy_targets, _ = agent.training_data[i]
                        agent.training_data[i] = (tf.cast(tf.expand_dims(state_vector, axis=-1), tf.float32), policy_targets, z)

                    self.training_data.extend(agent.training_data)
                    agent.training_data.clear()
                
                if self.training_data:
                    batch_size = 20
                    steps = len(self.training_data) // batch_size

                    history = self.network.fit(
                        x=prepare_data(self.training_data, batch_size),
                        steps_per_epoch=steps,
                        epochs=1,
                        verbose=1,
                    )

                    loss = history.history['loss'][0]
                    pbar.set_postfix({'Avg Score': '{0:.2f}'.format(avg_score), 'Score': final_score, 'Avg Loss': '{0:.4f}'.format(loss)})
                    pbar.update(1)
                    self.training_data.clear()
                    self.network.save('saved_models/resnet18-1600-100.keras')
                else:
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
    

if __name__ == "__main__":
    start_time = time.time()

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

    # Save the model
    runner.network.save('saved_models/alphazero_resnet18.keras')

    print(f"Total Time: {time.time() - start_time:.2f} seconds")
