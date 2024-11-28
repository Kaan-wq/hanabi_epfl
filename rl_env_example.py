import getopt
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from agents.alphazero.alphazero_agent import AlphaZero_Agent, AlphaZeroP_Agent
from agents.alphazero.alphazero_buffer import ReplayBuffer
from agents.alphazero.alphazero_network import SimpleNetwork, prepare_data
from agents.human_agent import HumanAgent
from agents.mcts.mcts_agent import MCTS_Agent, PMCTS_Agent
from agents.rule_based.rule_based_agents import (FlawedAgent, IGGIAgent,
                                                 InnerAgent, LegalRandomAgent,
                                                 MuteAgent, OuterAgent,
                                                 PiersAgent, VanDenBerghAgent)
from rl_env import make

AGENT_CLASSES = {
    "MCTS_Agent": MCTS_Agent,
    "PMCTS_Agent": PMCTS_Agent,
    "AlphaZero_Agent": AlphaZero_Agent,
    "AlphaZeroP_Agent": AlphaZeroP_Agent,
    "VanDenBerghAgent": VanDenBerghAgent,
    "FlawedAgent": FlawedAgent,
    "OuterAgent": OuterAgent,
    "InnerAgent": InnerAgent,
    "PiersAgent": PiersAgent,
    "IGGIAgent": IGGIAgent,
    "LegalRandomAgent": LegalRandomAgent,
    "MuteAgent": MuteAgent,
    "HumanAgent": HumanAgent,
}


class Runner(object):
    """Runner class."""

    def __init__(self, flags):
        """Initialize runner."""
        self.flags = flags
        self.agent_config = {
            "players": flags["players"],
            "player_id": 0,
            "mcts_types": flags["mcts_types"],
        }
        self.environment = make("Hanabi-Full", num_players=flags["players"])

        self.agent_classes = [
            AGENT_CLASSES[agent_class] for agent_class in flags["agent_classes"]
        ]

        self.training_data = []
        self.replay_buffer = ReplayBuffer(capacity=10000)
        self.num_actions = self.environment.num_moves()
        self.obs_shape = self.environment.vectorized_observation_shape()[0]

        # Set device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize PyTorch network
        self.network = SimpleNetwork(self.num_actions, self.obs_shape)
        self.network.to(self.device)

        # Load model weights if available
        self.network.load_state_dict(torch.load('saved_models/policy.pth', map_location=self.device))

        # Optimizer and loss functions
        self.optimizer = optim.AdamW(
            self.network.parameters(), lr=1e-4, weight_decay=1e-4
        )
        self.criterion_value = nn.MSELoss()

    def run(self):
        """Run episodes."""
        scores = []
        losses = []
        agents = []
        names = ["agent_one", "agent_two", "agent_three", "agent_four", "agent_five"]

        for i in range(len(self.agent_classes)):
            self.agent_config.update(
                {
                    "player_id": i,
                    "num_actions": self.num_actions,
                    "obs_shape": self.obs_shape,
                    "network": self.network,
                    "agent_name": names[i],
                }
            )

            agents.append(self.agent_classes[i](self.agent_config))

        errors = 0

        with tqdm(
            total=self.flags["num_episodes"],
            desc="Running Episodes",
            unit="episode",
            ncols=190,
        ) as pbar:
            pbar.set_postfix({"Avg Score": "N/A", "Score": "N/A", "Avg Loss": "N/A", "Loss": "N/A"})
            for episode in range(self.flags["num_episodes"]):
                done = False
                observations = self.environment.reset()

                while not done:
                    for agent_id, agent in enumerate(agents):
                        observation = observations["player_observations"][agent_id]

                        if isinstance(
                            agent, (MCTS_Agent, PMCTS_Agent, AlphaZero_Agent)
                        ):
                            action = agent.act(observation, self.environment.state)
                        else:
                            action = agent.act(observation)

                        if observation["current_player"] == agent_id:
                            assert action is not None
                            current_player_action = action
                        else:
                            assert action is None

                    observations, reward, done, unused_info = self.environment.step(
                        current_player_action
                    )

                final_score = (
                    sum(v for k, v in observation["fireworks"].items())
                    if observation["life_tokens"] > 0
                    else 0
                )
                scores.append(final_score)

                z = 2 * (final_score / 25) - 1  # Value target

                avg_score = sum(scores) / (episode + 1)

                # Collect training data from AlphaZero agents
                for agent in agents:
                    if not isinstance(agent, (AlphaZero_Agent, AlphaZeroP_Agent)):
                        continue
                    for i in range(len(agent.training_data)):
                        state_vector, policy_targets, _ = agent.training_data[i]
                        state_vector = torch.tensor(state_vector, dtype=torch.float32)
                        policy_targets = torch.tensor(policy_targets, dtype=torch.float32)
                        value_target = torch.tensor(z, dtype=torch.float32)
                        agent.training_data[i] = (state_vector, policy_targets, value_target)

                    self.replay_buffer.add(agent.training_data)
                    agent.training_data.clear()

                    
                batch_size = 128
                latest_loss = "N/A"
                if len(self.replay_buffer) >= batch_size:
                    batch_data = self.replay_buffer.sample(batch_size)
                    dataloader = prepare_data(batch_data, batch_size)

                    total_loss = 0.0
                    steps = 0

                    self.network.train()  # Set model to training mode

                    for states, policy_targets, value_targets in dataloader:
                        states = states.to(self.device)
                        policy_targets = policy_targets.to(self.device)
                        value_targets = value_targets.to(self.device)

                        self.optimizer.zero_grad()

                        #TODO: Add value head
                        policy_logits = self.network(states)

                        # Compute policy loss with soft labels
                        policy_log_probs = nn.functional.log_softmax(
                            policy_logits, dim=1
                        )
                        policy_loss = -torch.mean(
                            torch.sum(policy_targets * policy_log_probs, dim=1)
                        )

                        # Compute value loss
                        #value_loss = self.criterion_value(
                        #    value.squeeze(-1), value_targets
                        #)

                        # Total loss
                        loss = policy_loss #+ value_loss

                        # Backpropagation
                        loss.backward()
                        self.optimizer.step()

                        total_loss += loss.item()
                        latest_loss = loss.item()
                        losses.append(latest_loss)
                        steps += 1

                    avg_loss = total_loss / steps if steps > 0 else 0.0

                    pbar.set_postfix(
                        {
                            "Avg Score": "{0:.2f}".format(avg_score),
                            "Score": final_score,
                            "Avg Loss": "{0:.2f}".format(avg_loss),
                            "Loss": "{0:.2f}".format(latest_loss),
                        }
                    )
                    pbar.update(1)
                    self.training_data.clear()

                    # Save the model
                    torch.save(self.network.state_dict(), 'saved_models/policy.pth')
                else:
                    pbar.update(1)

        avg_score = np.mean(scores)
        std_dev = np.std(scores)
        std_error = std_dev / np.sqrt(len(scores))

        print("\nLosses: ", [f"{loss:.2f}" for loss in losses])
        print("Average Loss: ", np.mean(losses) if losses else "N/A")

        print(f"\nScores: {scores}")
        print(f"Average Score: {avg_score}")
        print(f"Standard Deviation: {std_dev}")
        print(f"Standard Error: {std_error}")
        print(f"Errors: {errors}\n")


if __name__ == "__main__":
    start_time = time.time()

    flags = {
        "players": 3,
        "num_episodes": 1,
        "agent": "VanDenBerghAgent",
        "agents": "VanDenBerghAgent",
        "mcts_types": "000",
    }
    options, arguments = getopt.getopt(
        sys.argv[1:],
        "",
        ["players=", "num_episodes=", "agent=", "agents=", "mcts_types="],
    )
    if arguments:
        sys.exit(
            "usage: rl_env_example.py [options]\n"
            "--players       number of players in the game.\n"
            "--num_episodes  number of game episodes to run.\n"
            "--agent         class name of single agent. Supported: {}\n"
            "--agents        class name of agents to play with.\n"
            "--mcts_types    000 each character is the type of the mcts agent in that position.\n"
            "".format(" or ".join(AGENT_CLASSES.keys()))
        )

    for flag, value in options:
        flag = flag[2:]  # Strip leading --
        flags[flag] = type(flags[flag])(value)

    flags["agent_classes"] = [flags["agent"]] + [
        flags["agents"] for _ in range(1, flags["players"])
    ]

    if len(flags["agent_classes"]) != flags["players"]:
        sys.exit(
            f'Number of agent classes: {len(flags["agent_classes"])} not same as number of players: {flags["players"]}'
        )

    runner = Runner(flags)
    runner.run()

    print(f"Total Time: {time.time() - start_time:.2f} seconds")
