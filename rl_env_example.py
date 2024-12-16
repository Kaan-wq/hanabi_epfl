import getopt
import sys
import time

import numpy as np
import torch
from tqdm import tqdm

from agents.alphazero.alphazero_agent import AlphaZero_Agent, AlphaZeroP_Agent
from agents.alphazero.alphazero_buffer import (configure_quality_buffer,
                                               configure_replay_buffer)
from agents.alphazero.alphazero_network import (collect_alphazero_data,
                                                collect_mcts_data,
                                                initialize_training_components,
                                                requires_mcts_data,
                                                requires_training,
                                                train_network)
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

        # Initialize data collection components if required
        self.mcts_data = requires_mcts_data(self.agent_classes, record_data=False)
        if self.mcts_data:
            self.replay_buffer = configure_replay_buffer(
                capacity=10000,
                storage_mode="hybrid",
                file_path="experiments/policies/mcts.jsonl",
            )
            self.num_actions = self.environment.num_moves()

        # Initialize training components if required
        self.requires_training = requires_training(self.agent_classes)
        if self.requires_training:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.replay_buffer = configure_quality_buffer(
                capacity=10000,
                storage_mode="hybrid",
                file_path="experiments/policies/alphazero.jsonl",
                mcts_data_path="experiments/policies/mcts.jsonl",
            )
            (self.network, self.optimizer, self.criterion_value, self.num_actions) = (
                initialize_training_components(self.environment, self.device, lr=1e-4)
            )
            # ,from_pretrained="saved_models/policy_model_200.pth"

    def run(self):
        """Run episodes."""
        scores = []
        policy_losses = []
        value_losses = []
        policy_contributions = []
        value_contributions = []
        agents = []

        for i, agent_class in enumerate(self.agent_classes):
            self.agent_config.update({"player_id": i})

            if self.requires_training and issubclass(
                agent_class, (AlphaZero_Agent, AlphaZeroP_Agent)
            ):
                self.agent_config["network"] = self.network
                self.agent_config["num_actions"] = self.num_actions
            elif (
                self.mcts_data
                and issubclass(agent_class, (MCTS_Agent, PMCTS_Agent))
                and not issubclass(agent_class, (AlphaZero_Agent, AlphaZeroP_Agent))
            ):
                self.agent_config["num_actions"] = self.num_actions
                self.agent_config["collect_data"] = True
                self.agent_config.pop("network", None)
            else:
                self.agent_config.pop("network", None)
                self.agent_config.pop("num_actions", None)
                self.agent_config.pop("collect_data", None)

            agents.append(agent_class(self.agent_config))

        with tqdm(
            total=self.flags["num_episodes"],
            desc="Running Episodes",
            unit="episode",
            ncols=190,
        ) as pbar:
            pbar.set_postfix(
                {
                    "Avg Score": "N/A",
                    "Score": "N/A",
                    "Policy Loss": "N/A",
                    "Value Loss": "N/A",
                }
            )
            for episode in range(self.flags["num_episodes"]):
                done = False
                observations = self.environment.reset()

                while not done:
                    for agent_id, agent in enumerate(agents):
                        observation = observations["player_observations"][agent_id]

                        if isinstance(
                            agent,
                            (
                                MCTS_Agent,
                                PMCTS_Agent,
                                AlphaZero_Agent,
                                AlphaZeroP_Agent,
                            ),
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
                avg_score = np.mean(scores)
                latest_policy_loss = "N/A"
                latest_value_loss = "N/A"
                latest_policy_contrib = "N/A"
                latest_value_contrib = "N/A"

                if self.mcts_data:
                    collect_mcts_data(agents, self.replay_buffer, final_score)

                if self.requires_training:
                    collect_alphazero_data(agents, self.replay_buffer, final_score)
                    loss_dict = train_network(
                        self.replay_buffer,
                        self.network,
                        self.optimizer,
                        self.device,
                        batch_size=128,
                        value_loss_weight=10.0,
                    )

                    if loss_dict is not None:
                        latest_policy_loss = loss_dict["policy_loss"]
                        latest_value_loss = loss_dict["value_loss"]
                        latest_policy_contrib = loss_dict[
                            "effective_policy_contribution"
                        ]
                        latest_value_contrib = loss_dict["effective_value_contribution"]

                        policy_losses.append(latest_policy_loss)
                        value_losses.append(latest_value_loss)
                        policy_contributions.append(latest_policy_contrib)
                        value_contributions.append(latest_value_contrib)

                    # Save the model
                    torch.save(
                        self.network.state_dict(), "saved_models/alphazero_model_100.pth"
                    )

                pbar.set_postfix(
                    {
                        "Avg Score": f"{avg_score:.2f}",
                        "Score": final_score,
                        "Policy Loss": (
                            f"{latest_policy_loss:.4f} ({latest_policy_contrib:.1%})"
                            if latest_policy_loss != "N/A"
                            else "N/A"
                        ),
                        "Value Loss": (
                            f"{latest_value_loss:.4f} ({latest_value_contrib:.1%})"
                            if latest_value_loss != "N/A"
                            else "N/A"
                        ),
                    }
                )
                pbar.update(1)

        avg_score = np.mean(scores)
        std_dev = np.std(scores)
        std_error = std_dev / np.sqrt(len(scores))

        # Print final statistics with contributions
        print("\nLoss History (value : contribution):")
        print(
            "Policy Losses:",
            [
                f"{loss:.4f} ({contrib:.1%})"
                for loss, contrib in zip(policy_losses, policy_contributions)
            ],
        )
        print(
            "Value Losses:",
            [
                f"{loss:.4f} ({contrib:.1%})"
                for loss, contrib in zip(value_losses, value_contributions)
            ],
        )

        print(f"\nScores: {scores}")
        print(f"Average Score: {avg_score}")
        print(f"Standard Deviation: {std_dev}")
        print(f"Standard Error: {std_error}")


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
