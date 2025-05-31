# Solving Hanabi with MCTS and AlphaZero

**Author:** Kaan Uçar  
**Supervisor:** Giulio Romanelli

## Overview

This project explores solving the cooperative card game Hanabi using Monte Carlo Tree Search (MCTS) and AlphaZero algorithms. Built upon Google's Hanabi Learning Environment, this research investigates the effectiveness of different AI approaches for mastering this challenging game of imperfect information and communication.

## Project Report

For detailed methodology, experiments, results, and analysis, please refer to the complete project report: **[Hanabi.pdf](./Hanabi.pdf)**

## About the Environment

> **Note:** This is not an officially supported Google product.

The `hanabi_learning_environment` serves as a research platform for Hanabi experiments. The file `rl_env.py` provides an RL environment using an API similar to OpenAI Gym. A lower level game interface is provided in `pyhanabi.py` for non-RL methods like Monte Carlo tree search.

## Getting Started

### Installation

Install the learning environment:
```bash
sudo apt-get install g++            # if you don't already have a CXX compiler
sudo apt-get install python-pip     # if you don't already have pip
pip install .                       # or pip install git+repo_url to install directly from github
pip install requirements.txt        # required libraries to run the code
python setup.py build_ext           # required for cython
```

### Running Examples

Basic examples:
```bash
pip install numpy                   # game_example.py uses numpy
python examples/rl_env_example.py   # Runs RL episodes
python examples/game_example.py     # Plays a game using the lower level interface
```

### MCTS Experiments

Run MCTS agents with different configurations:
```bash
# 2-player PMCTS experiments
python rl_env_example.py --players 2 --num_episodes 100 --agent PMCTS_Agent --agents PMCTS_Agent --mcts_types 00

# 5-player MCTS experiments  
python rl_env_example.py --players 5 --num_episodes 10 --agent MCTS_Agent --agents MCTS_Agent --mcts_types 00000
```

## Research Focus

This project specifically investigates:
- Monte Carlo Tree Search implementations for incomplete information games
- AlphaZero adaptations for cooperative games

---

*Based on Google's Hanabi Learning Environment research platform*
