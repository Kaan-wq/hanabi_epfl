This is not an officially supported Google product.

hanabi_learning_environment is a research platform for Hanabi experiments. The file rl_env.py provides an RL environment using an API similar to OpenAI Gym. A lower level game interface is provided in pyhanabi.py for non-RL methods like Monte Carlo tree search.

### Getting started

Install the learning environment:

```
sudo apt-get install g++            # if you don't already have a CXX compiler
sudo apt-get install python-pip     # if you don't already have pip
pip install .                       # or pip install git+repo_url to install directly from github
pip install requirements.txt        # required libraries to run the code
python setup.py build_ext           # required for cython
```

Run the examples:

```
pip install numpy                   # game_example.py uses numpy
python examples/rl_env_example.py   # Runs RL episodes
python examples/game_example.py     # Plays a game using the lower level interface
```

Alternatively:

```
python rl_env_example.py --players 2 --num_episodes 100 --agent PMCTS_Agent --agents PMCTS_Agent --mcts_types 00
python rl_env_example.py --players 5 --num_episodes 10 --agent MCTS_Agent --agents MCTS_Agent --mcts_types 00000
```
