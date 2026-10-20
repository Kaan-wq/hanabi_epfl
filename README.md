# Solving Hanabi with MCTS and AlphaZero

**Author:** Kaan Uçar
**Supervisor:** Giulio Romanelli

Agents for the cooperative card game Hanabi: re-determinizing information-set
Monte Carlo Tree Search (MCTS), an AlphaZero-style variant guided by a
policy/value network, and rule-based baselines from the literature. They run on
our fork of DeepMind's [Hanabi Learning Environment](https://github.com/deepmind/hanabi-learning-environment)
(see [engine/UPSTREAM.md](engine/UPSTREAM.md) and [engine/CHANGES.md](engine/CHANGES.md)).

The methodology, experiments and results are in the project report:
**[docs/report.pdf](docs/report.pdf)**.

## Setup

Requirements: a C++23 compiler (clang ≥ 18 or GCC ≥ 16, for `std::mdspan`), CMake ≥ 3.24
and [uv](https://docs.astral.sh/uv/).
On macOS, `xcode-select --install` provides the compiler. nlohmann/json is used
from the system if installed, otherwise CMake downloads it.

```bash
# 1. Build the engine (Python loads build/release/engine/libpyhanabi)
cmake -S . -B build/release -DCMAKE_BUILD_TYPE=Release
cmake --build build/release
ctest --test-dir build/release --output-on-failure

# 2. Create the Python environment (Python 3.14)
uv sync
```

Rebuild the engine (`cmake --build build`) after changing anything in `engine/`.
To use a library built elsewhere, set `HANABI_SEARCH_LIB=/path/to/libpyhanabi.dylib`.

## Usage

Every run is defined by one YAML file in [`configs/`](configs):

```bash
# Play games and report mean score ± standard error
uv run python scripts/play.py --config configs/vandenbergh_selfplay_2p.yaml --num-games 100
uv run python scripts/play.py --config configs/mcts_vs_vandenbergh_3p.yaml --num-games 10

# Play yourself against a bot in the terminal
uv run python scripts/play.py --config configs/human_vs_vandenbergh_2p.yaml --num-games 1

# Train AlphaZero by self-play; writes runs/<experiment>/<timestamp>_<seed>/
uv run python scripts/train.py --config configs/alphazero_2p.yaml

# Evaluate on the held-out evaluation seeds
uv run python scripts/evaluate.py --config configs/alphazero_2p.yaml \
    --checkpoint runs/alphazero_2p/<run>/model.pt --num-games 1000
```

Agent kinds: `rule_based` (policies `VanDenBergh`, `Piers`, `IGGI`, `Outer`,
`Inner`, `Flawed`, `Mute`, `LegalRandom`), `mcts`, `alphazero` and `human`. See [`src/hanabi_search/config.py`](src/hanabi_search/config.py)
for every option and its default.

## Development

```bash
uv run pytest                 # all tests; add -m "not slow" to skip script tests
uv run ruff check . && uv run ruff format --check .
uv run mypy                   # strict
./build/release/engine/benchmarks/bench_engine
```

Conventions, the repository layout and the rules for changing the engine are in
[CLAUDE.md](CLAUDE.md).

### Troubleshooting

- **`EngineLoadError: Engine library libpyhanabi not found`**: build the engine
  (step 1 above).
- **macOS linker error `unknown architecture` in a `.tbd` file**: the default SDK
  is newer than the installed linker. Update the Command Line Tools
  (`softwareupdate --list`), then reconfigure from a fresh `build/release/`.

## Results

Experiment logs from the report are in [docs/results/](docs/results), analysis
notebooks in [notebooks/](notebooks), and trained AlphaZero weights in
[saved_models/](saved_models).

## License

Apache-2.0 (see [LICENSE](LICENSE)). The engine is derived from DeepMind's Hanabi
Learning Environment, also Apache-2.0.
