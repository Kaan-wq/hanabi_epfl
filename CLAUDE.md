# CLAUDE.md

Guidance for Claude Code when working in this repository.
Keep the "Repository structure" section in sync with reality.

---

## Project overview

Reinforcement learning agents for the cooperative card game **Hanabi**, built on
our own fork of Google DeepMind's **Hanabi Learning Environment (HLE)**.

- `engine/` is our fork of HLE: a C++ game engine with a C API (`pyhanabi.h`),
  loaded from Python via **cffi**. We actively modify it for two goals:
  1. **Speed**: faster game simulation for RL throughput.
  2. **Richer state**: improved observations / state features returned to Python.
- Python side (package `hanabi_search`): MCTS and AlphaZero agents (plus rule-based
  baselines), environment wrappers, training and evaluation in PyTorch.
- Python 3.14, managed with **uv**. Training runs on Apple Silicon (MPS) or CPU.

Primary metric: **mean game score (0–25)** per player count, plus the
perfect-game rate (score 25). Always report the player count and number of
evaluation games alongside a score.

---

## Commands

```bash
# Engine (run from repo root). Python loads build/engine/libpyhanabi, so build
# this first and rebuild after every engine change.
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
cmake --build build
ctest --test-dir build --output-on-failure     # engine tests, incl. parity tests
./build/engine/benchmarks/bench_engine         # engine speed benchmark
cmake -S . -B build-debug -DCMAKE_BUILD_TYPE=Debug && cmake --build build-debug \
  && ctest --test-dir build-debug              # tests under ASan + UBSan
uv run clang-format -i engine/**/*.cc engine/**/*.h
SDKROOT=$(xcrun --show-sdk-path) uv run clang-tidy -p build \
  --extra-arg=-resource-dir=$(xcrun clang -print-resource-dir) <files>  # static analysis (macOS)

uv sync                          # install / update the environment
uv run pytest                    # Python tests (-m "not slow" skips Ray/script tests)
uv run ruff check --fix .        # lint + autofix
uv run ruff format .             # format Python
uv run mypy                      # strict type-check (src, tests, scripts)

uv run python scripts/play.py --config configs/<experiment>.yaml --num-games 10
uv run python scripts/train.py --config configs/<experiment>.yaml
uv run python scripts/evaluate.py --config configs/<experiment>.yaml --checkpoint <path> --num-games 1000
```

Run the relevant tests, linters and formatters before declaring a task done.
Never leave the repo in a state where any of the above fail.

---

## Repository structure

```
.
├── CLAUDE.md
├── README.md
├── pyproject.toml            # single source of Python config (deps, ruff, mypy, pytest)
├── uv.lock                   # committed, never edited by hand
├── CMakeLists.txt            # top-level build, adds engine/
├── .clang-format / .clang-tidy
├── .github/workflows/engine.yml  # CI: Linux GCC 15 build, engine tests, sanitizers, Python checks
├── .git-blame-ignore-revs    # formatting-only commits, skipped by git blame
├── .gitignore
├── engine/                   # our fork of DeepMind's HLE
│   ├── UPSTREAM.md           # upstream repo URL, forked commit, license notes
│   ├── CHANGES.md            # log of every change vs upstream, and why
│   ├── LICENSE               # upstream Apache-2.0 license, kept as-is
│   ├── hanabi_lib/           # C++ game engine (rules, state, observations)
│   ├── capi/                 # C API exposed to Python: pyhanabi.h, pyhanabi.cc,
│   │                         #   pyhanabi_observation.cc (Obs*), pyhanabi_internal.h
│   ├── tests/                # C++ unit tests + parity/ (golden games from upstream)
│   └── benchmarks/           # speed benchmarks (bench_engine)
├── src/
│   └── hanabi_search/
│       ├── __init__.py       # re-exports the public API only
│       ├── config.py         # Pydantic experiment config schema (YAML in configs/)
│       ├── exceptions.py     # project exception hierarchy
│       ├── env/              # the ONLY place that touches the engine:
│       │   ├── _pyhanabi.py  #   cffi declarations + library loading
│       │   ├── cards.py, moves.py, game.py, state.py, observation.py  # engine wrappers
│       │   ├── observation_types.py  # TypedDicts for dict observations/actions
│       │   ├── hanabi_env.py         # gym-like environment
│       │   ├── determinized_env.py   # re-determinizing env for MCTS rollouts
│       │   └── hand_sampler.py       # hidden-hand sampling (wraps the engine's HandSampler)
│       ├── agents/           # agent policies (one subpackage per family):
│       │   ├── base.py, factory.py, human.py
│       │   ├── rule_based/   #   ruleset.py, card_counting.py, agents.py
│       │   ├── mcts/         #   node.py, agent.py, parallel.py (Ray workers)
│       │   └── alphazero/    #   node.py, agent.py, parallel.py (Ray workers)
│       ├── networks/         # nn.Module definitions, no training logic
│       ├── training/         # transitions, replay buffer, AlphaZero trainer
│       ├── evaluation/       # game runner, score metrics
│       ├── experiment.py     # wires a config into agents, run dirs, training loop
│       ├── reproducibility.py  # seeding, device selection
│       └── logging_utils.py  # logger setup
├── tests/                    # Python tests, mirroring src/hanabi_search/
│   ├── conftest.py           # fixtures
│   ├── support.py            # shared helpers and constants
│   ├── unit/
│   └── integration/
├── scripts/                  # entry points: train.py, evaluate.py, play.py
├── configs/                  # experiment configs (one YAML per experiment)
├── notebooks/                # analysis only, never imported by the package
├── saved_models/             # earlier AlphaZero checkpoints (state dicts)
└── docs/                     # report, slides, results/; papers/ is local-only
```

Git-ignored, never committed: `build*/`, `dist/`, `.venv/`, `checkpoints/`,
`runs/`, `logs/`, `outputs/`, `wandb/`, `docs/papers/`, `*.so`, `*.dylib`, `*.o`,
`__pycache__/`.

### Where a new file goes

| You are adding…                                   | Put it in…                          |
|---------------------------------------------------|-------------------------------------|
| Game logic, rules, state representation (C++)     | `engine/hanabi_lib/`                |
| A new function exposed to Python                  | `engine/capi/` + `env/_pyhanabi.py` |
| Engine test / benchmark                           | `engine/tests/` / `engine/benchmarks/` |
| Observation encoding, env wrapper, action masking | `src/hanabi_search/env/`                 |
| A new agent / algorithm                           | `src/hanabi_search/agents/`              |
| A network architecture                            | `src/hanabi_search/networks/`            |
| Loss, buffer, training loop                       | `src/hanabi_search/training/`            |
| Metric or evaluation protocol                     | `src/hanabi_search/evaluation/`          |
| A runnable entry point                            | `scripts/`                          |
| Hyperparameters for a run                         | `configs/`                          |
| Python test                                       | `tests/unit/` or `tests/integration/` |
| Plots and result analysis                         | `notebooks/`                        |

### Structure rules

- **The engine is touched only through `env/`.** Agents, networks and training
  code work on tensors/arrays and our own types, never on raw cffi objects.
  Search agents (MCTS, AlphaZero) may use the `env` wrapper types (`HanabiState`,
  `HanabiMove`, `DeterminizedEnv`) to simulate, but only through
  re-determinization: never read their own hidden cards.
- **Do not create new top-level directories or files** without asking first.
- **Look before you create.** Check whether a module for that concern already
  exists and extend it instead.
- **No dumping grounds.** No `utils.py`, `helpers.py`, `misc.py`, `common.py`.
  Name modules after what they do (`observation_encoder.py`, not `utils.py`).
- **No stray files.** No scratch scripts, `test_tmp.py`, `*_old.py`, `*_v2.py`,
  `*_backup.*`, `train_new.py`, `hanabi_state_fast.cc` next to `hanabi_state.cc`,
  or commented-out code blocks. Improve code in place; history lives in git.
  Experiments differ by config file, not by copied scripts.
- **Split, don't sprawl.** A Python module past ~400 lines or a C++ file past
  ~800 lines is a signal to split by responsibility.
- **Move, don't copy.** When relocating code, move it and update imports/includes.
- **Keep this file current.** If you add a sub-package or change the layout,
  update the "Repository structure" section in the same change.

---

## Engine modification rules

The engine defines the game. A speed-up that changes the rules silently
invalidates every result, so correctness comes before speed.

### Correctness
- **Never change game rules** (deck composition, hints, life/information tokens,
  scoring, end-of-game conditions) unless explicitly asked.
- **Parity tests are mandatory.** `engine/tests/` keeps tests that replay
  fixed-seed games and compare against recorded upstream trajectories
  (legal moves, observations, rewards, final score) for every player count 2–5.
  Every engine change must pass them.
- If a change intentionally alters behavior or observations, update the golden
  data in the same change and explain why in `CHANGES.md`.

### Performance
- **Measure, don't guess.** Run `bench_engine` before and after any optimization,
  in Release mode, and record the numbers (games/s or steps/s) in the commit
  message and `CHANGES.md`. Reject changes that don't measurably help.
- Profile first (Instruments / `perf`-style sampling) to find the real hotspot.
- Typical wins to consider: avoid per-step heap allocations, avoid copying
  state where a reference or in-place update works, avoid string building on
  hot paths, reserve vector capacity, cache legal moves, batch many games per
  C API call to amortize Python↔C overhead.
- Never trade away readability for a gain under ~5% on the benchmark.

### Richer state / observations
- New state features are exposed through **new, explicitly named C API
  functions** (e.g. `pyhanabi_state_card_knowledge`), not by changing the
  meaning of existing ones.
- Write features directly into caller-provided buffers (flat arrays of
  fixed-width types) so Python can wrap them as NumPy arrays without copies.
- Respect hidden information: a player's observation must never include their
  own cards. Full-state features (for centralized critics) live in clearly
  separate functions named `*_full_state_*`.
- Document each new feature's layout (shape, dtype, meaning of each index) in
  the C header and mirror it in the Python encoder docstring.
- Bump the observation-encoder version when any feature layout changes; it is
  stored in checkpoints.

### Upstream hygiene
- Keep the upstream Apache-2.0 license and copyright headers in modified files;
  add our own notice below them.
- Log every engine change in `engine/CHANGES.md`: what, why, benchmark delta.

---

## RL and experiment standards

### Reproducibility
- Every run is fully defined by one file in `configs/` plus the git commit.
  No hyperparameters hard-coded in scripts.
- Seed Python, NumPy, PyTorch **and the engine** from the config, and log the
  seeds. Save the resolved config and encoder version next to every checkpoint.
- Each run writes to its own directory: `runs/<experiment>/<timestamp>_<seed>/`.

### Environment handling
- Always apply the legal-move mask before sampling or argmax. An illegal action
  is a bug, never something to "handle" with a penalty.
- Player count, hand size, colors and ranks come from the config; never assume
  2 players or the full game.

### Training and evaluation
- Separate training metrics (loss, entropy, return) from evaluation metrics
  (score over fixed evaluation seeds, greedy policy).
- Evaluate on a fixed, held-out set of seeds with ≥1,000 games; report mean
  score ± standard error.
- For coordination work, report both self-play and cross-play scores.
- Select the device explicitly (`mps` / `cpu`); never assume CUDA.
- Fail loudly on NaN/Inf losses rather than continuing silently.

---

## Python standards

### Style and tooling
- Formatting and linting: **ruff** (config in `pyproject.toml`). Line length 100.
- Full type hints on every function signature, including return types. Code must
  pass `mypy`. Modern syntax: `list[int]`, `X | None`, no `typing.List`.
- Annotate tensor shapes in docstrings or comments (`# (batch, num_actions)`).
- Naming: `snake_case` functions/variables, `PascalCase` classes, `UPPER_SNAKE`
  constants, leading `_` for private. Absolute imports, no wildcards.

### Design
- Small, single-purpose functions. Isolate I/O and environment stepping at the edges.
- `@dataclass` (or Pydantic at config boundaries) for structured data instead of
  loose dicts or tuples (e.g. a `Transition` dataclass, not a 6-tuple).
- Every `__init__.py` re-exports only the public interface via `__all__`.
- Google-style docstrings on all public functions and classes.

### Errors and logging
- Raise specific exceptions from `exceptions.py`; never `raise Exception`, never
  bare `except:`, never silently swallow errors.
- One logger per module: `logger = logging.getLogger(__name__)`. No `print()`
  outside `scripts/`. Lazy formatting: `logger.info("Score %.2f", score)`.

### Testing
- **pytest**. Files `test_<module>.py`, functions `test_<behavior>`.
- Every public function gets a happy-path test and an error-case test.
- Env wrappers: test observation shapes, legal-move masks, and that a full
  random-policy game runs to completion for every supported player count.
- Agents: smoke test that a few training steps run and loss is finite.

### Dependencies
- Add with `uv add <pkg>` (dev tools: `uv add --dev <pkg>`). Never `pip install`.
- Ask before adding a substantial new dependency.

---

## C++ / C standards (engine)

### Language and build
- Engine core: **C++23** (clang 18+ or GCC 15+; `std::mdspan` sets the floor).
  C API (`engine/capi/`): plain C types only inside `extern "C"` — no C++
  types, exceptions or templates cross the boundary.
- Warnings: `-Wall -Wextra -Wpedantic -Wshadow`. New and modified code compiles
  warning-free; fix inherited warnings in files you touch.
- Debug builds use `-fsanitize=address,undefined`; tests must run clean under them.
- Formatting: **clang-format**. Static analysis: **clang-tidy**.

### C++ style
- RAII for every resource; no raw `new`/`delete` — use values,
  `std::unique_ptr`, or containers. Exception: the opaque handles the C API
  hands to Python, freed by their matching `Delete*` function.
- Pass large objects by `const&`; return by value (rely on move/RVO).
- `const`-correct everywhere; mark single-argument constructors `explicit`.
- Prefer `std::array` / fixed-size storage for small fixed-size game data
  (hands, fireworks) over `std::vector` on hot paths.
- No exceptions escape into the C API: catch at the boundary and convert to a
  status code.

### C API rules
- Prefix every new exported symbol with `pyhanabi_`. Existing upstream functions
  (`NewGame`, `StateApplyMove`, ...) keep their unprefixed names.
- Use fixed-width types (`int32_t`, `uint8_t`, `size_t`) for all parameters.
- Errors via return status codes, never `exit()`/`abort()`: a bad index, id,
  move or parameter returns `false` (or `-1` from an `int` getter). Agreed
  exception: a NULL pointer argument is a caller bug and aborts via `REQUIRE`.
- Every function documents who owns returned memory. Pair every allocating
  call with a `*_delete` / `*_free` function. Prefer caller-provided buffers.
- Doc comment on every exported function: purpose, params, return, ownership,
  buffer layout.

---

## Python ↔ engine boundary

- `src/hanabi_search/env/_pyhanabi.py` holds the cffi declarations; they must match
  `engine/capi/pyhanabi.h` exactly (`tests/unit/env/test_pyhanabi_binding.py`
  checks this). When a C API signature changes, update the
  header, the cffi declarations, the Python wrapper, its type hints and tests on
  both sides **in the same change**.
- cffi objects own C memory: free them deterministically, never keep them in
  replay buffers. Convert to NumPy/tensors first.
- Minimize Python↔C crossings on hot paths: prefer batched calls that step or
  encode many games at once over per-game, per-field calls.
- Validate array shape, dtype and contiguity before handing buffers to C.

---

## Communication

Keep replies short; keep the work thorough. These rules limit what you write
to me, never how carefully you think, plan, read code or verify.

- Lead with the result: what changed, or the answer, in 1–3 sentences.
- Then at most a short bullet list: files touched, test/benchmark results,
  anything I must decide or check. No recap of steps you took.
- Don't restate my request, don't paste back code you just wrote (I can read
  the diff), don't explain standard concepts unless I ask.
- Report numbers, not adjectives: "tests pass (142), +38% steps/s", not
  "everything works great and it's much faster".
- Ask questions only when blocked; one question at a time, with your
  recommended option.
- Go long only when I ask for an explanation, a plan, or a review, or when
  something went wrong and the details matter.
- Never shorten by skipping work: still read the relevant code, run the tests
  and benchmarks, and mention any failure or risk, however briefly.

---

## Workflow expectations

- Read the surrounding code before editing; follow existing patterns.
- Make the smallest change that fully solves the task; do not refactor unrelated
  code unless asked.
- For a large or structural change, anything touching game rules, observation
  layout, the C API or the checkpoint format, propose a plan first and wait for
  approval.
- After changes: format, lint, type-check, build, run tests (including engine
  parity tests) and, for engine changes, the benchmark.
- Never launch long training runs without being asked; use short smoke runs.
- Commit messages: imperative mood, short summary line
  (`Cache legal moves in HanabiState (+38% steps/s)`).