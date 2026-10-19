# Upstream

`engine/` is a hard fork of DeepMind's Hanabi Learning Environment (HLE). We do
not merge from upstream; this file records where the code came from.

| | |
|---|---|
| Repository | https://github.com/deepmind/hanabi-learning-environment |
| Forked from | `5df6a73` (2019-12-02, "Merge pull request #25 from carlbalmer/pip_install") |
| License | Apache-2.0, see `LICENSE` (kept as-is) |

## How the fork point was determined

The original fork was not recorded. Our `hanabi_lib/` and C API were compared
against every upstream commit. Upstream's engine files are identical across
`add3da8`…`5df6a73`, and that range is the closest match to ours. Later upstream
commits (comment-only changes to `HanabiGame` in 2020 and a `StateParentGame` fix
in 2021) are not in our fork. `5df6a73` is the last commit of that range.

## Mapping to upstream paths

| Ours | Upstream (`hanabi_learning_environment/`) |
|---|---|
| `engine/hanabi_lib/` | `hanabi_lib/` |
| `engine/capi/pyhanabi.{h,cc}` | `pyhanabi.{h,cc}` |
| `src/hanabi_search/env/` (Python binding, env wrapper) | `pyhanabi.py`, `rl_env.py` |

Not taken from upstream: `engine/tests/`, `engine/benchmarks/` and the CMake
build, which are ours. The parity golden data in `engine/tests/parity/` is
recorded from upstream at the commit above.
