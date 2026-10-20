# Engine changes vs upstream

Every change to `engine/` relative to upstream HLE (see `UPSTREAM.md`): what,
why, and the benchmark delta. Newest first.

Benchmark: `./build/engine/benchmarks/bench_engine` (Release, default 20,000
games), on an Apple Silicon MacBook unless stated otherwise.

---

## 2026-10-20: Observation buffers in the C API

Python built each observation through 100+ small C calls (one or more per
card, hint, discard and legal move) and encoded it by parsing a
comma-separated string. Seven new functions in `capi/pyhanabi_observation.cc`
each fill one whole field into caller-provided buffers (layouts in
`pyhanabi.h`):

| Function | Output |
|---|---|
| `pyhanabi_observation_hands` | hand sizes (P); colors, ranks (P, H), -1 hidden/empty |
| `pyhanabi_observation_card_knowledge` | hinted color, rank (P, H); plausible mask (P, H, C, R) |
| `pyhanabi_observation_discard_pile` | colors, ranks, oldest first |
| `pyhanabi_observation_fireworks` | heights (C) |
| `pyhanabi_observation_legal_move_uids` | move ids, engine order |
| `pyhanabi_observation_unseen_card_counts` | copies the observer cannot see (C, R): new feature |
| `pyhanabi_observation_encode` | canonical encoding as uint8 bits |

- The counting is in `hanabi_lib/observation_features.{h,cc}`
  (`UnseenCardCounts`, `PlausibleCards`); both read only the observation, so
  the observer's own cards are never used. The canonical encoding is
  unchanged (encoder version stays `hle-canonical-1`).
- Wrong buffer sizes return false (or -1) and write nothing.
- Tests (`tests/test_observation_buffers.cc`): every buffer equals the per-card
  `Obs*` getters along random games (2-5 players, 2 and 5 colors); unseen
  counts equal deck + own hand; swapping one of the observer's cards with a
  deck card changes no buffer and no encoding bit; wrong sizes are rejected.
- No game logic changed; parity tests pass.

Benchmark: new row `observation features` (observation + both features):
786k / 527k / 487k / 398k obs/s for 2-5 players. Other rows within noise
(median of 5 alternating runs, largest delta -4.9%).

End to end, with the Python side reading these buffers (observation dicts and
rule-based card counting): MCTS 2p 2,617 -> 4,855, 3p 1,943 -> 3,833, 5p
1,539 -> 3,171 rollouts/s (+86%, +97%, +106%); observation dict 2.1-2.6x;
encoding from Python 30-44x (2p 32k -> 971k/s). Observation dicts (all
players and observation types), encodings, rule outputs and MCTS decisions
are identical to before.

## 2026-10-20: Split observation functions out of `capi/pyhanabi.cc`

Code move only, no behavior change: the `Obs*` and observation-encoder
functions moved to `capi/pyhanabi_observation.cc`, and the helpers both files
use (`InRange`, `CopyCard`, `Observation`) to `capi/pyhanabi_internal.h`.
`pyhanabi.cc` was 931 lines, past the ~800-line split threshold, and the
observation buffer functions that follow belong with the other `Obs*` ones.
`pyhanabi.h` stays the one header (cffi reads it).

## 2026-10-20: Hidden-hand sampling in the engine (`HandSampler`)

Re-determinizing MCTS resampled other players' hands in Python
(`src/hanabi_search/env/card_sampler.py`), with one C API call per returned and
dealt card. It now runs in the engine: `hanabi_lib/hand_sampler.{h,cc}`.

- Same algorithm as the Python version: card-by-card sampling from the cards
  the player cannot see, filtered by their hints, restarting on a dead end;
  restoring deals the remembered hand back and replaces a copy of the played
  or discarded card if it is no longer available (forgetting the hints for
  that card if nothing else fits). Hands are still swapped through return and
  deal-specific moves, so the move history is unchanged.
- Differences: a failed resample or restore leaves the state unchanged
  (computed on a copy) and reports failure, where Python could loop forever
  or stop half-dealt; resampling gives up after 10,000 restarts.
- Randomness: each call takes a seed and uses SplitMix64, which is cheap to
  seed and gives the same results on every platform. `DeterminizedEnv` draws
  the seed from Python's `random`, so seeding a game still makes MCTS games
  reproducible (`test_same_seeds_give_identical_mcts_games`). The game's RNG is
  never used. MCTS games for a given seed differ from before, since the
  random stream differs.
- C API: `pyhanabi_state_candidate_counts` (cards that could be at a hand
  position, into a caller-provided `int32_t` buffer), `pyhanabi_state_resample_hand`,
  `pyhanabi_state_restore_hand`. Python: `candidate_counts`, `resample_hand`,
  `restore_hand` in `env/hand_sampler.py`; `card_sampler.py` is removed.
- Equivalence with the Python sampler, checked before removing it:
  `candidate_counts` matched `valid_cards` in 4,880 of 4,880 comparisons
  (2-5 players, random states, with and without hints and exclusions); hand
  distributions over 10,000 rollout steps in 4 scenarios (54 hand positions)
  differ from the old ones by at most 0.029 total variation, the same as two
  runs of the old code with different seeds (max 0.031, mean 0.011 for both).
- Tests: `tests/test_hand_sampler.cc` (8, including a frequency test of the
  first sampled card and whole RESTORE rollouts for 2-5 players) and 1 new C
  API test; all pass in Release and under ASan + UBSan.
- Benchmark: new `resample hand` row, 762k / 632k / 651k / 573k hands/s for
  2-5 players (the Python version did about 6k/s). Other rows unchanged
  within noise (medians of 5-6 alternating runs, random games 0 to -2%).
- End to end: MCTS (3 players, 50 rollouts per move, 12 fixed states, best of
  5) went from 769 to 1,096 rollouts/s (+42%).

---

## 2026-10-20: Status returns for bad arguments across the C API

About 25 C API functions took an index, id or parameter without checking it:
a bad value from Python read out of bounds, threw an uncaught `std::out_of_range`
(terminating the process), or hit an abort. They now check and report failure.
Valid calls behave exactly as before. **C API signatures changed**; the cffi
declarations and Python wrappers changed with them.

- `void` → `bool` (false on a bad argument, output handle untouched):
  `GetMove`, `StateApplyMove` (illegal move), `StateDealCard` (not a chance
  turn, or empty deck), `StateRemoveKnowledge`, `StateGetDiscard`,
  `StateGetHandCard`, `StateGetMoveHistory`, `NewGame`, `GetMoveByUid`,
  `NewObservation`, `ObsGetHandCard`, `ObsGetHandCardKnowledge`,
  `ObsGetDiscard`, `ObsGetLastMove`, `ObsGetLegalMove`, `NewObservationEncoder`.
- `int` getters return -1 for an out-of-range argument: `StateFireworks`,
  `StateGetHandSize`, `ObsGetHandSize`, `ObsFireworks`, `ColorIsPlausible`,
  `RankIsPlausible`.
- `Get*Move` return false when a value does not fit the move's 8-bit fields
  (`GetReturnMove(0, 300)` used to become player 44).
- Removed `StateParentGame`: it returned a raw `HanabiGame*`, and nothing used it.
- `HanabiGame`'s constructor throws `std::invalid_argument` instead of aborting
  on invalid parameters, and also rejects `hand_size < 1`,
  `max_information_tokens < 0`, `max_life_tokens < 1` and an unknown
  `observation_type` (all previously accepted, then crashed or misbehaved
  later). `NewGame` catches and returns false; `fromJSON` now relies on the
  constructor for its range checks.
- NULL pointer arguments still abort through `REQUIRE`: they are bugs in our
  own wrapper, never bad data (agreed exception, recorded in CLAUDE.md).
- Python: wrappers raise `IllegalMoveError` (illegal move), `ConfigError`
  (invalid game parameters) or `EngineCallError` (bad index or id).
- Tests: 5 new tests in `tests/test_capi_robustness.cc` (13 in total); all
  engine tests pass in Release and under ASan + UBSan; pytest 184 pass.
- Benchmark (median of 5 alternating runs, previous commit → this one): every
  row within ±3% (2p random games 1074k → 1069k steps/s, JSON round-trip
  11.8k → 11.9k/s), i.e. no measurable change. `bench_engine` does not go
  through the C API; there, `StateApplyMove` now also calls `MoveIsLegal`.

---

## 2026-10-20: Reject bad input at the C API instead of crashing

Malformed or out-of-range input could terminate the Python process or read
outside the engine's arrays. Valid input behaves exactly as before.

- **JSON loaders** (`{Move,Game,HistoryItem,State}FromJson`): every exception
  is caught at the C API boundary and the function returns `false`. Before,
  only `StateFromJson` caught; the other three terminated the process on any
  bad input. A failed load leaves the handle's previous object untouched
  (`MoveFromJson` used to delete it first and leave a dangling pointer).
  `*ToJson` return `NULL` on failure.
- **Field checks** (`fromJSON` in card, move, history item, hand, deck, game,
  state; helpers in `hanabi_lib/json_checks.{h,cc}`): every field must be
  present, of the right JSON type, and in range. Before, presence was checked
  with `assert` (removed in Release), integers were silently truncated to
  `int8_t`, and a missing key read through `operator[]` was undefined behavior.
  - A state is checked against its game: number of hands, cards per hand,
    card colors and ranks, card-knowledge shapes, deck shape and total, token
    counts, fireworks, players and cards named in the move history. Whether
    the cards add up to the full deck is not checked (decided: format only).
  - A game's parameters are checked before the constructor runs, since it
    aborts on invalid ones. It is now built from the JSON's typed fields;
    `params` is stored as before. `cards_per_color`, `moves` and
    `chance_outcomes` must match what the parameters produce, and `rng_state`
    must parse.
- **`MoveIsLegal`** returns `false` instead of reading out of bounds for:
  - return moves with a player outside `[0, players)` or a negative card index;
  - deal-specific moves with a bad player, an insert index outside
    `[0, cards in hand]`, a full hand, or a color/rank outside the game;
  - deal moves with a color/rank outside the game;
  - play and discard moves with a negative card index, or on a chance turn.

  Deal-specific moves no longer print to stdout when the card is not in the
  deck.
- Also: `-Wsign-compare` and `-Wdeprecated-copy` warnings fixed in the files
  touched (defaulted copy assignment for `HanabiState` and `HanabiHand`).
- Tests: `tests/test_capi_robustness.cc` (8 tests, through the C API). Run
  against the previous engine, the first one terminates the process and the
  legality tests hit a heap-buffer-overflow under ASan. All tests and the
  parity test pass in Release and under ASan + UBSan.
- Benchmark (median of 5 alternating runs, previous engine → this one; both
  built with Command Line Tools 27.0, so higher than the table below):

| Players | random games (steps/s) | + observation encode (steps/s) | state copy (/s) | state JSON round-trip (/s) |
|---|---|---|---|---|
| 2 | 1038k → 1056k | 412k → 419k | 2003k → 1995k | 11.9k → 11.8k |
| 3 | 1022k → 1032k | 329k → 328k | 1431k → 1438k | 8.9k → 9.0k |
| 4 | 1030k → 1032k | 311k → 310k | 1316k → 1294k | 8.5k → 8.6k |
| 5 | 896k → 900k | 265k → 265k | 1078k → 1071k | 7.2k → 7.4k |

  All within ±2%, i.e. no measurable change. A first version built the error
  messages on every check and cost 6% on the JSON round-trip; they are now
  built only on failure.

---

## 2026-10-19: Fix `HanabiState::fromJSON` advancing the game's RNG

`fromJSON` built its temporary state with the default start player. With
`random_start_player` set, that sampled a start player from the game's RNG, so
every deserialized state's game was one RNG draw ahead of the original, and the
restored state dealt different cards. It now passes an explicit start player
(the real one is read from the JSON afterwards).

- Regression test: `StateFromJsonDoesNotAdvanceGameRng`.
- Parity tests pass (they do not deserialize).
- Benchmark: state JSON round-trip 11.2k / 8.5k / 8.0k / 6.8k per second for
  2-5 players, unchanged within noise (was 11.3k / 8.5k / 8.0k / 6.9k).

---

## 2026-10-19: Move into `engine/`, CMake build, parity tests, benchmark

No game-logic changes.

- Moved `hanabi_lib/` to `engine/hanabi_lib/` and `pyhanabi.{h,cc}` to
  `engine/capi/`.
- Build: C++17 (was C++11); warnings `-Wall -Wextra -Wpedantic -Wshadow`;
  `-march=native` only in Release and switchable with `HANABI_NATIVE_ARCH`;
  Debug builds use ASan + UBSan. nlohmann/json comes from the system if present,
  otherwise it is fetched at a pinned version and hash.
- Added `tests/parity/`: 56 golden games recorded from upstream `5df6a73` (2–5
  players, all observation types, a reduced game). Our engine matches upstream
  at every step. The parity and fork-extension tests pass under ASan + UBSan.
- Added `tests/test_fork_extensions.cc` covering the pre-existing additions
  listed below.
- Added `benchmarks/bench_engine.cc`. Baseline:

| Players | random games (steps/s) | + observation encode (steps/s) | state copy (/s) | state JSON round-trip (/s) |
|---|---|---|---|---|
| 2 | 947k | 244k | 951k | 11.3k |
| 3 | 874k | 198k | 661k | 8.5k |
| 4 | 881k | 188k | 554k | 8.0k |
| 5 | 772k | 159k | 495k | 6.9k |

A state JSON round-trip costs about as much as 80 state copies.

---

## Before 2026-10: pre-existing fork changes (reconstructed)

These predate this log. They were reconstructed by diffing against upstream
`5df6a73`. Most come from Michael Brooks (MBlogs), whose repository this project
started from: he had begun adding RIS-MCTS support (re-determinization moves,
knowledge removal). His changes carry `MB:` comments in the code.

### Re-determinization moves (for MCTS)
Used by the MCTS agents to swap the hidden cards in a player's hand.
- New move types `kReturn` and `kDealSpecific` (`hanabi_move.h`).
- `kReturn`: puts a card from any player's hand back into the deck.
  `TargetOffset` holds the **absolute player id**, not an offset. The card's
  knowledge is kept.
- `kDealSpecific`: chance move that inserts a chosen card at a chosen index in a
  player's hand (`TargetOffset` is again an absolute player id). The turn then
  returns to the player who was acting (`AdvanceToNextPlayer(stayOnPlayer)`).
- Neither move counts down the end-game turn counter when the deck is empty.
- Both are recorded in the move history. They therefore appear in observations'
  last moves; the canonical encoder skips their move-type bits instead of
  aborting (`canonical_encoders.cc`).
- `MoveIsLegal` for `kDealSpecific` prints a diagnostic to stdout when the card
  is not in the deck (removed 2026-10-20).
- New helpers: `HanabiHand::InsertCard`, `HanabiHand::ReturnFromHand`,
  `HanabiDeck::ReturnCard`.

### Knowledge removal
- `HanabiState::RemoveKnowledge(player, card_index)` and
  `HanabiHand::RemoveKnowledge` reset one card's hint knowledge.

### JSON serialization (nlohmann/json)
- `toJSON()` / `fromJSON()` on card, move, hand (and card knowledge), history
  item, deck, state, observation and game. Game serialization includes the RNG
  state.
- Used to ship states to Ray workers.

### C API (`capi/pyhanabi.h`)
- Added: `GetReturnMove`, `GetDealSpecificMove`, `StateRemoveKnowledge`,
  `StateTurnsToPlay`, and `{Move,Game,HistoryItem,State}{To,From}Json`.
- Renamed: `StateDealRandomCard` → `StateDealCard` (same behavior).
- Long declarations were joined onto single lines (no signature changes).
