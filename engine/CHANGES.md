# Engine changes vs upstream

Every change to `engine/` relative to upstream HLE (see `UPSTREAM.md`): what,
why, and the benchmark delta. Newest first.

Benchmark: `./build/engine/benchmarks/bench_engine` (Release, default 20,000
games), on an Apple Silicon MacBook unless stated otherwise.

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
