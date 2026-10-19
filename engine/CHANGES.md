# Engine changes vs upstream

Every change to `engine/` relative to upstream HLE (see `UPSTREAM.md`): what,
why, and the benchmark delta. Newest first.

Benchmark: `./build/engine/benchmarks/bench_engine` (Release, default 20,000
games), on an Apple Silicon MacBook unless stated otherwise.

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
`5df6a73`. Several carry `MB:` comments in the code; the author of those is not
recorded.

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
  is not in the deck.
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
