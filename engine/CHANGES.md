# Engine changes vs upstream

Every change to `engine/` relative to upstream HLE (see `UPSTREAM.md`): what,
why, and the benchmark delta. Newest first.

Benchmark: `./build/engine/benchmarks/bench_engine` (Release, default 20,000
games), on an Apple Silicon MacBook unless stated otherwise.

---

## 2026-10-20: Link-time optimization; static C++ runtime on Linux

Modernization step 5 (build only, no source change).

- Release builds use link-time optimization when the compiler supports it
  (`check_ipo_supported`; option `HANABI_LTO`, default ON). Debug builds are
  unaffected.
- On Linux, `libpyhanabi` links `-static-libstdc++ -static-libgcc`, so Python
  can load an engine built with a newer compiler than the system's.

Median of 5 (search iteration) / 3 (other rows), 3 players:

| Row | Before | After |
|---|---|---|
| search iteration | 257,006 | 268,612 (1.05x; 5.66x original baseline) |
| random games | 3,354,891 | 3,784,928 (1.13x) |
| random games + encode | 1,031,219 | 1,130,683 (1.10x) |
| observation features | 2,557,299 | 2,424,194 (0.95x) |

Search iteration 2/4/5 players: 1.03x / 1.04x / 1.05x. Observation features
loses 4-5% at every player count (different inlining); kept because the
search loop, the headline metric, gains. Other rows within noise.

---

## 2026-10-20: `std::expected` and error reasons through the C API

Modernization step 4. Failures now say why, from the engine to Python.

- `HanabiGame::Create(params)` returns `std::expected<HanabiGame, std::string>`
  (the throwing constructor stays for internal use).
  `HanabiState::ReplaceHand`, `HandSampler::ResampleHand` and `RestoreHand`
  return `Status` (`std::expected<void, std::string>`, in `util.h`) with the
  reason, e.g. "no hand fits player 1's knowledge (10000 attempts)".
- New C API function `const char* pyhanabi_last_error(void)`: the reason for
  the calling thread's most recent failing call (thread-local, owned by the
  engine, valid until the next failure). Every C API path that returns
  false / -1 / NULL records one ("player 5 is not in [0, 3)", "move (Play 0)
  is not legal in this state", "invalid JSON: 'cur_player' ...").
- Python: `engine_message(context)` (in `_pyhanabi.py`) appends the reason to
  every `ConfigError`, `EngineCallError`, `SerializationError` and
  `IllegalMoveError` raised after an engine failure.
- Inside the JSON parser, exceptions remain (dozens of nested reads; without
  a `?` operator, `expected` would add an error-forwarding line after each);
  they are caught once at the C API boundary and their message recorded.

Tests: `FailuresExplainThemselves` (C API), sampler reasons, and Python tests
matching the reasons in exception messages. No behavior change otherwise
(dumps, rule outputs and MCTS decisions identical); search iteration
unchanged (257,939 -> 258,221, 3p).

---

## 2026-10-20: `std::mdspan` for multi-dimensional buffers

Modernization step 3. Code that indexed flat buffers by hand
(`((offset * hand_size + index) * num_colors + color) * num_ranks + rank`)
uses `std::mdspan` views instead (`grid[offset, index, color, rank]`):
`UnseenCardCounts`, `PlausibleCards`, and the C API's hand and knowledge
buffers. The observation and state hand buffers now share one writer
(`pyhanabi_internal::WriteHands`). The sampler's card counts keep their flat
index, which is the card's identity there.

Performance note: the first version made `observation features` 13-16%
slower. Writes through `uint8_t*` may alias any object, so the compiler
reloaded `game.NumColors()` / `NumRanks()` in every loop test; they are now
locals (as the old code had them), and each card's [color, rank] grid is
located once. Final, median of 5 (3 players): observation features
2,575,895 -> 2,485,295 (-4%; 2/4/5 players -2% / -5% / +2%, near noise);
search iteration 254,863 -> 257,010 (+1%). No behavior change.

---

## 2026-10-20: `std::span` for card sequences; warning-free engine (search iteration 5.43x)

Modernization step 2. Functions taking "some cards" take
`std::span<const HanabiCard>` instead of `const std::vector<HanabiCard>&`:
`HanabiState::ReplaceHand`, `HandSampler::CandidateCounts` and
`RestoreHand`, and the C API's `CopyCard` (no longer a template). A hand's
cards (a `FixedVector`) now pass without being copied into a vector. The
sampler's planned hand, its "forget knowledge" flags and its list of later
cards became fixed-size, removing its last per-call allocations; that is a
measurable gain on its own. `RestoreHand` rejects a `remembered` longer than
8 cards. `HanabiCard::operator==` is `= default` (`HanabiMove`'s stays
hand-written: it compares only the fields each move type uses).

Cleanup: the remaining reserved include guards (`__X_H__`) were renamed,
duplicate switch branches in `hanabi_move.cc` merged and the last
sign-compare / emplace warnings in `hanabi_observation.cc` fixed: the engine
now builds with no compiler warnings and no clang-tidy findings.

No behavior change: observation/encoding dumps, rule outputs and MCTS
decisions identical; all tests pass.

| Row (3 players) | Before | After |
|---|---|---|
| search iteration | 229,149 | 257,618 (1.12x; 5.43x baseline) |

2/4/5 players: 1.11x / 1.12x / 1.12x over the previous step.

---

## 2026-10-20: Build as C++23

`CMAKE_CXX_STANDARD` 17 -> 23 (approved modernization plan, step 1 of 6). No
code change; the engine compiles warning-free as before and all tests pass.
Requirements are now clang >= 18 or GCC >= 15 (for `std::mdspan`, used in a
later step). bench_engine search iteration unchanged (231,111 vs 231,636 3p).
On macOS, clang-tidy now needs `--extra-arg=-resource-dir=$(xcrun clang
-print-resource-dir)` (see CLAUDE.md): from C++20 on, the SDK's `math.h`
expects the compiler's `float.h` to define `INFINITY`, and the pip LLVM's
does not.

---

## 2026-10-20: Hands stored inline, at most 8 cards (search iteration 4.91x)

A hand held two `std::vector`s (cards, knowledge): every copy of a state or
observation allocated two per player. Hands now store both in a new
`FixedVector<T, 8>` (`hanabi_lib/fixed_vector.h`: inline storage, the subset
of the vector interface hands use, aborts past capacity; its conversion to
`std::vector` is explicit so a hand is never copied by accident, which
caught the C API's `CopyCard` copying the whole hand to read one card).

**Parameter limit:** `hand_size` must now be in [1, 8] (engine, hand JSON and
the Python `GameConfig`). Larger hands were already broken: hint outcomes are
8-bit masks over hand positions, so hints silently missed cards past
position 8. No game behavior changes: observations, encodings, rule outputs
and MCTS decisions are identical; parity tests pass;
`tests/test_fast_paths.cc` checks `FixedVector` against `std::vector` on
random operation sequences.

| Row (3 players) | Before | After |
|---|---|---|
| search iteration | 209,527 | 232,646 (1.11x; 4.91x baseline) |
| state copy | 4,596,028 | 10,346,120 (2.25x) |
| observation features | 1,431,268 | 2,493,248 (1.74x) |
| random games + encode | 813,462 | 1,059,985 (1.30x) |

Search iteration 2/4/5 players: 4.14x / 5.46x / 5.89x baseline. JSON and
resample rows unchanged. Also: inherited reserved include guards and an
unparenthesized macro argument fixed in `util.h` and `pyhanabi.h`.

---

## 2026-10-20: Re-determinize outside the move history (fixes AlphaZero leaf inputs)

**Behavior change (approved).** `HandSampler` swapped hands through the
fork's return and deal-specific moves, which `ApplyMove` recorded in the move
history. Observations then took those swaps for real moves, and the
canonical encoder's "last action" picked them up: after resampling another
player's hand, the observer's last action was encoded as empty in 160/160
test states, and a deal-specific item (player -1) set the last bit of the
discard section (in a standard game: "the last blue 5 was discarded") in
151/160. Every AlphaZero search leaf is encoded from such a state, while its
training inputs come from real games.

- New `HanabiState::ReplaceHand(player, cards)` swaps a hand in place: old
  cards back to the deck, new ones drawn from it, knowledge kept per
  position. It is not a move, so history, current player, tokens and turn
  count are unchanged; impossible swaps return false and change nothing.
- `ResampleHand` / `RestoreHand` use it. Their random draws are unchanged, so
  they sample exactly the same hands as before.
- The encoder's last action now also skips return and deal-specific items, in
  case a caller applies those moves directly (real games never contain them,
  so parity is unaffected).
- Return and deal-specific moves stay available (C API, Python wrappers).

Effects: real-game observations, encodings and rule outputs are unchanged
(identical dumps); re-determinized states now look like real ones (new tests:
resampling is invisible to the resampled player, records no moves, and
leaves every observer's last moves unchanged). Python MCTS decisions on the
fixed test positions are unchanged (144/144).

Evaluation on the same seeds, before vs after (paired):

| Agents | Games | Before | After | Paired difference |
|---|---|---|---|---|
| MCTS + 2 VanDenBergh, 3p (`mcts_vs_vandenbergh_3p`) | 500 | 17.686 +- 0.095 | 17.686 +- 0.095 | 0 (all games identical) |
| AlphaZero self-play, 2p (`alphazero_2p`, `alphazero_model_100-NR.pth`) | 300 | 10.617 +- 0.121 | 11.227 +- 0.128 | +0.610 +- 0.175 |

Speed (swaps no longer go through `ApplyMove`, and history no longer grows
during search):

| Row (3 players) | Before | After |
|---|---|---|
| search iteration | 186,334 | 208,429 (1.12x; 4.49x baseline) |
| resample hand | 2,689,799 | 4,080,974 (1.52x) |

Search iteration 2/4/5 players: 3.83x / 5.03x / 5.46x baseline. Other rows
within noise (-4% to +4%).

---

## 2026-10-20: Encode into a caller buffer (search iteration 3.96x)

`CanonicalObservationEncoder::Encode` returns a fresh `std::vector<int>`
(658 ints for 3 players) and recomputed `Shape()` (another vector) on every
call; the C API then copied it into the caller's uint8 buffer. The section
writers are now templates on the entry type, and the new
`EncodeInto(obs, uint8_t* bits)` writes the same encoding straight into a
buffer; `pyhanabi_observation_encode` uses it. `Encode` is unchanged for
other callers. `tests/test_fast_paths.cc` checks both agree for every
observer of every state of random playouts (all three observation types),
including states re-determinized by `HandSampler`.

| Row (3 players) | Before | After |
|---|---|---|
| search iteration | 178,757 | 185,179 (1.04x; 3.96x baseline) |

Kept despite the small gain because it removes code (the C API's copy loop);
other rows within noise. Also: inherited sign-compare and clang-tidy
warnings fixed in the encoder files, and `CommitHand` no longer checks
legality twice (`ApplyMove` checks).

---

## 2026-10-20: Random deals from fixed arrays (search iteration 3.82x)

`ApplyRandomChance` built the full `ChanceOutcomes()` list (two growing
vectors, `MoveIsLegal` on each of the 25 outcomes) to draw one card. It now
fills fixed-size arrays with the same outcomes and probabilities in the same
order and draws with the same `discrete_distribution` on the game's RNG
(`HanabiGame::PickRandomIndex`, which `PickRandomChance` now uses too), so
every deal is identical: parity tests pass, and
`tests/test_fast_paths.cc` plays each variant in lockstep against the
`ChanceOutcomes()` path (uniform weights make it fail).

| Row (3 players) | Before | After |
|---|---|---|
| search iteration | 156,607 | 180,684 (1.15x; 3.82x baseline) |
| random games | 1,479,222 | 2,938,041 (1.99x) |
| random games + encode | 614,727 | 781,127 (1.27x) |

Search iteration 2/4/5 players: 3.35x / 4.18x / 4.45x baseline. Other rows
within noise (-3% to +3%).

---

## 2026-10-20: Generate legal moves directly (search iteration 3.33x)

`LegalMoves` tested every move id with `MoveIsLegal` (30 ids for 3 players,
each a full switch, hints scanning the target hand). It now generates the
legal moves directly in move-id order: discards while tokens are below the
maximum, every play, then the color and rank hints present in each target
hand (one pass per hand). New `tests/test_fast_paths.cc` checks it equals
the `MoveIsLegal` scan at every state of random playouts over 9 game
variants (2-5 players, 1-5 colors and ranks, hand sizes 1-8, 0-8 tokens);
a deliberately broken condition makes it fail.

| Row (3 players) | Before | After |
|---|---|---|
| search iteration | 124,342 | 156,225 (1.26x; 3.33x baseline) |
| random games | 1,150,052 | 1,486,291 (1.29x) |
| observation features | 1,117,875 | 1,441,991 (1.29x) |

Search iteration 2/4/5 players: 2.74x / 3.67x / 3.91x baseline. Other rows
within noise (-6% to +3%).

---

## 2026-10-20: Hand sampling without state copies (search iteration 2.69x)

`ResampleHand` and `RestoreHand` copied the whole state (move history
included) to stay atomic, assigned it back on success, and allocated count
vectors in their inner loops. They now plan the new hand on fixed-size card
counts (the deck plus the returned hand), then apply the returns and
deal-specific moves to the state only on success; a failure still leaves the
state untouched. Random draws happen in the same order, so samples are
identical: Python MCTS decisions match before and after (72/72, with and
without rule filtering). `RestoreHand` now also rejects cards that are not in
the game (the C API already did).

| Row (3 players) | Before | After |
|---|---|---|
| search iteration | 79,347 | 125,364 (1.58x; 2.69x baseline) |
| resample hand | 1,156,975 | 2,557,054 (2.21x) |

Search iteration 2/4/5 players: 2.26x / 2.92x / 3.10x baseline. Other rows
within noise (-6% to +1%).

---

## 2026-10-20: Card knowledge as a bitmask (search iteration 1.67x)

`ValueKnowledge` stored plausible values in a `std::vector<bool>`, so every
card's knowledge held two heap allocations, paid on every copy of a state,
hand or observation. Profiling the search-iteration row showed ~58% of time
in malloc/free. Plausible values are now bits of a `uint8_t` (games have at
most 5 colors and ranks; the constructor checks the range), making knowledge
6 bytes and trivially copyable. Behavior and the JSON format are unchanged;
parity tests pass.

| Row (3 players) | Before | After |
|---|---|---|
| search iteration | 47,458 | 79,344 (1.67x) |
| state copy | 1,457,668 | 4,670,051 (3.20x) |
| observation features | 525,347 | 1,103,715 (2.10x) |
| random games + encode | 326,990 | 511,419 (1.56x) |

Search iteration 2/4/5 players: 1.55x / 1.77x / 1.91x. JSON round-trip
-1 to -3% (noise range; not a hot path).

---

## 2026-10-20: `search iteration` benchmark row (headline engine metric)

`bench_engine` gets a `search iteration` row: one iteration of a
re-determinizing tree search done entirely in the engine (copy a mid-game
root state, resample the searcher's hand, replay up to 10 random legal moves
with dealing and RESTORE re-determinization, then encode the leaf observation
into a uint8 buffer). 2,000 fixed roots x 25 rounds per player count. This is
the metric engine optimizations are judged by from now on; a second argument
runs only the rows whose name contains it (`bench_engine 20000 "search"`).

Baseline (median of 5): 72,771 / 47,425 / 42,626 / 34,215 iterations/s for
2-5 players.

## 2026-10-20: State buffers for hands and legal moves

`HanabiState.player_hands()` made one C call per card, and `legal_moves()`
about 8 per legal move (heap copy, five field getters, delete). Two new
functions in `capi/pyhanabi.cc` (layouts in `pyhanabi.h`):

| Function | Output |
|---|---|
| `pyhanabi_state_hands` | hand sizes (P); colors, ranks (P, H), -1 past the end; every card visible |
| `pyhanabi_state_legal_move_uids` | acting player's legal move ids, ascending (the `LegalMoves` order); 0 when terminal or a card must be dealt |

- `legal_move_uids` repeats `LegalMoves`' scan over move ids without building
  the move list; unlike `StateLegalMoves` it is safe on the chance player.
- Python maps the ids to a per-game table of immutable `HanabiMove`s
  (`HanabiGame.all_moves`); `DeterminizedEnv` reads only the hand it needs
  (`HanabiState.player_hand`).
- Tests (`tests/test_capi_buffers.cc`, renamed from
  `test_observation_buffers.cc`): both buffers equal the per-card getters and
  `StateLegalMoves` along random games (2-5 players, 2 and 5 colors), no
  legal moves while a card must be dealt, wrong sizes rejected unwritten.
- `hanabi_lib` is unchanged, so `bench_engine` is unaffected.

End to end (with the MCTS replay change committed just before): MCTS 2p
4,868 -> 6,071, 3p 3,877 -> 5,091, 5p 3,193 -> 4,375 rollouts/s; MCTS
without rule filtering, 3p 4,597 -> 6,148. Observation dicts, rule outputs and
MCTS decisions (with and without rule filtering) are identical to before.

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
