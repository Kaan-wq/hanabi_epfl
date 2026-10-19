# Parity tests

`parity_test` replays games recorded from **unmodified upstream HLE** and checks
that our engine matches at every step. The golden file stores every move,
including which card each deal produced, plus FNV-1a digests of:

- the state: `ToString()`, current player, score, tokens, deck size, end status
  and the current player's legal moves;
- every player's observation string;
- every player's canonical observation encoding.

Deals are replayed from the file rather than re-drawn from the seeded RNG,
because `std::discrete_distribution` differs between standard libraries. The
test is therefore platform-independent.

Recorded variants (see `GoldenVariants()` in `parity_trace.h`): 2–5 players,
each with the full game under all three observation types and a reduced game
(2 colors, hand size 2, 3 information tokens, 1 life). The recording policy plays
sensibly enough that games reach the end-game, perfect scores and life-loss
endings.

## Re-recording the golden data

Needed if `GoldenVariants()` or the digest definition changes. Record from
upstream, never from our engine, so the reference stays independent of our
changes.

```bash
git clone https://github.com/deepmind/hanabi-learning-environment.git /tmp/hle
git -C /tmp/hle checkout 5df6a73      # commit named in engine/UPSTREAM.md
HLE=/tmp/hle/hanabi_learning_environment/hanabi_lib
JSON=<path to nlohmann/json single_include>
c++ -std=c++17 -O2 -I$HLE -Iengine/tests/parity -I$JSON \
    engine/tests/parity/record_golden.cc $HLE/*.cc -o /tmp/record_golden
/tmp/record_golden 5df6a73 engine/tests/parity/golden_upstream_5df6a73.json
```

If an engine change intentionally alters behavior or observations, update the
golden data in the same change (recorded from our engine, saved under a new
file name that says so) and explain the difference in `engine/CHANGES.md`.
