// Engine speed benchmark. Run in Release mode before and after any engine
// optimization, and record the numbers in engine/CHANGES.md.
//
// Usage: bench_engine [num_games] [row]
//   num_games: games per player count for the random-game rows (default 20000)
//   row:       run only the rows whose name contains this text (for profiling)
//
// The headline metric is "search iteration": one iteration of a
// re-determinizing tree search done entirely in the engine.

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>

#include "canonical_encoders.h"
#include "hanabi_game.h"
#include "hanabi_observation.h"
#include "hanabi_state.h"
#include "hand_sampler.h"
#include "observation_features.h"
#include "search/mcts.h"
#include "search/van_den_bergh.h"

namespace hle = hanabi_learning_env;

namespace {

using Clock = std::chrono::steady_clock;

struct Result {
  int64_t steps = 0;
  double seconds = 0.0;
};

double Seconds(Clock::time_point start) {
  return std::chrono::duration<double>(Clock::now() - start).count();
}

hle::HanabiGame MakeGame(int players, int seed) {
  return hle::HanabiGame(std::unordered_map<std::string, std::string>{
      {"players", std::to_string(players)}, {"seed", std::to_string(seed)}});
}

// Plays uniformly random legal games. With `encode`, also builds and encodes the
// current player's observation at every step, like an RL environment does.
Result PlayRandomGames(int players, int num_games, bool encode) {
  hle::HanabiGame game = MakeGame(players, 1);
  hle::CanonicalObservationEncoder encoder(&game);
  std::mt19937 rng(7);
  Result result;
  int64_t checksum = 0;
  const auto start = Clock::now();
  for (int g = 0; g < num_games; ++g) {
    hle::HanabiState state(&game, g % players);
    while (!state.IsTerminal()) {
      if (state.CurPlayer() == hle::kChancePlayerId) {
        state.ApplyRandomChance();
        continue;
      }
      if (encode) {
        hle::HanabiObservation observation(state, state.CurPlayer());
        checksum += encoder.Encode(observation)[0];
      }
      const auto legal = state.LegalMoves(state.CurPlayer());
      state.ApplyMove(legal[rng() % legal.size()]);
      ++result.steps;
    }
    checksum += state.Score();
  }
  result.seconds = Seconds(start);
  if (checksum == -1) std::puts("");  // Keep the work observable to the optimizer.
  return result;
}

// Collects mid-game states to measure per-state operations on realistic inputs.
std::vector<hle::HanabiState> SampleStates(const hle::HanabiGame& game, int count) {
  std::mt19937 rng(11);
  std::vector<hle::HanabiState> states;
  while (static_cast<int>(states.size()) < count) {
    hle::HanabiState state(&game, 0);
    while (!state.IsTerminal()) {
      if (state.CurPlayer() == hle::kChancePlayerId) {
        state.ApplyRandomChance();
        continue;
      }
      if (rng() % 8 == 0) states.push_back(state);
      const auto legal = state.LegalMoves(state.CurPlayer());
      state.ApplyMove(legal[rng() % legal.size()]);
    }
  }
  states.erase(states.begin() + count, states.end());
  return states;
}

Result CopyStates(const std::vector<hle::HanabiState>& states, int rounds) {
  Result result;
  int64_t checksum = 0;
  const auto start = Clock::now();
  for (int r = 0; r < rounds; ++r) {
    for (const auto& state : states) {
      hle::HanabiState copy(state);  // NOLINT(performance-unnecessary-copy-initialization)
      checksum += copy.InformationTokens();
      ++result.steps;
    }
  }
  result.seconds = Seconds(start);
  if (checksum == -1) std::puts("");
  return result;
}

Result JsonRoundTrips(const hle::HanabiGame& game, const std::vector<hle::HanabiState>& states) {
  Result result;
  int64_t checksum = 0;
  const auto start = Clock::now();
  for (const auto& state : states) {
    const std::string text = state.toJSON().dump();
    const hle::HanabiState restored =
        hle::HanabiState::fromJSON(nlohmann::json::parse(text), &game);
    checksum += restored.LifeTokens();
    ++result.steps;
  }
  result.seconds = Seconds(start);
  if (checksum == -1) std::puts("");
  return result;
}

// Re-determinizes the acting player's hand, as an MCTS rollout does before
// every other player's turn. States are copied before timing.
Result ResampleHands(const std::vector<hle::HanabiState>& states, int rounds) {
  std::vector<hle::HanabiState> copies;
  copies.reserve(states.size() * rounds);
  for (int r = 0; r < rounds; ++r) copies.insert(copies.end(), states.begin(), states.end());
  hle::HandSampler sampler(13);
  Result result;
  int64_t checksum = 0;
  const auto start = Clock::now();
  for (auto& state : copies) {
    checksum += sampler.ResampleHand(&state, state.CurPlayer()).has_value();
    ++result.steps;
  }
  result.seconds = Seconds(start);
  if (checksum == -1) std::puts("");
  return result;
}

// Builds the acting player's observation and its card-counting features, as
// a rule-based rollout policy needs them at every step.
Result ObservationFeatures(const hle::HanabiGame& game, const std::vector<hle::HanabiState>& states,
                           int rounds) {
  std::vector<int32_t> unseen(static_cast<size_t>(game.NumColors()) * game.NumRanks());
  std::vector<uint8_t> plausible(static_cast<size_t>(game.NumPlayers()) * game.HandSize() *
                                 unseen.size());
  Result result;
  int64_t checksum = 0;
  const auto start = Clock::now();
  for (int r = 0; r < rounds; ++r) {
    for (const auto& state : states) {
      const hle::HanabiObservation observation(state, state.CurPlayer());
      hle::UnseenCardCounts(observation, unseen.data());
      hle::PlausibleCards(observation, plausible.data());
      checksum += unseen[0] + plausible[0];
      ++result.steps;
    }
  }
  result.seconds = Seconds(start);
  if (checksum == -1) std::puts("");
  return result;
}

// Picks the acting player's Van den Bergh move from the state.
Result VanDenBerghMoves(const std::vector<hle::HanabiState>& states, int rounds) {
  Result result;
  int64_t checksum = 0;
  const auto start = Clock::now();
  for (int r = 0; r < rounds; ++r) {
    for (const auto& state : states) {
      checksum += hle::VanDenBerghRuleMove(state);
      ++result.steps;
    }
  }
  result.seconds = Seconds(start);
  if (checksum == -1) std::puts("");
  return result;
}

// An MCTS leaf evaluation: copy a state and play `steps` re-determinized
// Van den Bergh moves from it, as MCTSAgent.simulate does.
Result Rollouts(const std::vector<hle::HanabiState>& states, int rounds, int steps) {
  hle::HandSampler sampler(19);
  hle::SplitMix64 rng(29);
  Result result;
  int64_t checksum = 0;
  const auto start = Clock::now();
  for (int r = 0; r < rounds; ++r) {
    for (const auto& root : states) {
      hle::HanabiState state(root);
      hle::RememberedHand remembered;
      checksum += hle::PlayRollout(&state, steps, root.CurPlayer(), hle::Determinization::kRestore,
                                   &remembered, &sampler, &rng)
                      .has_value();
      checksum += hle::Reward(state, hle::RolloutReward::kScore);
      ++result.steps;
    }
  }
  result.seconds = Seconds(start);
  if (checksum == -1) std::puts("");
  return result;
}

// Pure MCTS decisions (MCTSAgent without rules): `iterations` per searched
// state, from the first `num_states` decision states. Counts iterations.
Result MctsIterations(const hle::HanabiGame& game, const std::vector<hle::HanabiState>& states,
                      int num_states, int iterations) {
  std::vector<int32_t> visits(static_cast<size_t>(game.MaxMoves()));
  const hle::MctsConfig config{.iterations = iterations};
  Result result;
  int64_t checksum = 0;
  uint64_t seed = 31;
  const auto start = Clock::now();
  for (const auto& state : states) {
    if (result.steps >= int64_t{num_states} * iterations) break;
    if (state.IsTerminal() || state.CurPlayer() == hle::kChancePlayerId) continue;
    checksum += hle::MctsSearch(state, config, seed++, visits).value_or(-1);
    result.steps += iterations;
  }
  result.seconds = Seconds(start);
  if (checksum == -1) std::puts("");
  return result;
}

// One iteration of a re-determinizing tree search, as a C++ search would run
// it: copy a root state, resample the searcher's hand, replay a path of up to
// `depth` moves (listing legal moves, applying one, dealing, restoring the
// previous player's hand and resampling the next player's), then encode the
// leaf observation into a network input buffer.
Result SearchIterations(const hle::HanabiGame& game, const std::vector<hle::HanabiState>& roots,
                        int rounds, int depth) {
  const hle::CanonicalObservationEncoder encoder(&game);
  std::vector<uint8_t> bits(encoder.Length());
  hle::HandSampler sampler(17);
  std::mt19937 rng(23);
  std::vector<hle::HanabiCard> remembered;
  Result result;
  int64_t checksum = 0;
  const auto start = Clock::now();
  for (int r = 0; r < rounds; ++r) {
    for (const auto& root : roots) {
      hle::HanabiState state(root);
      const int searcher = state.CurPlayer();
      checksum += sampler.ResampleHand(&state, searcher).has_value();
      for (int step = 0; step < depth && !state.IsTerminal(); ++step) {
        const int player = state.CurPlayer();
        const auto legal = state.LegalMoves(player);
        const hle::HanabiMove move = legal[rng() % legal.size()];
        hle::HanabiCard removed;
        if (move.MoveType() == hle::HanabiMove::kPlay ||
            move.MoveType() == hle::HanabiMove::kDiscard) {
          removed = state.Hands()[player].Cards()[move.CardIndex()];
        }
        state.ApplyMove(move);
        while (!state.IsTerminal() && state.CurPlayer() == hle::kChancePlayerId) {
          state.ApplyRandomChance();
        }
        if (player != searcher) {
          checksum += sampler.RestoreHand(&state, player, remembered, removed, move.CardIndex())
                          .has_value();
        }
        const int next = state.CurPlayer();
        if (!state.IsTerminal() && next != searcher) {
          const auto& cards = state.Hands()[next].Cards();
          remembered.assign(cards.begin(), cards.end());
          checksum += sampler.ResampleHand(&state, next).has_value();
        }
      }
      const int observer = state.IsTerminal() ? searcher : state.CurPlayer();
      const hle::HanabiObservation observation(state, observer);
      encoder.EncodeInto(observation, bits.data());
      checksum += bits[0];
      ++result.steps;
    }
  }
  result.seconds = Seconds(start);
  if (checksum == -1) std::puts("");
  return result;
}

void Report(const char* name, int players, const Result& result, const char* unit) {
  std::printf("%-22s %d players  %12.0f %s/s  (%lld in %.2fs)\n", name, players,
              static_cast<double>(result.steps) / result.seconds, unit,
              static_cast<long long>(result.steps), result.seconds);
}

}  // namespace

int main(int argc, char** argv) {  // NOLINT(bugprone-exception-escape): a tool may abort
  const int num_games = argc > 1 ? std::atoi(argv[1]) : 20000;
  if (num_games <= 0) {
    std::fprintf(stderr, "usage: bench_engine [num_games > 0] [row]\n");
    return 2;
  }
  const std::string only = argc > 2 ? argv[2] : "";
  const auto run = [&](const char* name) {
    return name == only || std::string(name).find(only) != std::string::npos;
  };
  for (int players = 2; players <= 5; ++players) {
    if (run("search iteration")) {
      const hle::HanabiGame game = MakeGame(players, 3);
      const auto roots = SampleStates(game, 2000);
      Report("search iteration", players, SearchIterations(game, roots, 25, 10), "iters");
    }
    if (run("random games")) {
      Report("random games", players, PlayRandomGames(players, num_games, false), "steps");
    }
    if (run("random games + encode")) {
      Report("random games + encode", players, PlayRandomGames(players, num_games, true), "steps");
    }
    const hle::HanabiGame game = MakeGame(players, 3);
    const auto states = SampleStates(game, 2000);
    if (run("state copy")) Report("state copy", players, CopyStates(states, 50), "copies");
    if (run("state JSON round-trip")) {
      Report("state JSON round-trip", players, JsonRoundTrips(game, states), "trips");
    }
    if (run("resample hand")) Report("resample hand", players, ResampleHands(states, 10), "hands");
    if (run("observation features")) {
      Report("observation features", players, ObservationFeatures(game, states, 10), "obs");
    }
    if (run("van den bergh move")) {
      Report("van den bergh move", players, VanDenBerghMoves(states, 100), "moves");
    }
    if (run("rollout (3 moves)")) {
      Report("rollout (3 moves)", players, Rollouts(states, 25, 3), "rollouts");
    }
    if (run("mcts (1000 iterations)")) {
      Report("mcts (1000 iterations)", players, MctsIterations(game, states, 200, 1000), "iters");
    }
  }
  return 0;
}
