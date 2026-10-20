// Engine speed benchmark. Run in Release mode before and after any engine
// optimization, and record the numbers in engine/CHANGES.md.
//
// Usage: bench_engine [num_games]   (default 20000 games per player count)

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
    checksum += sampler.ResampleHand(&state, state.CurPlayer());
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

void Report(const char* name, int players, const Result& result, const char* unit) {
  std::printf("%-22s %d players  %12.0f %s/s  (%lld in %.2fs)\n", name, players,
              static_cast<double>(result.steps) / result.seconds, unit,
              static_cast<long long>(result.steps), result.seconds);
}

}  // namespace

int main(int argc, char** argv) {  // NOLINT(bugprone-exception-escape): a tool may abort
  const int num_games = argc > 1 ? std::atoi(argv[1]) : 20000;
  if (num_games <= 0) {
    std::fprintf(stderr, "usage: bench_engine [num_games > 0]\n");
    return 2;
  }
  for (int players = 2; players <= 5; ++players) {
    Report("random games", players, PlayRandomGames(players, num_games, false), "steps");
    Report("random games + encode", players, PlayRandomGames(players, num_games, true), "steps");
    const hle::HanabiGame game = MakeGame(players, 3);
    const auto states = SampleStates(game, 2000);
    Report("state copy", players, CopyStates(states, 50), "copies");
    Report("state JSON round-trip", players, JsonRoundTrips(game, states), "trips");
    Report("resample hand", players, ResampleHands(states, 10), "hands");
    Report("observation features", players, ObservationFeatures(game, states, 10), "obs");
  }
  return 0;
}
