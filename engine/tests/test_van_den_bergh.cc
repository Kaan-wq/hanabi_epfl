// Tests for the Van den Bergh policy and rollouts (engine/search/van_den_bergh.h).
// Parity with the Python rules is tested in tests/unit/agents/.

#include <cstddef>
#include <cstdint>
#include <numeric>
#include <random>
#include <string>
#include <unordered_map>

#include "check.h"
#include "hanabi_game.h"
#include "hanabi_observation.h"
#include "hanabi_state.h"
#include "hand_sampler.h"
#include "search/van_den_bergh.h"

namespace hle = hanabi_learning_env;

namespace {

hle::HanabiGame MakeGame(int players, int observation_type) {
  return hle::HanabiGame(std::unordered_map<std::string, std::string>{
      {"players", std::to_string(players)},
      {"seed", "7"},
      {"observation_type", std::to_string(observation_type)}});
}

void DealPending(hle::HanabiState* state) {
  while (!state->IsTerminal() && state->CurPlayer() == hle::kChancePlayerId) {
    state->ApplyRandomChance();
  }
}

// Situations the random games below must reach, so the rules' edge cases are exercised.
struct Coverage {
  int no_hint_tokens = 0;
  int last_life = 0;
  int empty_deck = 0;
  int rank_lost = 0;  // Every copy of some card discarded.
  int no_rule = 0;    // The policy fell back to a random move.
};

bool SomeRankLost(const hle::HanabiState& state) {
  const hle::HanabiGame& game = *state.ParentGame();
  std::unordered_map<int, int> discarded;
  for (const auto& card : state.DiscardPile()) {
    const int index = card.Color() * game.NumRanks() + card.Rank();
    if (++discarded[index] == game.NumberCardInstances(card.Color(), card.Rank())) return true;
  }
  return false;
}

// Checks the policy at the current (decision) state, and records which situations it saw.
void CheckDecision(const hle::HanabiState& state, Coverage* coverage) {
  const int player = state.CurPlayer();
  const int from_state = hle::VanDenBerghRuleMove(state);
  const hle::HanabiObservation observation(state, player);
  CHECK_EQ(from_state, hle::VanDenBerghRuleMove(observation));
  if (from_state >= 0) {
    CHECK(state.MoveIsLegal(state.ParentGame()->GetMove(from_state)));
  } else {
    ++coverage->no_rule;
  }
  // Partners' observations are not their turn.
  const int partner = (player + 1) % state.ParentGame()->NumPlayers();
  CHECK_EQ(hle::VanDenBerghRuleMove(hle::HanabiObservation(state, partner)), -1);

  coverage->no_hint_tokens += state.InformationTokens() == 0;
  coverage->last_life += state.LifeTokens() == 1;
  coverage->empty_deck += state.Deck().Empty();
  coverage->rank_lost += SomeRankLost(state);
}

}  // namespace

// The state and observation entry points agree, pick legal moves, and every
// rule edge case comes up, for 2-5 players and every observation type.
HANABI_TEST(StateAndObservationAgreeAndMovesAreLegal) {
  for (int players = 2; players <= 5; ++players) {
    for (int type :
         {hle::HanabiGame::kMinimal, hle::HanabiGame::kCardKnowledge, hle::HanabiGame::kSeer}) {
      const hle::HanabiGame game = MakeGame(players, type);
      std::mt19937 rng(static_cast<uint32_t>(players * 10 + type));
      hle::SplitMix64 policy_rng(static_cast<uint64_t>(players));
      Coverage coverage;
      for (int episode = 0; episode < 60; ++episode) {
        hle::HanabiState state(&game, 0);
        DealPending(&state);
        while (!state.IsTerminal()) {
          CheckDecision(state, &coverage);
          // Mix in random moves so games reach states the policy alone avoids:
          // many in odd episodes (lost lives), few in even ones (long games).
          const auto legal = state.LegalMoves(state.CurPlayer());
          const uint32_t random_one_in = episode % 2 == 0 ? 20 : 3;
          const hle::HanabiMove move = rng() % random_one_in == 0
                                           ? legal[rng() % legal.size()]
                                           : game.GetMove(hle::VanDenBerghMove(state, &policy_rng));
          CHECK(state.MoveIsLegal(move));
          state.ApplyMove(move);
          DealPending(&state);
        }
      }
      CHECK(coverage.no_hint_tokens > 0);
      CHECK(coverage.last_life > 0);
      CHECK(coverage.empty_deck > 0);
      CHECK(coverage.rank_lost > 0);
    }
  }
}

// Rollouts end the game when given enough steps, under every determinization.
HANABI_TEST(RolloutsFinishGames) {
  for (int players = 2; players <= 5; ++players) {
    const hle::HanabiGame game = MakeGame(players, hle::HanabiGame::kCardKnowledge);
    for (int determinization = 0; determinization < 3; ++determinization) {
      for (uint64_t seed = 1; seed <= 20; ++seed) {
        hle::HanabiState state(&game, 0);
        DealPending(&state);
        hle::RememberedHand remembered;
        hle::HandSampler sampler(seed);
        hle::SplitMix64 rng(seed);
        CHECK(hle::PlayRollout(&state, 1000, 0, static_cast<hle::Determinization>(determinization),
                               &remembered, &sampler, &rng));
        CHECK(state.IsTerminal());
        const int progress = hle::Reward(state, hle::RolloutReward::kProgress);
        CHECK(progress >= hle::Reward(state, hle::RolloutReward::kScore));
        CHECK_EQ(progress, std::accumulate(state.Fireworks().begin(), state.Fireworks().end(), 0));
      }
    }
  }
}

HANABI_TEST(RolloutsStopAfterMaxSteps) {
  const hle::HanabiGame game = MakeGame(3, hle::HanabiGame::kCardKnowledge);
  hle::HanabiState state(&game, 0);
  DealPending(&state);
  hle::RememberedHand remembered;
  hle::HandSampler sampler(3);
  hle::SplitMix64 rng(3);
  CHECK(
      hle::PlayRollout(&state, 0, 0, hle::Determinization::kRestore, &remembered, &sampler, &rng));
  CHECK_EQ(state.MoveHistory().size(), size_t{3 * 5});  // Only the initial deals.
  CHECK(
      hle::PlayRollout(&state, 4, 0, hle::Determinization::kRestore, &remembered, &sampler, &rng));
  CHECK(!state.IsTerminal());
  int moves = 0;
  for (const auto& item : state.MoveHistory()) moves += item.player != hle::kChancePlayerId;
  CHECK_EQ(moves, 4);
}

int main() { return hanabi_test::RunAll(); }
