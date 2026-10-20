// Tests for the re-determinizing MCTS (engine/search/mcts.h).

#include <algorithm>
#include <cstdint>
#include <numeric>
#include <string>
#include <unordered_map>
#include <vector>

#include "check.h"
#include "hanabi_game.h"
#include "hanabi_state.h"
#include "search/mcts.h"

namespace hle = hanabi_learning_env;

namespace {

hle::HanabiGame MakeGame(int players) {
  return hle::HanabiGame(std::unordered_map<std::string, std::string>{
      {"players", std::to_string(players)}, {"seed", "7"}});
}

void DealPending(hle::HanabiState* state) {
  while (!state->IsTerminal() && state->CurPlayer() == hle::kChancePlayerId) {
    state->ApplyRandomChance();
  }
}

}  // namespace

// Searches play whole games with legal moves, for 2-5 players and every
// determinization; every iteration after the first visits one root move.
HANABI_TEST(SearchesPlayGamesWithLegalMoves) {
  for (int players = 2; players <= 5; ++players) {
    const hle::HanabiGame game = MakeGame(players);
    for (int determinization = 0; determinization < 3; ++determinization) {
      const hle::MctsConfig config{
          .iterations = 120,  // Enough to visit every root move twice.
          .determinization = static_cast<hle::Determinization>(determinization)};
      std::vector<int32_t> visits(static_cast<size_t>(game.MaxMoves()));
      hle::HanabiState state(&game, 0);
      DealPending(&state);
      uint64_t seed = 1;
      while (!state.IsTerminal()) {
        const auto move = hle::MctsSearch(state, config, seed++, visits);
        CHECK(move.has_value());
        CHECK(state.MoveIsLegal(game.GetMove(*move)));
        CHECK(visits[static_cast<size_t>(*move)] > 1);
        CHECK_EQ(std::accumulate(visits.begin(), visits.end(), 0), config.iterations - 1);
        for (int uid = 0; uid < game.MaxMoves(); ++uid) {
          if (visits[static_cast<size_t>(uid)] > 0) CHECK(state.MoveIsLegal(game.GetMove(uid)));
        }
        state.ApplyMove(game.GetMove(*move));
        DealPending(&state);
      }
    }
  }
}

// The searched state is left as it was.
HANABI_TEST(SearchLeavesTheRootUnchanged) {
  const hle::HanabiGame game = MakeGame(3);
  hle::HanabiState state(&game, 0);
  DealPending(&state);
  const std::string before = state.ToString();
  std::vector<int32_t> visits(static_cast<size_t>(game.MaxMoves()));
  CHECK(hle::MctsSearch(state, hle::MctsConfig{.iterations = 200}, 5, visits).has_value());
  CHECK_EQ(state.ToString(), before);
}

// With too few iterations to visit any root move twice, the first one wins.
HANABI_TEST(UnderVisitedRootPicksTheFirstMove) {
  const hle::HanabiGame game = MakeGame(2);
  hle::HanabiState state(&game, 0);
  DealPending(&state);
  std::vector<int32_t> visits(static_cast<size_t>(game.MaxMoves()));
  const auto move = hle::MctsSearch(state, hle::MctsConfig{.iterations = 3}, 1, visits);
  CHECK(move.has_value());
  const auto legal = state.LegalMoves(state.CurPlayer());
  int first = game.MaxMoves();
  for (const auto& legal_move : legal) first = std::min(first, game.GetMoveUid(legal_move));
  CHECK_EQ(*move, first);
}

int main() { return hanabi_test::RunAll(); }
