// Tests that the engine's optimized code paths match their reference
// definitions, over varied games (players, colors, ranks, hand sizes, token
// limits) and whole random playouts. Each optimization that replaces a
// straightforward computation adds its check here.

#include <random>
#include <string>
#include <unordered_map>
#include <vector>

#include "check.h"
#include "hanabi_game.h"
#include "hanabi_move.h"
#include "hanabi_state.h"

namespace hle = hanabi_learning_env;

namespace {

struct Variant {
  int players, colors, ranks, hand_size, max_information_tokens;
};

// Default games for every player count, plus reduced ones with edge limits.
std::vector<Variant> Variants() {
  std::vector<Variant> variants;
  for (int players = 2; players <= 5; ++players) variants.push_back({players, 5, 5, -1, 8});
  variants.push_back({2, 1, 1, 1, 0});
  variants.push_back({3, 2, 3, 2, 1});
  variants.push_back({4, 3, 5, 3, 2});
  variants.push_back({5, 5, 2, 1, 8});
  variants.push_back({2, 5, 5, 8, 3});
  return variants;
}

hle::HanabiGame MakeGame(const Variant& v, int seed) {
  std::unordered_map<std::string, std::string> params = {
      {"players", std::to_string(v.players)},
      {"colors", std::to_string(v.colors)},
      {"ranks", std::to_string(v.ranks)},
      {"max_information_tokens", std::to_string(v.max_information_tokens)},
      {"seed", std::to_string(seed)}};
  if (v.hand_size > 0) params["hand_size"] = std::to_string(v.hand_size);
  return hle::HanabiGame(params);
}

// Calls `check(state)` at every state of `games` random playouts per variant.
template <typename Check>
void ForEachState(int games, const Check& check) {
  for (const Variant& variant : Variants()) {
    const hle::HanabiGame game = MakeGame(variant, 3);
    std::mt19937 rng(variant.players * 100 + variant.colors * 10 + variant.ranks);
    for (int g = 0; g < games; ++g) {
      hle::HanabiState state(&game, -1);
      while (!state.IsTerminal()) {
        check(state);
        if (state.CurPlayer() == hle::kChancePlayerId) {
          state.ApplyRandomChance();
          continue;
        }
        const auto legal = state.LegalMoves(state.CurPlayer());
        state.ApplyMove(legal[rng() % legal.size()]);
      }
      check(state);
    }
  }
}

}  // namespace

HANABI_TEST(LegalMovesAreTheMoveIdsMoveIsLegalAccepts) {
  ForEachState(30, [](const hle::HanabiState& state) {
    const hle::HanabiGame& game = *state.ParentGame();
    for (int player = 0; player < game.NumPlayers(); ++player) {
      std::vector<hle::HanabiMove> expected;
      if (player == state.CurPlayer()) {
        for (int uid = 0; uid < game.MaxMoves(); ++uid) {
          if (state.MoveIsLegal(game.GetMove(uid))) expected.push_back(game.GetMove(uid));
        }
      }
      CHECK(state.LegalMoves(player) == expected);
    }
  });
}

HANABI_TEST(RandomChanceDealsWhatTheChanceOutcomeListWouldDeal) {
  // Two copies of each game (same seed) played in lockstep: one deals with
  // ApplyRandomChance, the other picks from the full ChanceOutcomes() list.
  for (const Variant& variant : Variants()) {
    const hle::HanabiGame fast_game = MakeGame(variant, 9);
    const hle::HanabiGame reference_game = MakeGame(variant, 9);
    std::mt19937 rng(variant.players + variant.hand_size);
    for (int g = 0; g < 30; ++g) {
      hle::HanabiState fast(&fast_game, 0);
      hle::HanabiState reference(&reference_game, 0);
      while (!reference.IsTerminal()) {
        if (reference.CurPlayer() == hle::kChancePlayerId) {
          fast.ApplyRandomChance();
          reference.ApplyMove(reference_game.PickRandomChance(reference.ChanceOutcomes()));
        } else {
          const auto legal = reference.LegalMoves(reference.CurPlayer());
          const hle::HanabiMove move = legal[rng() % legal.size()];
          fast.ApplyMove(move);
          reference.ApplyMove(move);
        }
        CHECK_EQ(fast.ToString(), reference.ToString());
      }
      CHECK(fast.IsTerminal());
    }
  }
}

int main() { return hanabi_test::RunAll(); }
