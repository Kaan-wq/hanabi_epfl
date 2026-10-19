// Tests for functionality our fork adds on top of upstream HLE: return and
// deal-specific moves (used for MCTS re-determinization), knowledge removal,
// and JSON serialization. Upstream behavior is covered by the parity test.

#include <string>
#include <unordered_map>
#include <vector>

#include "check.h"
#include "hanabi_game.h"
#include "hanabi_hand.h"
#include "hanabi_move.h"
#include "hanabi_state.h"

namespace hle = hanabi_learning_env;

namespace {

hle::HanabiGame MakeGame(int players, int seed) {
  return hle::HanabiGame(std::unordered_map<std::string, std::string>{
      {"players", std::to_string(players)}, {"seed", std::to_string(seed)}});
}

// Deals initial hands, then plays `num_moves` deterministic non-chance moves,
// never playing an unplayable card so the game does not end early.
void Advance(hle::HanabiState* state, int num_moves) {
  int played = 0;
  while (!state->IsTerminal() && played < num_moves) {
    if (state->CurPlayer() == hle::kChancePlayerId) {
      state->ApplyRandomChance();
      continue;
    }
    const auto& cards = state->Hands()[state->CurPlayer()].Cards();
    std::vector<hle::HanabiMove> safe;
    for (const auto& move : state->LegalMoves(state->CurPlayer())) {
      if (move.MoveType() != hle::HanabiMove::kPlay ||
          state->CardPlayableOnFireworks(cards[move.CardIndex()].Color(),
                                         cards[move.CardIndex()].Rank())) {
        safe.push_back(move);
      }
    }
    state->ApplyMove(safe[static_cast<size_t>(played * 7) % safe.size()]);
    ++played;
  }
  while (!state->IsTerminal() && state->CurPlayer() == hle::kChancePlayerId) {
    state->ApplyRandomChance();
  }
}

std::vector<std::string> LegalMoveStrings(const hle::HanabiState& state) {
  std::vector<std::string> moves;
  for (const auto& move : state.LegalMoves(state.CurPlayer())) moves.push_back(move.ToString());
  return moves;
}

}  // namespace

HANABI_TEST(ReturnThenDealSpecificReplacesOneCard) {
  const hle::HanabiGame game = MakeGame(3, 11);
  hle::HanabiState state(&game, 0);
  Advance(&state, 10);
  CHECK(!state.IsTerminal());
  if (state.IsTerminal()) return;

  const int player = state.CurPlayer();
  const auto hand_before = state.Hands()[player];
  const hle::HanabiCard old_card = hand_before.Cards()[0];
  const int deck_before = state.Deck().Size();
  const int info_before = state.InformationTokens();
  const int life_before = state.LifeTokens();

  // Pick a replacement card that is still in the deck and differs from the old one.
  int new_color = -1;
  int new_rank = -1;
  for (int c = 0; c < game.NumColors() && new_color < 0; ++c) {
    for (int r = 0; r < game.NumRanks(); ++r) {
      if (state.Deck().CardCount(c, r) > 0 && !(c == old_card.Color() && r == old_card.Rank())) {
        new_color = c;
        new_rank = r;
        break;
      }
    }
  }
  CHECK(new_color >= 0);

  state.ApplyMove(hle::HanabiMove(hle::HanabiMove::kReturn, 0, static_cast<int8_t>(player), -1, -1));
  CHECK_EQ(state.CurPlayer(), hle::kChancePlayerId);
  CHECK_EQ(state.Hands()[player].Cards().size(), hand_before.Cards().size() - 1);
  CHECK_EQ(state.Deck().Size(), deck_before + 1);

  state.ApplyMove(hle::HanabiMove(hle::HanabiMove::kDealSpecific, 0, static_cast<int8_t>(player),
                                  static_cast<int8_t>(new_color), static_cast<int8_t>(new_rank)));
  CHECK_EQ(state.CurPlayer(), player);
  CHECK_EQ(state.Deck().Size(), deck_before);
  CHECK_EQ(state.InformationTokens(), info_before);
  CHECK_EQ(state.LifeTokens(), life_before);

  const auto& hand_after = state.Hands()[player];
  CHECK_EQ(hand_after.Cards().size(), hand_before.Cards().size());
  CHECK_EQ(hand_after.Cards()[0].Color(), new_color);
  CHECK_EQ(hand_after.Cards()[0].Rank(), new_rank);
  for (size_t i = 1; i < hand_after.Cards().size(); ++i) {
    CHECK(hand_after.Cards()[i] == hand_before.Cards()[i]);
  }
  // Knowledge is kept: the player's beliefs about the card must not change.
  for (size_t i = 0; i < hand_after.Knowledge().size(); ++i) {
    CHECK_EQ(hand_after.Knowledge()[i].ToString(), hand_before.Knowledge()[i].ToString());
  }
}

HANABI_TEST(RemoveKnowledgeResetsOnlyThatCard) {
  const hle::HanabiGame game = MakeGame(2, 3);
  hle::HanabiState state(&game, 0);
  Advance(&state, 0);

  const int hinter = state.CurPlayer();
  const int target = (hinter + 1) % game.NumPlayers();
  const int color = state.Hands()[target].Cards()[0].Color();
  state.ApplyMove(hle::HanabiMove(hle::HanabiMove::kRevealColor, -1, 1,
                                  static_cast<int8_t>(color), -1));
  const auto knowledge_before = state.Hands()[target].Knowledge();
  CHECK(knowledge_before[0].ColorHinted());

  state.RemoveKnowledge(target, 0);
  const auto& knowledge_after = state.Hands()[target].Knowledge();
  const hle::HanabiHand::CardKnowledge fresh(game.NumColors(), game.NumRanks());
  CHECK_EQ(knowledge_after[0].ToString(), fresh.ToString());
  for (size_t i = 1; i < knowledge_after.size(); ++i) {
    CHECK_EQ(knowledge_after[i].ToString(), knowledge_before[i].ToString());
  }
}

HANABI_TEST(StateJsonRoundTripContinuesIdentically) {
  const hle::HanabiGame game = MakeGame(4, 5);
  for (int moves : {0, 5, 25, 60}) {
    hle::HanabiState original(&game, 1);
    Advance(&original, moves);

    const nlohmann::json serialized = original.toJSON();
    hle::HanabiState restored = hle::HanabiState::fromJSON(serialized, &game);
    CHECK_EQ(restored.ToString(), original.ToString());
    CHECK(restored.toJSON() == serialized);
    CHECK_EQ(restored.MoveHistory().size(), original.MoveHistory().size());
    if (!original.IsTerminal()) CHECK(LegalMoveStrings(restored) == LegalMoveStrings(original));

    // Play both to the end with the same moves (deals copied from the original).
    while (!original.IsTerminal()) {
      if (original.CurPlayer() == hle::kChancePlayerId) {
        original.ApplyRandomChance();
        restored.ApplyMove(original.MoveHistory().back().move);
      } else {
        const auto legal = original.LegalMoves(original.CurPlayer());
        original.ApplyMove(legal.front());
        restored.ApplyMove(legal.front());
      }
    }
    CHECK(restored.IsTerminal());
    CHECK_EQ(restored.Score(), original.Score());
    CHECK_EQ(restored.ToString(), original.ToString());
  }
}

HANABI_TEST(GameJsonRoundTripKeepsParametersAndRng) {
  hle::HanabiGame game = MakeGame(3, 42);
  hle::HanabiState warmup(&game, 0);
  Advance(&warmup, 3);  // Move the game's RNG away from its seeded state.

  const nlohmann::json serialized = game.toJSON();
  hle::HanabiGame restored = hle::HanabiGame::fromJSON(serialized);
  CHECK(restored.toJSON() == serialized);
  CHECK(restored.Parameters() == game.Parameters());

  // Same RNG state: dealing a fresh game from each must produce identical hands.
  hle::HanabiState from_original(&game, 0);
  hle::HanabiState from_restored(&restored, 0);
  Advance(&from_original, 0);
  Advance(&from_restored, 0);
  CHECK_EQ(from_restored.ToString(), from_original.ToString());
}

HANABI_TEST(MoveJsonRoundTripAllTypes) {
  const std::vector<hle::HanabiMove> moves = {
      hle::HanabiMove(hle::HanabiMove::kPlay, 2, -1, -1, -1),
      hle::HanabiMove(hle::HanabiMove::kDiscard, 0, -1, -1, -1),
      hle::HanabiMove(hle::HanabiMove::kRevealColor, -1, 1, 3, -1),
      hle::HanabiMove(hle::HanabiMove::kRevealRank, -1, 2, -1, 4),
      hle::HanabiMove(hle::HanabiMove::kDeal, -1, -1, 1, 0),
      hle::HanabiMove(hle::HanabiMove::kReturn, 1, 2, -1, -1),
      hle::HanabiMove(hle::HanabiMove::kDealSpecific, 1, 2, 4, 3),
  };
  for (const auto& move : moves) {
    const hle::HanabiMove restored = hle::HanabiMove::fromJSON(move.toJSON());
    CHECK(restored == move);
    CHECK_EQ(restored.ToString(), move.ToString());
  }
}

int main() { return hanabi_test::RunAll(); }
