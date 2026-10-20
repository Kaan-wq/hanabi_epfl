// Tests for HandSampler (hidden-hand sampling for re-determinizing MCTS).

#include <cmath>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>

#include "canonical_encoders.h"
#include "check.h"
#include "hanabi_game.h"
#include "hanabi_hand.h"
#include "hanabi_move.h"
#include "hanabi_observation.h"
#include "hanabi_state.h"
#include "hand_sampler.h"

namespace hle = hanabi_learning_env;

namespace {

hle::HanabiGame MakeGame(int players, int observation_type = hle::HanabiGame::kCardKnowledge) {
  return hle::HanabiGame(std::unordered_map<std::string, std::string>{
      {"players", std::to_string(players)},
      {"seed", "5"},
      {"observation_type", std::to_string(observation_type)}});
}

void DealPending(hle::HanabiState* state) {
  while (!state->IsTerminal() && state->CurPlayer() == hle::kChancePlayerId) {
    state->ApplyRandomChance();
  }
}

// Plays `num_moves` random legal moves (hints included, so knowledge builds up).
void PlayRandomly(hle::HanabiState* state, int num_moves, std::mt19937* rng) {
  DealPending(state);
  for (int i = 0; i < num_moves && !state->IsTerminal(); ++i) {
    const auto legal = state->LegalMoves(state->CurPlayer());
    state->ApplyMove(legal[(*rng)() % legal.size()]);
    DealPending(state);
  }
}

int CardsInGame(const hle::HanabiState& state) {
  int total = state.Deck().Size() + static_cast<int>(state.DiscardPile().size());
  for (const auto& hand : state.Hands()) total += static_cast<int>(hand.Cards().size());
  for (int height : state.Fireworks()) total += height;
  return total;
}

int FullDeckSize(const hle::HanabiGame& game) {
  int total = 0;
  for (int color = 0; color < game.NumColors(); ++color) {
    for (int rank = 0; rank < game.NumRanks(); ++rank) {
      total += game.NumberCardInstances(color, rank);
    }
  }
  return total;
}

bool FitsKnowledge(const hle::HanabiCard& card, const hle::HanabiHand::CardKnowledge& knowledge) {
  return knowledge.ColorPlausible(card.Color()) && knowledge.RankPlausible(card.Rank());
}

}  // namespace

HANABI_TEST(ResampledHandsFitKnowledgeAndCardCounts) {
  for (int players = 2; players <= 5; ++players) {
    const hle::HanabiGame game = MakeGame(players);
    std::mt19937 rng(players);
    hle::HandSampler sampler(players);
    for (int round = 0; round < 20; ++round) {
      hle::HanabiState state(&game, 0);
      PlayRandomly(&state, round, &rng);
      for (int player = 0; player < players; ++player) {
        const hle::HanabiHand before = state.Hands()[player];
        const int cur_player = state.CurPlayer();
        const int info = state.InformationTokens();
        CHECK(sampler.ResampleHand(&state, player));
        const hle::HanabiHand& after = state.Hands()[player];
        CHECK_EQ(after.Cards().size(), before.Cards().size());
        for (size_t i = 0; i < after.Cards().size(); ++i) {
          CHECK(FitsKnowledge(after.Cards()[i], before.Knowledge()[i]));
        }
        CHECK_EQ(CardsInGame(state), FullDeckSize(game));
        CHECK_EQ(state.CurPlayer(), cur_player);
        CHECK_EQ(state.InformationTokens(), info);
      }
    }
  }
}

HANABI_TEST(MinimalObservationsIgnoreHints) {
  // kMinimal players do not remember hints, so neither does sampling for them.
  const hle::HanabiGame game = MakeGame(2, hle::HanabiGame::kMinimal);
  hle::HanabiState state(&game, 0);
  DealPending(&state);
  const auto& partner_cards = state.Hands()[1].Cards();
  state.ApplyMove(
      hle::HanabiMove(hle::HanabiMove::kRevealRank, -1, 1, -1, partner_cards[0].Rank()));
  DealPending(&state);
  hle::HandSampler sampler(1);
  bool ignored_hint = false;
  for (int i = 0; i < 200 && !ignored_hint; ++i) {
    hle::HanabiState copy = state;
    CHECK(sampler.ResampleHand(&copy, 1));
    ignored_hint = !FitsKnowledge(copy.Hands()[1].Cards()[0], state.Hands()[1].Knowledge()[0]);
  }
  CHECK(ignored_hint);
}

HANABI_TEST(FirstPositionFollowsCandidateCounts) {
  // With no hints there are no restarts, so the first card is drawn with
  // probability proportional to its candidate count.
  const hle::HanabiGame game = MakeGame(3);
  hle::HanabiState state(&game, 0);
  DealPending(&state);
  const int player = 1;
  const std::vector<int> expected =
      hle::HandSampler::CandidateCounts(state, player, 0, nullptr, std::vector<hle::HanabiCard>());
  // Every card of the player's hand can be at position 0 once it is returned.
  std::vector<int> counts = expected;
  for (size_t i = 1; i < state.Hands()[player].Cards().size(); ++i) {
    const auto& card = state.Hands()[player].Cards()[i];
    ++counts[card.Color() * game.NumRanks() + card.Rank()];
  }
  int total = 0;
  for (int count : counts) total += count;

  constexpr int kSamples = 20000;
  std::vector<int> seen(counts.size(), 0);
  hle::HandSampler sampler(42);
  for (int i = 0; i < kSamples; ++i) {
    hle::HanabiState copy = state;
    CHECK(sampler.ResampleHand(&copy, player));
    const auto& card = copy.Hands()[player].Cards()[0];
    ++seen[card.Color() * game.NumRanks() + card.Rank()];
  }
  for (size_t i = 0; i < counts.size(); ++i) {
    const double p = static_cast<double>(counts[i]) / total;
    const double sigma = std::sqrt(kSamples * p * (1 - p));
    CHECK(std::abs(seen[i] - kSamples * p) <= 5 * sigma + 1);
  }
}

HANABI_TEST(SameSeedGivesSameHands) {
  const hle::HanabiGame game = MakeGame(4);
  hle::HanabiState state(&game, 0);
  std::mt19937 rng(3);
  PlayRandomly(&state, 10, &rng);
  hle::HandSampler first(9);
  hle::HandSampler second(9);
  for (int i = 0; i < 20; ++i) {
    hle::HanabiState a = state;
    hle::HanabiState b = state;
    CHECK(first.ResampleHand(&a, 2));
    CHECK(second.ResampleHand(&b, 2));
    CHECK_EQ(a.ToString(), b.ToString());
  }
}

HANABI_TEST(FailedResampleLeavesStateUnchanged) {
  const hle::HanabiGame game = MakeGame(2);
  hle::HanabiState state(&game, 0);
  DealPending(&state);
  // Knowledge that rules out every color: no card fits position 0.
  nlohmann::json j = state.toJSON();
  for (auto&& plausible : j["hands"][1]["card_knowledge"][0]["color"]["value_plausible"]) {
    plausible = false;
  }
  hle::HanabiState impossible = hle::HanabiState::fromJSON(j, &game);
  const std::string before = impossible.toJSON().dump();
  hle::HandSampler sampler(1);
  CHECK(!sampler.ResampleHand(&impossible, 1));
  CHECK(!sampler.ResampleHand(&impossible, 2));  // Not a player.
  CHECK(!sampler.RestoreHand(&impossible, -1, {}, hle::HanabiCard(), -1));
  CHECK_EQ(impossible.toJSON().dump(), before);
}

HANABI_TEST(RestoreDealsBackTheRememberedHand) {
  const hle::HanabiGame game = MakeGame(3);
  hle::HanabiState state(&game, 0);
  DealPending(&state);
  hle::HandSampler sampler(7);
  const int player = state.CurPlayer();
  const std::vector<hle::HanabiCard> remembered = state.Hands()[player].Cards();

  // The player plays position 1 of a sampled hand, then draws.
  CHECK(sampler.ResampleHand(&state, player));
  const hle::HanabiCard removed = state.Hands()[player].Cards()[1];
  state.ApplyMove(hle::HanabiMove(hle::HanabiMove::kPlay, 1, -1, -1, -1));
  DealPending(&state);

  CHECK(sampler.RestoreHand(&state, player, remembered, removed, 1));
  const auto& hand = state.Hands()[player].Cards();
  CHECK_EQ(hand.size(), remembered.size());
  std::vector<hle::HanabiCard> expected = remembered;
  expected.erase(expected.begin() + 1);
  for (size_t i = 0; i < expected.size(); ++i) {
    // A remembered copy of the discarded card may have been replaced.
    if (!(expected[i] == removed)) CHECK(hand[i] == expected[i]);
  }
  CHECK_EQ(CardsInGame(state), FullDeckSize(game));
}

HANABI_TEST(RestoreForgetsHintsWhenNothingElseFits) {
  const hle::HanabiGame game = MakeGame(2);
  hle::HanabiState state(&game, 0);
  DealPending(&state);
  // A card with every copy still in the deck.
  hle::HanabiCard target;
  for (int color = 0; color < game.NumColors() && !target.IsValid(); ++color) {
    for (int rank = 1; rank < game.NumRanks() - 1; ++rank) {  // Two copies each.
      if (state.Deck().CardCount(color, rank) == game.NumberCardInstances(color, rank)) {
        target = hle::HanabiCard(color, rank);
        break;
      }
    }
  }
  CHECK(target.IsValid());
  if (!target.IsValid()) return;

  // Player 1 "knows" position 0 is the target card.
  nlohmann::json j = state.toJSON();
  auto& knowledge = j["hands"][1]["card_knowledge"][0];
  for (int c = 0; c < game.NumColors(); ++c) {
    knowledge["color"]["value_plausible"][c] = c == target.Color();
  }
  for (int r = 0; r < game.NumRanks(); ++r) {
    knowledge["rank"]["value_plausible"][r] = r == target.Rank();
  }
  state = hle::HanabiState::fromJSON(j, &game);

  // Remembered: the target at 0 and both copies after it, so the copy for
  // position 0 is unavailable and nothing else fits its hints.
  const auto& cards = state.Hands()[1].Cards();
  const std::vector<hle::HanabiCard> remembered = {target, target, target, cards[3], cards[4]};
  hle::HandSampler sampler(3);
  CHECK(sampler.RestoreHand(&state, 1, remembered, target, 4));
  const auto& restored = state.Hands()[1];
  CHECK(!(restored.Cards()[0] == target));
  CHECK(restored.Cards()[1] == target);
  CHECK(restored.Cards()[2] == target);
  for (int c = 0; c < game.NumColors(); ++c) CHECK(restored.Knowledge()[0].ColorPlausible(c));
  CHECK_EQ(CardsInGame(state), FullDeckSize(game));
}

HANABI_TEST(RestoreRolloutsKeepCardCounts) {
  // Re-determinizes like a RESTORE rollout through whole games, so the
  // replacement and knowledge-reset paths run (under the sanitizers too).
  for (int players = 2; players <= 5; ++players) {
    const hle::HanabiGame game = MakeGame(players);
    std::mt19937 rng(100 + players);
    hle::HandSampler sampler(players);
    for (int round = 0; round < 10; ++round) {
      hle::HanabiState state(&game, 0);
      DealPending(&state);
      while (!state.IsTerminal()) {
        const int player = state.CurPlayer();
        const std::vector<hle::HanabiCard> remembered = state.Hands()[player].Cards();
        CHECK(sampler.ResampleHand(&state, player));
        const auto legal = state.LegalMoves(player);
        const hle::HanabiMove move = legal[rng() % legal.size()];
        hle::HanabiCard removed;
        if (move.MoveType() == hle::HanabiMove::kPlay ||
            move.MoveType() == hle::HanabiMove::kDiscard) {
          removed = state.Hands()[player].Cards()[move.CardIndex()];
        }
        state.ApplyMove(move);
        DealPending(&state);
        CHECK(sampler.RestoreHand(&state, player, remembered, removed, move.CardIndex()));
        CHECK_EQ(CardsInGame(state), FullDeckSize(game));
      }
    }
  }
}

HANABI_TEST(ResamplingIsInvisibleToTheResampledPlayer) {
  // A player's view of their own hand is their knowledge, so swapping their
  // cards must not change anything they observe: not the move history (no
  // swap moves are recorded), not the network encoding.
  for (int players = 2; players <= 5; ++players) {
    const hle::HanabiGame game = MakeGame(players);
    const hle::CanonicalObservationEncoder encoder(&game);
    std::mt19937 rng(players + 40);
    hle::HandSampler sampler(players);
    for (int round = 0; round < 15; ++round) {
      hle::HanabiState state(&game, 0);
      PlayRandomly(&state, round * 3, &rng);
      if (state.IsTerminal()) continue;
      const int player = state.CurPlayer();
      const std::vector<int> before = encoder.Encode(hle::HanabiObservation(state, player));
      const size_t history = state.MoveHistory().size();
      const std::vector<hle::HanabiCard> remembered = state.Hands()[player].Cards();
      CHECK(sampler.ResampleHand(&state, player));
      CHECK_EQ(state.MoveHistory().size(), history);
      CHECK(encoder.Encode(hle::HanabiObservation(state, player)) == before);
      CHECK(sampler.RestoreHand(&state, player, remembered, hle::HanabiCard(), -1));
      CHECK(state.Hands()[player].Cards() == remembered);
      CHECK_EQ(state.MoveHistory().size(), history);
      CHECK(encoder.Encode(hle::HanabiObservation(state, player)) == before);
    }
  }
}

int main() { return hanabi_test::RunAll(); }
