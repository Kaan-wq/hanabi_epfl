#include "hand_sampler.h"

#include <cstdint>
#include <numeric>
#include <vector>

#include "hanabi_game.h"
#include "hanabi_move.h"

namespace hanabi_learning_env {

namespace {

// Bounds the restarts of ResampleHand; a consistent hand normally takes one or two.
constexpr int kMaxAttempts = 10000;

int CardIndex(const HanabiGame& game, int color, int rank) {
  return color * game.NumRanks() + rank;
}

// Removes one copy of `card` if any is left.
void RemoveOne(const HanabiGame& game, const HanabiCard& card, std::vector<int>* counts) {
  int& count = (*counts)[CardIndex(game, card.Color(), card.Rank())];
  if (count > 0) --count;
}

// Zeroes the cards `knowledge` rules out; returns the remaining total.
int FilterByKnowledge(const HanabiGame& game, const HanabiHand::CardKnowledge* knowledge,
                      std::vector<int>* counts) {
  int total = 0;
  for (int color = 0; color < game.NumColors(); ++color) {
    for (int rank = 0; rank < game.NumRanks(); ++rank) {
      int& count = (*counts)[CardIndex(game, color, rank)];
      if (knowledge != nullptr &&
          !(knowledge->ColorPlausible(color) && knowledge->RankPlausible(rank))) {
        count = 0;
      }
      total += count;
    }
  }
  return total;
}

// What `player` knows about each card in their hand, as their observation
// shows it: with kMinimal observations, hints are not remembered.
std::vector<HanabiHand::CardKnowledge> KnowledgeOf(const HanabiState& state, int player) {
  const HanabiGame& game = *state.ParentGame();
  const HanabiHand& hand = state.Hands()[player];
  if (game.ObservationType() == HanabiGame::kMinimal) {
    return std::vector<HanabiHand::CardKnowledge>(
        hand.Cards().size(), HanabiHand::CardKnowledge(game.NumColors(), game.NumRanks()));
  }
  return {hand.Knowledge().begin(), hand.Knowledge().begin() + hand.Cards().size()};
}

void ReturnHand(HanabiState* state, int player) {
  const auto num_cards = state->Hands()[player].Cards().size();
  for (size_t i = 0; i < num_cards; ++i) {
    // Cards shift left after each return, so always return position 0.
    state->ApplyMove(HanabiMove(HanabiMove::kReturn, 0, player, -1, -1));
  }
}

bool DealSpecific(HanabiState* state, int card_index, int player, int color, int rank) {
  const HanabiMove move(HanabiMove::kDealSpecific, card_index, player, color, rank);
  if (!state->MoveIsLegal(move)) return false;
  state->ApplyMove(move);
  return true;
}

bool IsPlayer(const HanabiState& state, int player) {
  return player >= 0 && player < state.ParentGame()->NumPlayers();
}

}  // namespace

std::vector<int> HandSampler::CandidateCounts(const HanabiState& state, int player, int card_index,
                                              const HanabiHand::CardKnowledge* knowledge,
                                              const std::vector<HanabiCard>& excluded) {
  const HanabiGame& game = *state.ParentGame();
  std::vector<int> counts(static_cast<size_t>(game.NumColors()) * game.NumRanks());
  for (int color = 0; color < game.NumColors(); ++color) {
    for (int rank = 0; rank < game.NumRanks(); ++rank) {
      counts[CardIndex(game, color, rank)] = state.Deck().CardCount(color, rank);
    }
  }
  const auto& cards = state.Hands()[player].Cards();
  if (card_index >= 0 && card_index < static_cast<int>(cards.size())) {
    ++counts[CardIndex(game, cards[card_index].Color(), cards[card_index].Rank())];
  }
  for (const HanabiCard& card : excluded) RemoveOne(game, card, &counts);
  FilterByKnowledge(game, knowledge, &counts);
  return counts;
}

bool HandSampler::ResampleHand(HanabiState* state, int player) {
  if (!IsPlayer(*state, player)) return false;
  HanabiState sampled_state = *state;  // Committed only on success.
  const HanabiGame& game = *state->ParentGame();
  const std::vector<HanabiHand::CardKnowledge> knowledge = KnowledgeOf(*state, player);
  const int hand_size = static_cast<int>(knowledge.size());
  ReturnHand(&sampled_state, player);
  const std::vector<int> available =
      CandidateCounts(sampled_state, player, -1, nullptr, std::vector<HanabiCard>());

  std::vector<int> hand;  // Card indices, by position.
  for (int attempt = 0; attempt < kMaxAttempts && static_cast<int>(hand.size()) < hand_size;
       ++attempt) {
    hand.clear();
    for (int position = 0; position < hand_size; ++position) {
      std::vector<int> counts = available;
      for (int card : hand) {
        if (counts[card] > 0) --counts[card];
      }
      const int total = FilterByKnowledge(game, &knowledge[position], &counts);
      if (total == 0) break;
      hand.push_back(SampleIndex(counts, total));
    }
  }
  if (static_cast<int>(hand.size()) < hand_size) return false;

  for (int position = 0; position < hand_size; ++position) {
    if (!DealSpecific(&sampled_state, position, player, hand[position] / game.NumRanks(),
                      hand[position] % game.NumRanks())) {
      return false;
    }
  }
  *state = sampled_state;
  return true;
}

bool HandSampler::RestoreHand(HanabiState* state, int player,
                              const std::vector<HanabiCard>& remembered,
                              const HanabiCard& removed_card, int removed_index) {
  if (!IsPlayer(*state, player)) return false;
  HanabiState restored_state = *state;  // Committed only on success.
  const HanabiGame& game = *state->ParentGame();
  const std::vector<HanabiHand::CardKnowledge> knowledge = KnowledgeOf(*state, player);
  const int hand_size = static_cast<int>(knowledge.size());
  ReturnHand(&restored_state, player);

  // Deals the sampled card for `position`; false if no candidate is left.
  const auto deal_sampled = [&](int position, std::vector<int> counts) {
    const int total = std::accumulate(counts.begin(), counts.end(), 0);
    if (total == 0) return false;
    const int card = SampleIndex(counts, total);
    return DealSpecific(&restored_state, position, player, card / game.NumRanks(),
                        card % game.NumRanks());
  };

  int position = 0;
  for (int i = 0; i < static_cast<int>(remembered.size()); ++i) {
    if (i == removed_index) continue;
    if (position >= hand_size) return false;
    const HanabiCard& card = remembered[i];
    bool dealt = false;
    if (removed_card.IsValid() && card == removed_card) {
      // The played or discarded copy may have been the last one available.
      std::vector<HanabiCard> later_cards;
      for (int j = i + 1; j < static_cast<int>(remembered.size()); ++j) {
        if (j != removed_index) later_cards.push_back(remembered[j]);
      }
      std::vector<int> counts =
          CandidateCounts(restored_state, player, position, &knowledge[position], later_cards);
      if (counts[CardIndex(game, card.Color(), card.Rank())] == 0) {
        if (std::accumulate(counts.begin(), counts.end(), 0) == 0) {
          // Nothing fits the hints: forget them for this card.
          restored_state.RemoveKnowledge(player, position);
          counts = CandidateCounts(restored_state, player, position, nullptr, later_cards);
        }
        if (!deal_sampled(position, counts)) return false;
        dealt = true;
      }
    }
    if (!dealt && !DealSpecific(&restored_state, position, player, card.Color(), card.Rank())) {
      return false;
    }
    ++position;
  }
  if (position < hand_size) {
    // The card drawn after acting.
    if (!deal_sampled(position, CandidateCounts(restored_state, player, position,
                                                &knowledge[position], std::vector<HanabiCard>()))) {
      return false;
    }
  }
  *state = restored_state;
  return true;
}

int HandSampler::SampleIndex(const std::vector<int>& counts, int total) {
  // Rejection sampling: uniform in [0, total) without modulo bias.
  const uint64_t bound = static_cast<uint64_t>(total);
  const uint64_t limit = UINT64_MAX - UINT64_MAX % bound;
  uint64_t draw = 0;
  do {
    draw = rng_();
  } while (draw >= limit);
  int target = static_cast<int>(draw % bound);
  for (int i = 0; i < static_cast<int>(counts.size()); ++i) {
    if (target < counts[i]) return i;
    target -= counts[i];
  }
  return static_cast<int>(counts.size()) - 1;  // Unreachable when total is the sum.
}

}  // namespace hanabi_learning_env
