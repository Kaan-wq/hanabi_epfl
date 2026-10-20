#include "hand_sampler.h"

#include <array>
#include <cstdint>
#include <numeric>
#include <vector>

#include "hanabi_game.h"
#include "hanabi_move.h"
#include "util.h"

namespace hanabi_learning_env {

namespace {

// Bounds the restarts of ResampleHand; a consistent hand normally takes one or two.
constexpr int kMaxAttempts = 10000;

// Copies of each card, indexed color * num_ranks + rank. Fixed-size, so the
// sampling loops allocate nothing.
using Counts = std::array<int, static_cast<size_t>(kMaxNumColors) * kMaxNumRanks>;

int CardIndex(const HanabiGame& game, int color, int rank) {
  return color * game.NumRanks() + rank;
}

int NumCards(const HanabiGame& game) { return game.NumColors() * game.NumRanks(); }

// Removes one copy of `card` if any is left.
void RemoveOne(const HanabiGame& game, const HanabiCard& card, Counts* counts) {
  int& count = (*counts)[CardIndex(game, card.Color(), card.Rank())];
  if (count > 0) --count;
}

// Zeroes the cards `knowledge` rules out; returns the remaining total.
int FilterByKnowledge(const HanabiGame& game, const HanabiHand::CardKnowledge* knowledge,
                      Counts* counts) {
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

int Total(const HanabiGame& game, const Counts& counts) {
  return std::accumulate(counts.begin(), counts.begin() + NumCards(game), 0);
}

// The deck once `player` has returned their whole hand: what it can be dealt from.
Counts DeckWithHand(const HanabiState& state, int player) {
  const HanabiGame& game = *state.ParentGame();
  Counts counts{};
  for (int color = 0; color < game.NumColors(); ++color) {
    for (int rank = 0; rank < game.NumRanks(); ++rank) {
      counts[CardIndex(game, color, rank)] = state.Deck().CardCount(color, rank);
    }
  }
  for (const HanabiCard& card : state.Hands()[player].Cards()) {
    ++counts[CardIndex(game, card.Color(), card.Rank())];
  }
  return counts;
}

// What `player` knows about the card at `position`, as their observation
// shows it; nullptr (no filter) with kMinimal observations, where hints are
// not remembered.
const HanabiHand::CardKnowledge* KnowledgeAt(const HanabiState& state, int player, int position) {
  if (state.ParentGame()->ObservationType() == HanabiGame::kMinimal) return nullptr;
  return &state.Hands()[player].Knowledge()[position];
}

void ReturnHand(HanabiState* state, int player) {
  const auto num_cards = state->Hands()[player].Cards().size();
  for (size_t i = 0; i < num_cards; ++i) {
    // Cards shift left after each return, so always return position 0.
    state->ApplyMove(HanabiMove(HanabiMove::kReturn, 0, player, -1, -1));
  }
}

// Swaps `player`'s hand for `hand` (card indices by position), which must be
// drawable from the deck once the hand is returned. `forget[position]` resets
// the knowledge of that position first.
void CommitHand(HanabiState* state, int player, const std::vector<int>& hand,
                const std::vector<bool>& forget) {
  const HanabiGame& game = *state->ParentGame();
  ReturnHand(state, player);
  for (int position = 0; position < static_cast<int>(hand.size()); ++position) {
    if (!forget.empty() && forget[position]) state->RemoveKnowledge(player, position);
    const HanabiMove deal(HanabiMove::kDealSpecific, position, player,
                          hand[position] / game.NumRanks(), hand[position] % game.NumRanks());
    state->ApplyMove(deal);  // Legal by construction (drawn from DeckWithHand); checked there.
  }
}

bool IsPlayer(const HanabiState& state, int player) {
  return player >= 0 && player < state.ParentGame()->NumPlayers();
}

}  // namespace

std::vector<int> HandSampler::CandidateCounts(const HanabiState& state, int player, int card_index,
                                              const HanabiHand::CardKnowledge* knowledge,
                                              const std::vector<HanabiCard>& excluded) {
  const HanabiGame& game = *state.ParentGame();
  Counts counts{};
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
  return {counts.begin(), counts.begin() + NumCards(game)};
}

// Both sampling functions plan the new hand on counts alone, then apply it to
// the state in one go: a failure leaves the state untouched without copying it.

bool HandSampler::ResampleHand(HanabiState* state, int player) {
  if (!IsPlayer(*state, player)) return false;
  const HanabiGame& game = *state->ParentGame();
  const int hand_size = static_cast<int>(state->Hands()[player].Cards().size());
  const Counts available = DeckWithHand(*state, player);

  std::vector<int> hand;  // Card indices, by position.
  hand.reserve(hand_size);
  for (int attempt = 0; attempt < kMaxAttempts && static_cast<int>(hand.size()) < hand_size;
       ++attempt) {
    hand.clear();
    for (int position = 0; position < hand_size; ++position) {
      Counts counts = available;
      for (int card : hand) {
        if (counts[card] > 0) --counts[card];
      }
      const int total = FilterByKnowledge(game, KnowledgeAt(*state, player, position), &counts);
      if (total == 0) break;
      hand.push_back(SampleIndex(counts.data(), NumCards(game), total));
    }
  }
  if (static_cast<int>(hand.size()) < hand_size) return false;
  CommitHand(state, player, hand, {});
  return true;
}

bool HandSampler::RestoreHand(HanabiState* state, int player,
                              const std::vector<HanabiCard>& remembered,
                              const HanabiCard& removed_card, int removed_index) {
  if (!IsPlayer(*state, player)) return false;
  const HanabiGame& game = *state->ParentGame();
  const int hand_size = static_cast<int>(state->Hands()[player].Cards().size());
  Counts deck = DeckWithHand(*state, player);  // Minus the cards planned so far.
  std::vector<int> hand;
  hand.reserve(hand_size);
  std::vector<bool> forget(hand_size, false);

  // Candidates for the next position: the deck minus `later` cards, filtered
  // by `knowledge` (nullptr: no filter).
  const auto candidates = [&](const HanabiHand::CardKnowledge* knowledge,
                              const std::vector<HanabiCard>& later) {
    Counts counts = deck;
    for (const HanabiCard& card : later) RemoveOne(game, card, &counts);
    FilterByKnowledge(game, knowledge, &counts);
    return counts;
  };
  // Plans a sampled card for the next position; false if no candidate is left.
  const auto plan_sampled = [&](const Counts& counts) {
    const int total = Total(game, counts);
    if (total == 0) return false;
    hand.push_back(SampleIndex(counts.data(), NumCards(game), total));
    --deck[hand.back()];
    return true;
  };

  std::vector<HanabiCard> later_cards;
  for (int i = 0; i < static_cast<int>(remembered.size()); ++i) {
    if (i == removed_index) continue;
    const int position = static_cast<int>(hand.size());
    if (position >= hand_size) return false;
    const HanabiCard& card = remembered[i];
    if (game.NumberCardInstances(card) == 0) return false;  // Not a card of the game.
    const int card_index = CardIndex(game, card.Color(), card.Rank());
    if (removed_card.IsValid() && card == removed_card) {
      // The played or discarded copy may have been the last one available.
      later_cards.clear();
      for (int j = i + 1; j < static_cast<int>(remembered.size()); ++j) {
        if (j != removed_index) later_cards.push_back(remembered[j]);
      }
      Counts counts = candidates(KnowledgeAt(*state, player, position), later_cards);
      if (counts[card_index] == 0) {
        if (Total(game, counts) == 0) {
          // Nothing fits the hints: forget them for this card.
          forget[position] = true;
          counts = candidates(nullptr, later_cards);
        }
        if (!plan_sampled(counts)) return false;
        continue;
      }
    }
    if (deck[card_index] == 0) return false;
    hand.push_back(card_index);
    --deck[card_index];
  }
  const int position = static_cast<int>(hand.size());
  if (position < hand_size) {
    // The card drawn after acting.
    if (!plan_sampled(candidates(KnowledgeAt(*state, player, position), {}))) {
      return false;
    }
  }
  CommitHand(state, player, hand, forget);
  return true;
}

int HandSampler::SampleIndex(const int* counts, int num_counts, int total) {
  // Rejection sampling: uniform in [0, total) without modulo bias.
  const uint64_t bound = static_cast<uint64_t>(total);
  const uint64_t limit = UINT64_MAX - UINT64_MAX % bound;
  uint64_t draw = 0;
  do {
    draw = rng_();
  } while (draw >= limit);
  int target = static_cast<int>(draw % bound);
  for (int i = 0; i < num_counts; ++i) {
    if (target < counts[i]) return i;
    target -= counts[i];
  }
  return num_counts - 1;  // Unreachable when total is the sum.
}

}  // namespace hanabi_learning_env
