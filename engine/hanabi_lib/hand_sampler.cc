#include "hand_sampler.h"

#include <array>
#include <cstdint>
#include <format>
#include <numeric>
#include <span>
#include <vector>

#include "fixed_vector.h"
#include "hanabi_game.h"
#include "util.h"

namespace hanabi_learning_env {

namespace {

// Bounds the restarts of ResampleHand; a consistent hand normally takes one or two.
constexpr int kMaxAttempts = 10000;

// Card indices (color * num_ranks + rank) by hand position.
using PlannedHand = FixedVector<int, kMaxHandSize>;

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

// Swaps `player`'s hand for `hand` (card indices by position), which must be
// drawable from the deck once the hand is returned. `forget[position]` resets
// the knowledge of that position.
void CommitHand(HanabiState* state, int player, const PlannedHand& hand,
                std::span<const bool> forget) {
  const HanabiGame& game = *state->ParentGame();
  HanabiHand::CardList cards;
  for (int card : hand) cards.push_back(HanabiCard(card / game.NumRanks(), card % game.NumRanks()));
  REQUIRE(state->ReplaceHand(player, cards).has_value());  // Drawable by construction.
  for (int position = 0; position < static_cast<int>(forget.size()); ++position) {
    if (forget[position]) state->RemoveKnowledge(player, position);
  }
}

bool IsPlayer(const HanabiState& state, int player) {
  return player >= 0 && player < state.ParentGame()->NumPlayers();
}

std::unexpected<std::string> NoCandidate(int player, int position) {
  return std::unexpected(
      std::format("no card fits player {}'s knowledge at position {}", player, position));
}

std::unexpected<std::string> NotAPlayer(const HanabiState& state, int player) {
  return std::unexpected(
      std::format("player {} is not in [0, {})", player, state.ParentGame()->NumPlayers()));
}

}  // namespace

std::vector<int> HandSampler::CandidateCounts(const HanabiState& state, int player, int card_index,
                                              const HanabiHand::CardKnowledge* knowledge,
                                              std::span<const HanabiCard> excluded) {
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

// Both sampling functions plan the new hand on counts alone, then swap it in
// with HanabiState::ReplaceHand: a failure leaves the state untouched.

Status HandSampler::ResampleHand(HanabiState* state, int player) {
  if (!IsPlayer(*state, player)) return NotAPlayer(*state, player);
  const HanabiGame& game = *state->ParentGame();
  const int hand_size = static_cast<int>(state->Hands()[player].Cards().size());
  const Counts available = DeckWithHand(*state, player);

  PlannedHand hand;
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
  if (static_cast<int>(hand.size()) < hand_size) {
    return std::unexpected(
        std::format("no hand fits player {}'s knowledge ({} attempts)", player, kMaxAttempts));
  }
  CommitHand(state, player, hand, {});
  return {};
}

Status HandSampler::RestoreHand(HanabiState* state, int player,
                                std::span<const HanabiCard> remembered,
                                const HanabiCard& removed_card, int removed_index) {
  if (!IsPlayer(*state, player)) return NotAPlayer(*state, player);
  if (remembered.size() > HanabiHand::CardList::capacity()) {
    return std::unexpected(
        std::format("remembered hand has {} cards, more than {}", remembered.size(), kMaxHandSize));
  }
  const HanabiGame& game = *state->ParentGame();
  const int hand_size = static_cast<int>(state->Hands()[player].Cards().size());
  Counts deck = DeckWithHand(*state, player);  // Minus the cards planned so far.
  PlannedHand hand;
  std::array<bool, kMaxHandSize> forget{};

  // Candidates for the next position: the deck minus `later` cards, filtered
  // by `knowledge` (nullptr: no filter).
  const auto candidates = [&](const HanabiHand::CardKnowledge* knowledge,
                              std::span<const HanabiCard> later) {
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

  HanabiHand::CardList later_cards;
  for (int i = 0; i < static_cast<int>(remembered.size()); ++i) {
    if (i == removed_index) continue;
    const int position = static_cast<int>(hand.size());
    if (position >= hand_size) {
      return std::unexpected(
          std::format("remembered hand has more cards than player {} holds", player));
    }
    const HanabiCard& card = remembered[i];
    if (game.NumberCardInstances(card) == 0) {
      return std::unexpected(
          std::format("card (color {}, rank {}) is not in the game", card.Color(), card.Rank()));
    }
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
        if (!plan_sampled(counts)) return NoCandidate(player, position);
        continue;
      }
    }
    if (deck[card_index] == 0) {
      return std::unexpected(std::format("no copy of remembered {} is left", card.ToString()));
    }
    hand.push_back(card_index);
    --deck[card_index];
  }
  const int position = static_cast<int>(hand.size());
  if (position < hand_size) {
    // The card drawn after acting.
    if (!plan_sampled(candidates(KnowledgeAt(*state, player, position), {}))) {
      return NoCandidate(player, position);
    }
  }
  CommitHand(state, player, hand, std::span<const bool>(forget).first(hand.size()));
  return {};
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

Status AdvanceDeterminized(HanabiState* state, const HanabiMove& move, int searching_player,
                           Determinization determinization, RememberedHand* remembered,
                           HandSampler* sampler) {
  const int acting_player = state->CurPlayer();
  HanabiCard acted_card;
  if (move.MoveType() == HanabiMove::kPlay || move.MoveType() == HanabiMove::kDiscard) {
    acted_card = state->Hands()[acting_player].Cards()[move.CardIndex()];
  }
  state->ApplyMove(move);
  while (state->CurPlayer() == kChancePlayerId) state->ApplyRandomChance();

  if (determinization == Determinization::kRestore && acting_player != searching_player) {
    if (!remembered->valid) {
      return std::unexpected(
          std::format("no remembered hand to restore for player {}", acting_player));
    }
    const Status restored =
        sampler->RestoreHand(state, acting_player, remembered->cards, acted_card,
                             acted_card.IsValid() ? move.CardIndex() : -1);
    if (!restored) return restored;
  }
  const int next_player = state->CurPlayer();
  if (determinization != Determinization::kNone && !state->IsTerminal() &&
      next_player != searching_player) {
    remembered->cards = state->Hands()[next_player].Cards();
    remembered->valid = true;
    return sampler->ResampleHand(state, next_player);
  }
  return {};
}

}  // namespace hanabi_learning_env
