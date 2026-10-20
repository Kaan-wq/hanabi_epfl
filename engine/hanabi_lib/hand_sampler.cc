#include "hand_sampler.h"

#include <array>
#include <bit>
#include <cstdint>
#include <format>
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

int CardIndex(const HanabiGame& game, int color, int rank) {
  return color * game.NumRanks() + rank;
}

int NumCards(const HanabiGame& game) { return game.NumColors() * game.NumRanks(); }

// Removes one copy of `card` if any is left.
void RemoveOne(const HanabiGame& game, const HanabiCard& card, CardCounts* counts) {
  int& count = (*counts)[CardIndex(game, card.Color(), card.Rank())];
  if (count > 0) --count;
}

CardMask AnyCard(const HanabiGame& game) { return AllCards(game.NumColors(), game.NumRanks()); }

// The cards `knowledge` allows; every card for nullptr (no filter).
CardMask AllowedCards(const HanabiGame& game, const HanabiHand::CardKnowledge* knowledge) {
  if (knowledge == nullptr) return AnyCard(game);
  return PlausibleCards(*knowledge, game.NumColors(), game.NumRanks());
}

// Zeroes the cards `knowledge` rules out (nullptr: none).
void FilterByKnowledge(const HanabiGame& game, const HanabiHand::CardKnowledge* knowledge,
                       CardCounts* counts) {
  const CardMask plausible = AllowedCards(game, knowledge);
  for (int card = 0; card < NumCards(game); ++card) {
    if (!Contains(plausible, card)) (*counts)[card] = 0;
  }
}

// The deck once `player` has returned their whole hand: what it can be dealt from.
CardCounts DeckWithHand(const HanabiState& state, int player) {
  const HanabiGame& game = *state.ParentGame();
  CardCounts counts = state.Deck().Counts();
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
  CardCounts counts = state.Deck().Counts();
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
  const CardCounts available = DeckWithHand(*state, player);
  std::array<CardMask, kMaxHandSize> plausible{};
  for (int position = 0; position < hand_size; ++position) {
    plausible[position] = AllowedCards(game, KnowledgeAt(*state, player, position));
  }

  PlannedHand hand;
  for (int attempt = 0; attempt < kMaxAttempts && static_cast<int>(hand.size()) < hand_size;
       ++attempt) {
    hand.clear();
    CardCounts remaining = available;  // Minus the cards planned so far.
    for (int position = 0; position < hand_size; ++position) {
      const int total = Total(remaining, plausible[position]);
      if (total == 0) break;
      hand.push_back(SampleIndex(remaining, plausible[position], total));
      --remaining[hand.back()];
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
  CardCounts deck = DeckWithHand(*state, player);  // Minus the cards planned so far.
  PlannedHand hand;
  std::array<bool, kMaxHandSize> forget{};

  // Candidates for the next position: the deck minus `later` cards, filtered
  // by `knowledge` (nullptr: no filter).
  const auto candidates = [&](const HanabiHand::CardKnowledge* knowledge,
                              std::span<const HanabiCard> later) {
    CardCounts counts = deck;
    for (const HanabiCard& card : later) RemoveOne(game, card, &counts);
    FilterByKnowledge(game, knowledge, &counts);
    return counts;
  };
  // Plans a sampled card for the next position; false if no candidate is left.
  const auto plan_sampled = [&](const CardCounts& counts) {
    const int total = Total(counts, AnyCard(game));
    if (total == 0) return false;
    hand.push_back(SampleIndex(counts, AnyCard(game), total));
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
      CardCounts counts = candidates(KnowledgeAt(*state, player, position), later_cards);
      if (counts[card_index] == 0) {
        if (Total(counts, AnyCard(game)) == 0) {
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

void HandSampler::DealPendingCards(HanabiState* state) {
  const HanabiGame& game = *state->ParentGame();
  while (state->CurPlayer() == kChancePlayerId) {
    // Chance outcome ids are card indices.
    state->ApplyMove(game.GetChanceOutcome(
        SampleIndex(state->Deck().Counts(), AnyCard(game), state->Deck().Size())));
  }
}

int HandSampler::SampleIndex(const CardCounts& counts, CardMask mask, int total) {
  // Uniform in [0, total) by Lemire's multiply-and-shift: the high half of
  // draw * total, with the few biased low halves rejected. Unlike a modulo,
  // it needs a division only on the rare draw that might be biased.
  const auto bound = static_cast<uint32_t>(total);
  uint64_t product = (rng_() >> 32) * bound;
  if (static_cast<uint32_t>(product) < bound) {
    const uint32_t threshold = (0U - bound) % bound;  // 2^32 mod bound.
    while (static_cast<uint32_t>(product) < threshold) product = (rng_() >> 32) * bound;
  }
  int target = static_cast<int>(product >> 32);
  int last = 0;
  for (; mask != 0; mask &= mask - 1) {
    last = std::countr_zero(mask);
    if (target < counts[last]) return last;
    target -= counts[last];
  }
  return last;  // Unreachable when total is the sum.
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
  sampler->DealPendingCards(state);

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
