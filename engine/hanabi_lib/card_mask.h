// Sets of card identities as bitmasks, and per-identity counts, both indexed
// color * num_ranks + rank. Hand sampling and the Van den Bergh rules filter
// counts by what a player knows about a card many times per search iteration;
// with masks, each filter is a few bit operations instead of a loop over
// every color and rank.

#ifndef HANABI_LIB_CARD_MASK_H_
#define HANABI_LIB_CARD_MASK_H_

#include <array>
#include <cstddef>
#include <cstdint>

#include "hanabi_hand.h"
#include "util.h"

namespace hanabi_learning_env {

using CardMask = uint32_t;
static_assert(kMaxNumColors * kMaxNumRanks <= 32, "a CardMask has one bit per card identity");

// Copies of each card identity; fixed-size, so counting never allocates.
using CardCounts = std::array<int, static_cast<size_t>(kMaxNumColors) * kMaxNumRanks>;

inline CardMask AllCards(int num_colors, int num_ranks) {
  return (CardMask{1} << (num_colors * num_ranks)) - 1;
}

// The identities `knowledge` allows: plausible color and plausible rank.
inline CardMask PlausibleCards(const HanabiHand::CardKnowledge& knowledge, int num_colors,
                               int num_ranks) {
  CardMask ranks = 0;  // The plausible ranks of one color.
  for (int rank = 0; rank < num_ranks; ++rank) {
    if (knowledge.RankPlausible(rank)) ranks |= CardMask{1} << rank;
  }
  CardMask mask = 0;
  for (int color = 0; color < num_colors; ++color) {
    if (knowledge.ColorPlausible(color)) mask |= ranks << (color * num_ranks);
  }
  return mask;
}

inline bool Contains(CardMask mask, int card) { return ((mask >> card) & 1U) != 0; }

// Copies of the identities in `mask`.
inline int Total(const CardCounts& counts, CardMask mask) {
  int total = 0;
  // A fixed trip count without branches, so the compiler vectorizes it.
  for (size_t card = 0; card < counts.size(); ++card) {
    total += static_cast<int>((mask >> card) & 1U) * counts[card];
  }
  return total;
}

}  // namespace hanabi_learning_env

#endif  // HANABI_LIB_CARD_MASK_H_
