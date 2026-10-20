// Hidden-hand sampling for re-determinizing information-set MCTS.
//
// Before another player acts in a search rollout, their hand is replaced by
// one sampled from what *they* know (hints, and every card they can see), so
// the searcher cannot exploit hidden information. After they act, the
// previous hand is restored as closely as possible. Both go through the
// fork's return and deal-specific moves, so the move history records them.

#ifndef HANABI_LIB_HAND_SAMPLER_H_
#define HANABI_LIB_HAND_SAMPLER_H_

#include <cstdint>
#include <vector>

#include "hanabi_card.h"
#include "hanabi_hand.h"
#include "hanabi_state.h"

namespace hanabi_learning_env {

// SplitMix64: cheap to seed (a sampler is often built per call) and gives the
// same sequence on every platform, unlike the standard distributions.
class SplitMix64 {
 public:
  explicit SplitMix64(uint64_t seed) : state_(seed) {}
  uint64_t operator()() {
    uint64_t z = (state_ += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
  }

 private:
  uint64_t state_;
};

class HandSampler {
 public:
  // The sampler has its own random generator; it never uses the game's.
  explicit HandSampler(uint64_t seed) : rng_(seed) {}

  // Copies of each card, indexed color * num_ranks + rank, that could be at
  // position `card_index` of `player`'s hand given every other card in the
  // state: the deck, plus the card now at that position (if any), minus one
  // copy per card in `excluded`. Cards that `knowledge` rules out get 0; pass
  // nullptr to ignore hints.
  static std::vector<int> CandidateCounts(const HanabiState& state, int player, int card_index,
                                          const HanabiHand::CardKnowledge* knowledge,
                                          const std::vector<HanabiCard>& excluded);

  // Returns all of `player`'s cards to the deck, then deals a hand of the same
  // size sampled card by card from what the player knows, restarting when a
  // position has no candidate left. Returns false, leaving `state` unchanged,
  // if `player` is out of range or no consistent hand is found.
  bool ResampleHand(HanabiState* state, int player);

  // Returns all of `player`'s cards to the deck, then deals back
  // `remembered` (their hand before they acted) without the card at
  // `removed_index`, and samples a card for any position left over (the card
  // they drew). `removed_card` is the card they played or discarded (invalid
  // after a hint); a remembered copy of it that is no longer available is
  // replaced by a sampled card, ignoring hints if nothing else fits. Returns
  // false, leaving `state` unchanged, if `player` is out of range or no
  // consistent hand is found.
  bool RestoreHand(HanabiState* state, int player, const std::vector<HanabiCard>& remembered,
                   const HanabiCard& removed_card, int removed_index);

 private:
  // Picks an index in [0, num_counts) with probability counts[i] / total;
  // total must be the (positive) sum of the counts.
  int SampleIndex(const int* counts, int num_counts, int total);

  SplitMix64 rng_;
};

}  // namespace hanabi_learning_env

#endif  // HANABI_LIB_HAND_SAMPLER_H_
