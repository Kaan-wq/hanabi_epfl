#include "observation_features.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>

#include "hanabi_card.h"
#include "hanabi_game.h"
#include "hanabi_hand.h"

namespace hanabi_learning_env {

void UnseenCardCounts(const HanabiObservation& observation, int32_t* counts) {
  const HanabiGame& game = *observation.ParentGame();
  const int num_ranks = game.NumRanks();
  for (int color = 0; color < game.NumColors(); ++color) {
    for (int rank = 0; rank < num_ranks; ++rank) {
      counts[color * num_ranks + rank] = game.NumberCardInstances(color, rank);
    }
  }
  auto remove = [&](const HanabiCard& card) {
    // The observer's own cards are hidden (invalid), so they are skipped.
    if (card.IsValid()) --counts[card.Color() * num_ranks + card.Rank()];
  };
  for (const HanabiHand& hand : observation.Hands()) {
    for (const HanabiCard& card : hand.Cards()) remove(card);
  }
  for (const HanabiCard& card : observation.DiscardPile()) remove(card);
  const std::vector<int>& fireworks = observation.Fireworks();
  for (int color = 0; color < static_cast<int>(fireworks.size()); ++color) {
    for (int rank = 0; rank < fireworks[color]; ++rank) --counts[color * num_ranks + rank];
  }
}

void PlausibleCards(const HanabiObservation& observation, uint8_t* plausible) {
  const HanabiGame& game = *observation.ParentGame();
  const int num_colors = game.NumColors();
  const int num_ranks = game.NumRanks();
  const size_t cards = static_cast<size_t>(num_colors) * num_ranks;
  const size_t hand_size = game.HandSize();
  std::fill(plausible, plausible + game.NumPlayers() * hand_size * cards, 0);
  const std::vector<HanabiHand>& hands = observation.Hands();
  for (int offset = 0; offset < static_cast<int>(hands.size()); ++offset) {
    // Returned cards keep their knowledge slot, so the cards bound the positions.
    const auto& knowledge = hands[offset].Knowledge();
    const int num_cards = static_cast<int>(hands[offset].Cards().size());
    for (int index = 0; index < num_cards; ++index) {
      uint8_t* out = plausible + (offset * hand_size + index) * cards;
      for (int color = 0; color < num_colors; ++color) {
        if (!knowledge[index].ColorPlausible(color)) continue;
        for (int rank = 0; rank < num_ranks; ++rank) {
          out[color * num_ranks + rank] = knowledge[index].RankPlausible(rank) ? 1 : 0;
        }
      }
    }
  }
}

}  // namespace hanabi_learning_env
