#include "observation_features.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <mdspan>
#include <vector>

#include "hanabi_card.h"
#include "hanabi_game.h"
#include "hanabi_hand.h"

namespace hanabi_learning_env {

void UnseenCardCounts(const HanabiObservation& observation, int32_t* counts) {
  const HanabiGame& game = *observation.ParentGame();
  const std::mdspan unseen(counts, game.NumColors(), game.NumRanks());  // [color, rank]
  for (int color = 0; color < game.NumColors(); ++color) {
    for (int rank = 0; rank < game.NumRanks(); ++rank) {
      unseen[color, rank] = game.NumberCardInstances(color, rank);
    }
  }
  auto remove = [&](const HanabiCard& card) {
    // The observer's own cards are hidden (invalid), so they are skipped.
    if (card.IsValid()) --unseen[card.Color(), card.Rank()];
  };
  for (const HanabiHand& hand : observation.Hands()) {
    for (const HanabiCard& card : hand.Cards()) remove(card);
  }
  for (const HanabiCard& card : observation.DiscardPile()) remove(card);
  const std::vector<int>& fireworks = observation.Fireworks();
  for (int color = 0; color < static_cast<int>(fireworks.size()); ++color) {
    for (int rank = 0; rank < fireworks[color]; ++rank) --unseen[color, rank];
  }
}

void PlausibleCards(const HanabiObservation& observation, uint8_t* plausible) {
  const HanabiGame& game = *observation.ParentGame();
  // Locals, not game getters, in the loops: byte stores may alias the game,
  // which would force a reload of every getter after each write.
  const int num_colors = game.NumColors();
  const int num_ranks = game.NumRanks();
  // [player offset, hand position, color, rank]
  const std::mdspan grid(plausible, game.NumPlayers(), game.HandSize(), num_colors, num_ranks);
  std::fill(plausible, plausible + grid.size(), 0);
  const std::vector<HanabiHand>& hands = observation.Hands();
  for (int offset = 0; offset < static_cast<int>(hands.size()); ++offset) {
    // Returned cards keep their knowledge slot, so the cards bound the positions.
    const auto& knowledge = hands[offset].Knowledge();
    const int num_cards = static_cast<int>(hands[offset].Cards().size());
    for (int index = 0; index < num_cards; ++index) {
      // The [color, rank] grid of this card, located once per card.
      const std::mdspan card(&grid[offset, index, 0, 0], num_colors, num_ranks);
      for (int color = 0; color < num_colors; ++color) {
        if (!knowledge[index].ColorPlausible(color)) continue;
        for (int rank = 0; rank < num_ranks; ++rank) {
          card[color, rank] = knowledge[index].RankPlausible(rank) ? 1 : 0;
        }
      }
    }
  }
}

}  // namespace hanabi_learning_env
