// Helpers shared by the C API source files; not part of the exported API.

#ifndef PYHANABI_INTERNAL_H_
#define PYHANABI_INTERNAL_H_

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <format>
#include <mdspan>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include "hanabi_lib/hanabi_card.h"
#include "hanabi_lib/hanabi_hand.h"
#include "hanabi_lib/hanabi_observation.h"
#include "hanabi_lib/util.h"
#include "pyhanabi.h"

namespace pyhanabi_internal {

// Whether `index` is a valid position in a container of `size` elements.
inline bool InRange(int index, size_t size) {
  return index >= 0 && static_cast<size_t>(index) < size;
}

// Records why the current call fails, for pyhanabi_last_error().
void SetLastError(std::string message);

// Records `message`; returns false (or -1, for int getters).
inline bool Fail(std::string message) {
  SetLastError(std::move(message));
  return false;
}
inline int FailInt(std::string message) {
  SetLastError(std::move(message));
  return -1;
}

// True if `index` is in [0, size); otherwise records "<what> <index> is not
// in [0, <size>)" and returns false.
inline bool CheckIndex(int index, size_t size, const char* what) {
  if (InRange(index, size)) return true;
  return Fail(std::format("{} {} is not in [0, {})", what, index, size));
}

// Copies card `index` into the caller's struct; false if it is out of range.
inline bool CopyCard(std::span<const hanabi_learning_env::HanabiCard> cards, int index,
                     pyhanabi_card_t* card) {
  REQUIRE(card != nullptr);
  if (!CheckIndex(index, cards.size(), "card index")) return false;
  card->color = cards[index].Color();
  card->rank = cards[index].Rank();
  return true;
}

// Writes every hand into caller buffers: `hand_sizes` [player] and `colors`,
// `ranks` [player, position] (players * hand_size entries; -1 for hidden
// cards and past the end of a hand).
inline void WriteHands(const std::vector<hanabi_learning_env::HanabiHand>& hands, int hand_size,
                       int32_t* hand_sizes, int8_t* colors, int8_t* ranks) {
  const std::mdspan color_grid(colors, hands.size(), hand_size);
  const std::mdspan rank_grid(ranks, hands.size(), hand_size);
  std::fill(colors, colors + color_grid.size(), -1);
  std::fill(ranks, ranks + rank_grid.size(), -1);
  for (size_t player = 0; player < hands.size(); ++player) {
    const auto& cards = hands[player].Cards();
    hand_sizes[player] = static_cast<int32_t>(cards.size());
    for (size_t index = 0; index < cards.size(); ++index) {
      color_grid[player, index] = static_cast<int8_t>(cards[index].Color());
      rank_grid[player, index] = static_cast<int8_t>(cards[index].Rank());
    }
  }
}

inline const hanabi_learning_env::HanabiObservation& Observation(
    const pyhanabi_observation_t* observation) {
  REQUIRE(observation != nullptr);
  REQUIRE(observation->observation != nullptr);
  return *static_cast<const hanabi_learning_env::HanabiObservation*>(observation->observation);
}

}  // namespace pyhanabi_internal

#endif  // PYHANABI_INTERNAL_H_
