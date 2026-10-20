// Helpers shared by the C API source files; not part of the exported API.

#ifndef PYHANABI_INTERNAL_H_
#define PYHANABI_INTERNAL_H_

#include <cstddef>
#include <span>
#include <vector>

#include "hanabi_lib/hanabi_card.h"
#include "hanabi_lib/hanabi_observation.h"
#include "hanabi_lib/util.h"
#include "pyhanabi.h"

namespace pyhanabi_internal {

// Whether `index` is a valid position in a container of `size` elements.
inline bool InRange(int index, size_t size) {
  return index >= 0 && static_cast<size_t>(index) < size;
}

// Copies a card into the caller's struct; false if `index` is out of range.
inline bool CopyCard(std::span<const hanabi_learning_env::HanabiCard> cards, int index,
                     pyhanabi_card_t* card) {
  REQUIRE(card != nullptr);
  if (!InRange(index, cards.size())) return false;
  card->color = cards[index].Color();
  card->rank = cards[index].Rank();
  return true;
}

inline const hanabi_learning_env::HanabiObservation& Observation(
    const pyhanabi_observation_t* observation) {
  REQUIRE(observation != nullptr);
  REQUIRE(observation->observation != nullptr);
  return *static_cast<const hanabi_learning_env::HanabiObservation*>(observation->observation);
}

}  // namespace pyhanabi_internal

#endif  // PYHANABI_INTERNAL_H_
