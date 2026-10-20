// Card-counting features of one player's observation, for agents and networks.
//
// Both functions read only the observation, so they never reveal the
// observer's own cards. Outputs are flat row-major arrays written into
// caller-provided buffers; card identities are indexed color * num_ranks + rank.

#ifndef HANABI_LIB_OBSERVATION_FEATURES_H_
#define HANABI_LIB_OBSERVATION_FEATURES_H_

#include <cstdint>

#include "hanabi_observation.h"

namespace hanabi_learning_env {

// Writes, for each card identity, how many copies the observer cannot see:
// the full deck minus the other players' hands, the discard pile and the
// fireworks. These copies are in the deck or in the observer's own hand.
// `counts` holds num_colors * num_ranks values.
void UnseenCardCounts(const HanabiObservation& observation, int32_t* counts);

// Writes, for each player offset p, hand position h and card identity, 1 if
// the hints that player received still allow that card there, else 0.
// Positions past the end of a hand are all 0. `plausible` holds
// num_players * hand_size * num_colors * num_ranks values, indexed
// ((p * hand_size + h) * num_colors + color) * num_ranks + rank.
void PlausibleCards(const HanabiObservation& observation, uint8_t* plausible);

}  // namespace hanabi_learning_env

#endif  // HANABI_LIB_OBSERVATION_FEATURES_H_
