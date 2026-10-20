// Copyright 2018 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// Modified for hanabi_search; see engine/CHANGES.md for every change.

// The standard Open Hanabi observation encoders. These encoders translate
// HanabiObservations to input tensors that an agent can train on.

#ifndef HANABI_LIB_CANONICAL_ENCODERS_H_
#define HANABI_LIB_CANONICAL_ENCODERS_H_

#include <cstdint>
#include <vector>

#include "hanabi_game.h"
#include "hanabi_observation.h"
#include "observation_encoder.h"

namespace hanabi_learning_env {

// This is the canonical observation encoding.
class CanonicalObservationEncoder : public ObservationEncoder {
 public:
  explicit CanonicalObservationEncoder(const HanabiGame* parent_game) : parent_game_(parent_game) {}

  std::vector<int> Shape() const override;
  std::vector<int> Encode(const HanabiObservation& obs) const override;
  // Number of entries in an encoding (the product of Shape()).
  int Length() const;
  // Writes the same encoding as Encode() into `bits` (Length() entries, 0 or
  // 1), without allocating.
  void EncodeInto(const HanabiObservation& obs, uint8_t* bits) const;

  ObservationEncoder::Type type() const override { return ObservationEncoder::Type::kCanonical; }

 private:
  template <typename Bit>
  void EncodeSections(const HanabiObservation& obs, Bit* encoding) const;

  const HanabiGame* parent_game_ = nullptr;
};

}  // namespace hanabi_learning_env

#endif
