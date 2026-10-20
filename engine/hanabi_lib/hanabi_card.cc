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

#include "hanabi_card.h"

#include <nlohmann/json.hpp>
using json = nlohmann::json;
#include "json_checks.h"
#include "util.h"

namespace hanabi_learning_env {

std::string HanabiCard::ToString() const {
  if (!IsValid()) {
    return std::string("XX");
  }
  return std::string() + ColorIndexToChar(Color()) + RankIndexToChar(Rank());
}

// ===== Serialization + Deserialization =====

json HanabiCard::toJSON() const {
  json j;
  j["color"] = color_;
  j["rank"] = rank_;
  return j;
}

HanabiCard HanabiCard::fromJSON(const json& j) {
  // -1 marks an unknown card (e.g. a player's own card in an observation).
  int color = JsonInt(j, "color", -1, kMaxNumColors - 1);
  int rank = JsonInt(j, "rank", -1, kMaxNumRanks - 1);
  return HanabiCard(color, rank);
}

// ===========================================

}  // namespace hanabi_learning_env
