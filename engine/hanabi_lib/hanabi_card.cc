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

#include "hanabi_card.h"
#include <nlohmann/json.hpp>
using json = nlohmann::json;
#include "util.h"

namespace hanabi_learning_env {

bool HanabiCard::operator==(const HanabiCard& other_card) const {
  return other_card.Color() == Color() && other_card.Rank() == Rank();
}

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
    int color = j.at("color").get<int>();
    int rank = j.at("rank").get<int>();
    return HanabiCard(color, rank);
}

// ===========================================

}  // namespace hanabi_learning_env
