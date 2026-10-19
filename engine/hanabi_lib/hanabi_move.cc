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

#include "hanabi_move.h"

#include <cstdint>
#include <nlohmann/json.hpp>

#include "json_checks.h"
#include "util.h"
using json = nlohmann::json;

namespace hanabi_learning_env {

bool HanabiMove::operator==(const HanabiMove& other_move) const {
  // MB: Check if two moves are the same? not quite sure
  if (MoveType() != other_move.MoveType()) {
    return false;
  }
  switch (MoveType()) {
    case kPlay:
    case kDiscard:
      return CardIndex() == other_move.CardIndex();
    case kReturn:
      return CardIndex() == other_move.CardIndex() && TargetOffset() == other_move.TargetOffset();
    case kRevealColor:
      return TargetOffset() == other_move.TargetOffset() && Color() == other_move.Color();
    case kRevealRank:
      return TargetOffset() == other_move.TargetOffset() && Rank() == other_move.Rank();
    case kDeal:
      return Color() == other_move.Color() && Rank() == other_move.Rank();
    case kDealSpecific:
      return Color() == other_move.Color() && Rank() == other_move.Rank();
    default:
      return true;
  }
}

std::string HanabiMove::ToString() const {
  switch (MoveType()) {
    case kPlay:
      return "(Play " + std::to_string(CardIndex()) + ")";
    case kReturn:
      return "(Return " + std::to_string(CardIndex()) + "from Player " +
             std::to_string(TargetOffset()) + ")";
    case kDiscard:
      return "(Discard " + std::to_string(CardIndex()) + ")";
    case kRevealColor:
      return "(Reveal player +" + std::to_string(TargetOffset()) + " color " +
             ColorIndexToChar(Color()) + ")";
    case kRevealRank:
      return "(Reveal player +" + std::to_string(TargetOffset()) + " rank " +
             RankIndexToChar(Rank()) + ")";
    case kDeal:
      if (color_ >= 0) {
        return std::string("(Deal ") + ColorIndexToChar(Color()) + RankIndexToChar(Rank()) + ")";
      } else {
        return std::string("(Deal XX)");
      }
    case kDealSpecific:
      if (color_ >= 0) {
        return std::string("(Deal ") + ColorIndexToChar(Color()) + RankIndexToChar(Rank()) + ")";
      } else {
        return std::string("(Deal XX)");
      }
    default:
      return "(INVALID)";
  }
}

// =========================== Serialization + Deserialization ===========================

// Serialization
json HanabiMove::toJSON() const {
  json j;

  // Serialize enum type
  j["move_type"] = static_cast<int>(move_type_);

  // Serialize simple types
  j["card_index"] = card_index_;
  j["target_offset"] = target_offset_;
  j["color"] = color_;
  j["rank"] = rank_;

  return j;
}

// Deserialization
HanabiMove HanabiMove::fromJSON(const json& j) {
  // Ranges a move can hold in any game; ranges that depend on the game are
  // checked by MoveIsLegal (and by HanabiState::fromJSON for move history).
  Type move_type = static_cast<Type>(JsonInt(j, "move_type", kInvalid, kDealSpecific));
  int card_index = JsonInt(j, "card_index", INT8_MIN, INT8_MAX);
  int target_offset = JsonInt(j, "target_offset", INT8_MIN, INT8_MAX);
  int color = JsonInt(j, "color", -1, kMaxNumColors - 1);
  int rank = JsonInt(j, "rank", -1, kMaxNumRanks - 1);

  return HanabiMove(move_type, card_index, target_offset, color, rank);
}

// =======================================================================================

}  // namespace hanabi_learning_env
