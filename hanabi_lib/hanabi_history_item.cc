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

#include "hanabi_history_item.h"

#include <cassert>
#include <nlohmann/json.hpp>
using json = nlohmann::json;

#include "util.h"

namespace hanabi_learning_env {

std::string HanabiHistoryItem::ToString() const {
  std::string str = "<" + move.ToString();
  if (player >= 0) {
    str += " by player " + std::to_string(player);
  }
  if (scored) {
    str += " scored";
  }
  if (information_token) {
    str += " info_token";
  }
  if (color >= 0) {
    assert(rank >= 0);
    str += " ";
    str += ColorIndexToChar(color);
    str += RankIndexToChar(rank);
  }
  if (reveal_bitmask) {
    str += " reveal ";
    bool first = true;
    for (int i = 0; i < 8; ++i) {  // 8 bits in reveal_bitmask
      if (reveal_bitmask & (1 << i)) {
        if (first) {
          first = false;
        } else {
          str += ",";
        }
        str += std::to_string(i);
      }
    }
  }
  str += ">";
  return str;
}

void ChangeToObserverRelative(int observer_pid, int player_count,
                              HanabiHistoryItem* item) {
  if (item->move.MoveType() == HanabiMove::kDeal) {
    assert(item->player < 0 && item->deal_to_player >= 0);
    item->deal_to_player =
        (item->deal_to_player - observer_pid + player_count) % player_count;
    if (item->deal_to_player == 0) {
      // Hide cards dealt to observer.
      item->move = HanabiMove(HanabiMove::kDeal, -1, -1, -1, -1);
    }
  } else {
    assert(item->player >= 0);
    item->player = (item->player - observer_pid + player_count) % player_count;
  }
}

/*=================================================================================
               HanabiHistoryItem Serialization + Deserialization
=================================================================================*/ 

json HanabiHistoryItem::toJSON() const {
    json j;

    j["move"] = move.toJSON();
    j["player"] = player;
    j["scored"] = scored;
    j["information_token"] = information_token;
    j["color"] = color;
    j["rank"] = rank;
    j["reveal_bitmask"] = reveal_bitmask;
    j["newly_revealed_bitmask"] = newly_revealed_bitmask;
    j["deal_to_player"] = deal_to_player;
    return j;
}

HanabiHistoryItem HanabiHistoryItem::fromJSON(const json& j) {
    HanabiHistoryItem item = HanabiHistoryItem(HanabiMove::fromJSON(j.at("move")));

    item.player = j.at("player").get<int8_t>();
    item.scored = j.at("scored").get<bool>();
    item.information_token = j.at("information_token").get<bool>();
    item.color = j.at("color").get<int8_t>();
    item.rank = j.at("rank").get<int8_t>();
    item.reveal_bitmask = j.at("reveal_bitmask").get<uint8_t>();
    item.newly_revealed_bitmask = j.at("newly_revealed_bitmask").get<uint8_t>();
    item.deal_to_player = j.at("deal_to_player").get<int8_t>();

    return item;
}

}  // namespace hanabi_learning_env
