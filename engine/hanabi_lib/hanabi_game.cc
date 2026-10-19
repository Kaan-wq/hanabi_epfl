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

#include "hanabi_game.h"

#include <limits>
#include <nlohmann/json.hpp>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>

#include "json_checks.h"
#include "util.h"
using json = nlohmann::json;

namespace hanabi_learning_env {

namespace {
// Constants.
const int kDefaultPlayers = 2;
const int kInformationTokens = 8;
const int kLifeTokens = 3;
const bool kDefaultRandomStart = false;

// Thrown, instead of aborting, so the C API can report invalid parameters.
void CheckParameter(bool valid, const char* message) {
  if (!valid) {
    throw std::invalid_argument(message);
  }
}
}  // namespace

HanabiGame::HanabiGame(const std::unordered_map<std::string, std::string>& params) {
  params_ = params;
  num_players_ = ParameterValue<int>(params_, "players", kDefaultPlayers);
  CheckParameter(num_players_ >= MinPlayers() && num_players_ <= MaxPlayers(),
                 "players must be in [2, 5]");
  num_colors_ = ParameterValue<int>(params_, "colors", kMaxNumColors);
  CheckParameter(num_colors_ > 0 && num_colors_ <= kMaxNumColors, "colors must be in [1, 5]");
  num_ranks_ = ParameterValue<int>(params_, "ranks", kMaxNumRanks);
  CheckParameter(num_ranks_ > 0 && num_ranks_ <= kMaxNumRanks, "ranks must be in [1, 5]");
  hand_size_ = ParameterValue<int>(params_, "hand_size", HandSizeFromRules());
  CheckParameter(hand_size_ > 0, "hand_size must be positive");
  max_information_tokens_ =
      ParameterValue<int>(params_, "max_information_tokens", kInformationTokens);
  CheckParameter(max_information_tokens_ >= 0, "max_information_tokens must be >= 0");
  max_life_tokens_ = ParameterValue<int>(params_, "max_life_tokens", kLifeTokens);
  CheckParameter(max_life_tokens_ > 0, "max_life_tokens must be positive");
  seed_ = ParameterValue<int>(params_, "seed", -1);
  random_start_player_ = ParameterValue<bool>(params_, "random_start_player", kDefaultRandomStart);
  const int observation_type =
      ParameterValue<int>(params_, "observation_type", AgentObservationType::kCardKnowledge);
  CheckParameter(observation_type >= kMinimal && observation_type <= kSeer,
                 "observation_type must be in [0, 2]");
  observation_type_ = AgentObservationType(observation_type);
  while (seed_ == -1) {
    seed_ = std::random_device()();
  }
  rng_.seed(seed_);

  // Work out number of cards per color, and check deck size is large enough.
  cards_per_color_ = 0;
  for (int rank = 0; rank < num_ranks_; ++rank) {
    cards_per_color_ += NumberCardInstances(0, rank);
  }
  CheckParameter(hand_size_ * num_players_ <= cards_per_color_ * num_colors_,
                 "the hands need more cards than the deck has");

  // Build static list of moves.
  for (int uid = 0; uid < MaxMoves(); ++uid) {
    moves_.push_back(ConstructMove(uid));
  }
  for (int uid = 0; uid < MaxChanceOutcomes(); ++uid) {
    chance_outcomes_.push_back(ConstructChanceOutcome(uid));
  }
}

int HanabiGame::MaxMoves() const {
  return MaxDiscardMoves() + MaxPlayMoves() + MaxRevealColorMoves() + MaxRevealRankMoves();
}

int HanabiGame::GetMoveUid(HanabiMove move) const {
  return GetMoveUid(move.MoveType(), move.CardIndex(), move.TargetOffset(), move.Color(),
                    move.Rank());
}

// MB: RETURN and DEAL_SPECIFIC aren't here. Problem?
int HanabiGame::GetMoveUid(HanabiMove::Type move_type, int card_index, int target_offset, int color,
                           int rank) const {
  switch (move_type) {
    case HanabiMove::kDiscard:
      return card_index;
    case HanabiMove::kPlay:
      return MaxDiscardMoves() + card_index;
    case HanabiMove::kRevealColor:
      return MaxDiscardMoves() + MaxPlayMoves() + (target_offset - 1) * NumColors() + color;
    case HanabiMove::kRevealRank:
      return MaxDiscardMoves() + MaxPlayMoves() + MaxRevealColorMoves() +
             (target_offset - 1) * NumRanks() + rank;
    default:
      return -1;
  }
}

int HanabiGame::MaxChanceOutcomes() const { return NumColors() * NumRanks(); }

int HanabiGame::GetChanceOutcomeUid(HanabiMove move) const {
  if (move.MoveType() != HanabiMove::kDeal) {
    return -1;
  }
  return move.Color() * NumRanks() + move.Rank();
}

HanabiMove HanabiGame::PickRandomChance(
    const std::pair<std::vector<HanabiMove>, std::vector<double>>& chance_outcomes) const {
  std::discrete_distribution<std::mt19937::result_type> dist(chance_outcomes.second.begin(),
                                                             chance_outcomes.second.end());
  return chance_outcomes.first[dist(rng_)];
}

std::unordered_map<std::string, std::string> HanabiGame::Parameters() const {
  return {{"players", std::to_string(num_players_)},
          {"colors", std::to_string(NumColors())},
          {"ranks", std::to_string(NumRanks())},
          {"hand_size", std::to_string(HandSize())},
          {"max_information_tokens", std::to_string(MaxInformationTokens())},
          {"max_life_tokens", std::to_string(MaxLifeTokens())},
          {"seed", std::to_string(seed_)},
          {"random_start_player", random_start_player_ ? "true" : "false"},
          {"observation_type", std::to_string(observation_type_)}};
}

int HanabiGame::NumberCardInstances(int color, int rank) const {
  if (color < 0 || color >= NumColors() || rank < 0 || rank >= NumRanks()) {
    return 0;
  }
  if (rank == 0) {
    return 3;
  } else if (rank == NumRanks() - 1) {
    return 1;
  }
  return 2;
}

int HanabiGame::GetSampledStartPlayer() const {
  if (random_start_player_) {
    std::uniform_int_distribution<std::mt19937::result_type> dist(0, num_players_ - 1);
    return dist(rng_);
  }
  return 0;
}

int HanabiGame::HandSizeFromRules() const {
  if (num_players_ < 4) {
    return 5;
  }
  return 4;
}

// Uid mapping.  h=hand_size, p=num_players, c=colors, r=ranks
// 0, h-1: discard
// h, 2h-1: play
// 2h, 2h+(p-1)c-1: color hint
// 2h+(p-1)c, 2h+(p-1)c+(p-1)r-1: rank hint
HanabiMove HanabiGame::ConstructMove(int uid) const {
  if (uid < 0 || uid >= MaxMoves()) {
    return HanabiMove(HanabiMove::kInvalid, /*card_index=*/-1,
                      /*target_offset=*/-1, /*color=*/-1, /*rank=*/-1);
  }
  if (uid < MaxDiscardMoves()) {
    return HanabiMove(HanabiMove::kDiscard, /*card_index=*/uid,
                      /*target_offset=*/-1, /*color=*/-1, /*rank=*/-1);
  }
  uid -= MaxDiscardMoves();
  if (uid < MaxPlayMoves()) {
    return HanabiMove(HanabiMove::kPlay, /*card_index=*/uid,
                      /*target_offset=*/-1, /*color=*/-1, /*rank=*/-1);
  }
  uid -= MaxPlayMoves();
  if (uid < MaxRevealColorMoves()) {
    return HanabiMove(HanabiMove::kRevealColor, /*card_index=*/-1,
                      /*target_offset=*/1 + uid / NumColors(),
                      /*color=*/uid % NumColors(), /*rank=*/-1);
  }
  uid -= MaxRevealColorMoves();
  return HanabiMove(HanabiMove::kRevealRank, /*card_index=*/-1,
                    /*target_offset=*/1 + uid / NumRanks(),
                    /*color=*/-1, /*rank=*/uid % NumRanks());
}

HanabiMove HanabiGame::ConstructChanceOutcome(int uid) const {
  if (uid < 0 || uid >= MaxChanceOutcomes()) {
    return HanabiMove(HanabiMove::kInvalid, /*card_index=*/-1,
                      /*target_offset=*/-1, /*color=*/-1, /*rank=*/-1);
  }
  return HanabiMove(HanabiMove::kDeal, /*card_index=*/-1,
                    /*target_offset=*/-1,
                    /*color=*/uid / NumRanks() % NumColors(),
                    /*rank=*/uid % NumRanks());
}

// =========================== Serialization + Deserialization ===========================

// Serialization
json HanabiGame::toJSON() const {
  json j;

  // Serialize simple types
  j["num_players"] = num_players_;
  j["num_colors"] = num_colors_;
  j["num_ranks"] = num_ranks_;
  j["hand_size"] = hand_size_;
  j["max_information_tokens"] = max_information_tokens_;
  j["max_life_tokens"] = max_life_tokens_;
  j["seed"] = seed_;
  j["random_start_player"] = random_start_player_;
  j["observation_type"] = static_cast<int>(observation_type_);
  j["cards_per_color"] = cards_per_color_;

  // Serialize params_
  j["params"] = params_;

  // Serialize moves_ and chance_outcomes_
  j["moves"] = json::array();
  for (const auto& move : moves_) {
    j["moves"].push_back(move.toJSON());
  }
  j["chance_outcomes"] = json::array();
  for (const auto& outcome : chance_outcomes_) {
    j["chance_outcomes"].push_back(outcome.toJSON());
  }

  // Serialize rng_ state
  std::stringstream ss;
  ss << rng_;
  j["rng_state"] = ss.str();

  return j;
}

// Deserialization
HanabiGame HanabiGame::fromJSON(const nlohmann::json& j) {
  // Ranges are checked by the constructor below.
  constexpr int kMin = std::numeric_limits<int>::min();
  constexpr int kMax = std::numeric_limits<int>::max();
  const int num_players = JsonInt(j, "num_players", kMin, kMax);
  const int num_colors = JsonInt(j, "num_colors", kMin, kMax);
  const int num_ranks = JsonInt(j, "num_ranks", kMin, kMax);
  const int hand_size = JsonInt(j, "hand_size", kMin, kMax);
  const int max_information_tokens = JsonInt(j, "max_information_tokens", kMin, kMax);
  const int max_life_tokens = JsonInt(j, "max_life_tokens", kMin, kMax);
  const int seed = JsonInt(j, "seed", kMin, kMax);
  const bool random_start_player = JsonBool(j, "random_start_player");
  const int observation_type = JsonInt(j, "observation_type", kMin, kMax);
  const int cards_per_color = JsonInt(j, "cards_per_color", kMin, kMax);

  std::unordered_map<std::string, std::string> params;
  const json& params_json = JsonField(j, "params");
  if (!params_json.is_object()) {
    throw std::invalid_argument("'params' must be an object");
  }
  for (auto it = params_json.begin(); it != params_json.end(); ++it) {
    if (!it.value().is_string()) {
      throw std::invalid_argument("'params' values must be strings");
    }
    params[it.key()] = it.value().get<std::string>();
  }

  // Built from the checked fields; "params" is only stored, as in the original.
  HanabiGame game({{"players", std::to_string(num_players)},
                   {"colors", std::to_string(num_colors)},
                   {"ranks", std::to_string(num_ranks)},
                   {"hand_size", std::to_string(hand_size)},
                   {"max_information_tokens", std::to_string(max_information_tokens)},
                   {"max_life_tokens", std::to_string(max_life_tokens)},
                   {"seed", std::to_string(seed)},
                   {"random_start_player", random_start_player ? "true" : "false"},
                   {"observation_type", std::to_string(observation_type)}});
  game.params_ = std::move(params);
  game.seed_ = seed;
  CheckRange(cards_per_color, game.cards_per_color_, game.cards_per_color_, "'cards_per_color'");

  // The move tables are derived from the parameters; they must match.
  const auto check_moves = [](const json& moves_json, const std::vector<HanabiMove>& expected,
                              const char* key) {
    if (moves_json.size() != expected.size()) {
      throw std::invalid_argument(std::string("'") + key + "' does not match the game");
    }
    for (size_t i = 0; i < expected.size(); ++i) {
      if (!(HanabiMove::fromJSON(moves_json[i]) == expected[i])) {
        throw std::invalid_argument(std::string("'") + key + "' does not match the game");
      }
    }
  };
  check_moves(JsonArray(j, "moves"), game.moves_, "moves");
  check_moves(JsonArray(j, "chance_outcomes"), game.chance_outcomes_, "chance_outcomes");

  std::stringstream ss(JsonString(j, "rng_state"));
  ss >> game.rng_;
  if (ss.fail()) {
    throw std::invalid_argument("'rng_state' is not a valid mt19937 state");
  }

  return game;
}

// =======================================================================================

}  // namespace hanabi_learning_env
