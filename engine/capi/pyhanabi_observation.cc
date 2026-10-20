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
//
// C API functions on observations and observation encoders (moved out of
// pyhanabi.cc); declared in pyhanabi.h.

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <format>
#include <mdspan>
#include <string>
#include <vector>

#include "hanabi_lib/canonical_encoders.h"
#include "hanabi_lib/hanabi_history_item.h"
#include "hanabi_lib/hanabi_move.h"
#include "hanabi_lib/hanabi_observation.h"
#include "hanabi_lib/hanabi_state.h"
#include "hanabi_lib/observation_encoder.h"
#include "hanabi_lib/observation_features.h"
#include "hanabi_lib/util.h"
#include "pyhanabi.h"
#include "pyhanabi_internal.h"

using pyhanabi_internal::CheckIndex;
using pyhanabi_internal::CopyCard;
using pyhanabi_internal::Fail;
using pyhanabi_internal::FailInt;
using pyhanabi_internal::Observation;

namespace {

const hanabi_learning_env::HanabiGame& Game(const pyhanabi_observation_t* observation) {
  return *Observation(observation).ParentGame();
}

}  // namespace

extern "C" {

/* Wrapper definitions for HanabiObservation. */
bool NewObservation(pyhanabi_state_t* state, int player, pyhanabi_observation_t* observation) {
  REQUIRE(state != nullptr);
  REQUIRE(state->state != nullptr);
  REQUIRE(observation != nullptr);
  const auto& hanabi_state = *static_cast<hanabi_learning_env::HanabiState*>(state->state);
  if (!CheckIndex(player, hanabi_state.ParentGame()->NumPlayers(), "player")) return false;
  observation->observation = new hanabi_learning_env::HanabiObservation(hanabi_state, player);
  return true;
}

void DeleteObservation(pyhanabi_observation_t* observation) {
  REQUIRE(observation != nullptr);
  REQUIRE(observation->observation != nullptr);
  delete reinterpret_cast<hanabi_learning_env::HanabiObservation*>(observation->observation);
  observation->observation = nullptr;
}

char* ObsToString(pyhanabi_observation_t* observation) {
  REQUIRE(observation != nullptr);
  REQUIRE(observation->observation != nullptr);
  std::string str =
      reinterpret_cast<hanabi_learning_env::HanabiObservation*>(observation->observation)
          ->ToString();
  return strdup(str.c_str());
}

int ObsCurPlayerOffset(pyhanabi_observation_t* observation) {
  REQUIRE(observation != nullptr);
  REQUIRE(observation->observation != nullptr);
  return reinterpret_cast<hanabi_learning_env::HanabiObservation*>(observation->observation)
      ->CurPlayerOffset();
}

int ObsNumPlayers(pyhanabi_observation_t* observation) {
  REQUIRE(observation != nullptr);
  REQUIRE(observation->observation != nullptr);
  return reinterpret_cast<hanabi_learning_env::HanabiObservation*>(observation->observation)
      ->ParentGame()
      ->NumPlayers();
}

int ObsGetHandSize(pyhanabi_observation_t* observation, int pid) {
  const auto& hands = Observation(observation).Hands();
  return CheckIndex(pid, hands.size(), "player offset")
             ? static_cast<int>(hands[pid].Cards().size())
             : -1;
}

bool ObsGetHandCard(pyhanabi_observation_t* observation, int pid, int index,
                    pyhanabi_card_t* card) {
  const auto& hands = Observation(observation).Hands();
  return CheckIndex(pid, hands.size(), "player offset") &&
         CopyCard(hands[pid].Cards(), index, card);
}

bool ObsGetHandCardKnowledge(pyhanabi_observation_t* observation, int pid, int index,
                             pyhanabi_card_knowledge_t* knowledge) {
  REQUIRE(knowledge != nullptr);
  const auto& hands = Observation(observation).Hands();
  if (!CheckIndex(pid, hands.size(), "player offset") ||
      !CheckIndex(index, hands[pid].Knowledge().size(), "card index")) {
    return false;
  }
  knowledge->knowledge = &hands[pid].Knowledge()[index];
  return true;
}

int ObsDiscardPileSize(pyhanabi_observation_t* observation) {
  return reinterpret_cast<hanabi_learning_env::HanabiObservation*>(observation->observation)
      ->DiscardPile()
      .size();
}

bool ObsGetDiscard(pyhanabi_observation_t* observation, int index, pyhanabi_card_t* card) {
  return CopyCard(Observation(observation).DiscardPile(), index, card);
}

int ObsFireworks(pyhanabi_observation_t* observation, int color) {
  const auto& fireworks = Observation(observation).Fireworks();
  return CheckIndex(color, fireworks.size(), "color") ? fireworks[color] : -1;
}

int ObsDeckSize(pyhanabi_observation_t* observation) {
  REQUIRE(observation != nullptr);
  REQUIRE(observation->observation != nullptr);
  return reinterpret_cast<hanabi_learning_env::HanabiObservation*>(observation->observation)
      ->DeckSize();
}

int ObsNumLastMoves(pyhanabi_observation_t* observation) {
  REQUIRE(observation != nullptr);
  REQUIRE(observation->observation != nullptr);
  return reinterpret_cast<hanabi_learning_env::HanabiObservation*>(observation->observation)
      ->LastMoves()
      .size();
}

bool ObsGetLastMove(pyhanabi_observation_t* observation, int index, pyhanabi_history_item_t* item) {
  REQUIRE(item != nullptr);
  const auto& last_moves = Observation(observation).LastMoves();
  if (!CheckIndex(index, last_moves.size(), "last move index")) return false;
  item->item = new hanabi_learning_env::HanabiHistoryItem(last_moves[index]);
  return true;
}

int ObsInformationTokens(pyhanabi_observation_t* observation) {
  REQUIRE(observation != nullptr);
  REQUIRE(observation->observation != nullptr);
  return reinterpret_cast<hanabi_learning_env::HanabiObservation*>(observation->observation)
      ->InformationTokens();
}

int ObsLifeTokens(pyhanabi_observation_t* observation) {
  REQUIRE(observation != nullptr);
  REQUIRE(observation->observation != nullptr);
  return reinterpret_cast<hanabi_learning_env::HanabiObservation*>(observation->observation)
      ->LifeTokens();
}

int ObsNumLegalMoves(pyhanabi_observation_t* observation) {
  REQUIRE(observation != nullptr);
  REQUIRE(observation->observation != nullptr);
  return reinterpret_cast<hanabi_learning_env::HanabiObservation*>(observation->observation)
      ->LegalMoves()
      .size();
}

bool ObsGetLegalMove(pyhanabi_observation_t* observation, int index, pyhanabi_move_t* move) {
  REQUIRE(move != nullptr);
  const auto& legal_moves = Observation(observation).LegalMoves();
  if (!CheckIndex(index, legal_moves.size(), "legal move index")) return false;
  move->move = new hanabi_learning_env::HanabiMove(legal_moves[index]);
  return true;
}

bool ObsCardPlayableOnFireworks(const pyhanabi_observation_t* observation, int color, int rank) {
  return reinterpret_cast<const hanabi_learning_env::HanabiObservation*>(observation->observation)
      ->CardPlayableOnFireworks(color, rank);
}

bool NewObservationEncoder(pyhanabi_observation_encoder_t* encoder, pyhanabi_game_t* game,
                           int type) {
  REQUIRE(encoder != nullptr);
  REQUIRE(game != nullptr);
  REQUIRE(game->game != nullptr);
  if (type != hanabi_learning_env::ObservationEncoder::Type::kCanonical) {
    return Fail(std::format("unknown observation encoder type {} (0: canonical)", type));
  }
  encoder->encoder = new hanabi_learning_env::CanonicalObservationEncoder(
      static_cast<hanabi_learning_env::HanabiGame*>(game->game));
  return true;
}

void DeleteObservationEncoder(pyhanabi_observation_encoder_t* encoder) {
  REQUIRE(encoder != nullptr);
  REQUIRE(encoder->encoder != nullptr);
  delete reinterpret_cast<hanabi_learning_env::ObservationEncoder*>(encoder->encoder);
  encoder->encoder = nullptr;
}

char* ObservationShape(pyhanabi_observation_encoder_t* encoder) {
  REQUIRE(encoder != nullptr);
  REQUIRE(encoder->encoder != nullptr);
  auto obs_enc = reinterpret_cast<hanabi_learning_env::ObservationEncoder*>(encoder->encoder);
  std::vector<int> shape = obs_enc->Shape();
  std::string shape_str = "";
  for (int i = 0; i < static_cast<int>(shape.size()); i++) {
    shape_str += std::to_string(shape[i]);
    if (i != static_cast<int>(shape.size()) - 1) {
      shape_str += ",";
    }
  }
  return strdup(shape_str.c_str());
}

char* EncodeObservation(pyhanabi_observation_encoder_t* encoder,
                        pyhanabi_observation_t* observation) {
  REQUIRE(encoder != nullptr);
  REQUIRE(encoder->encoder != nullptr);
  REQUIRE(observation != nullptr);
  REQUIRE(observation->observation != nullptr);
  auto obs_enc = reinterpret_cast<hanabi_learning_env::ObservationEncoder*>(encoder->encoder);
  auto obs = reinterpret_cast<hanabi_learning_env::HanabiObservation*>(observation->observation);
  std::vector<int> encoding = obs_enc->Encode(*obs);
  std::string obs_str = "";
  for (int i = 0; i < static_cast<int>(encoding.size()); i++) {
    obs_str += (encoding[i] ? "1" : "0");
    if (i != static_cast<int>(encoding.size()) - 1) {
      obs_str += ",";
    }
  }
  return strdup(obs_str.c_str());
}

/* Observation buffers. */

bool pyhanabi_observation_hands(const pyhanabi_observation_t* observation, int32_t* hand_sizes,
                                int8_t* colors, int8_t* ranks, int32_t num_players,
                                int32_t hand_size) {
  REQUIRE(hand_sizes != nullptr && colors != nullptr && ranks != nullptr);
  const auto& game = Game(observation);
  if (num_players != game.NumPlayers() || hand_size != game.HandSize()) {
    return Fail(std::format("buffers are for {} players x {} cards; the game has {} x {}",
                            num_players, hand_size, game.NumPlayers(), game.HandSize()));
  }
  pyhanabi_internal::WriteHands(Observation(observation).Hands(), hand_size, hand_sizes, colors,
                                ranks);
  return true;
}

bool pyhanabi_observation_card_knowledge(const pyhanabi_observation_t* observation,
                                         int8_t* hinted_colors, int8_t* hinted_ranks,
                                         uint8_t* plausible, int32_t num_players, int32_t hand_size,
                                         int32_t num_colors, int32_t num_ranks) {
  REQUIRE(hinted_colors != nullptr && hinted_ranks != nullptr && plausible != nullptr);
  const auto& game = Game(observation);
  if (num_players != game.NumPlayers() || hand_size != game.HandSize() ||
      num_colors != game.NumColors() || num_ranks != game.NumRanks()) {
    return Fail(std::format(
        "buffers are for {} players x {} cards x {} colors x {} ranks; the game has {} x {} x {} "
        "x {}",
        num_players, hand_size, num_colors, num_ranks, game.NumPlayers(), game.HandSize(),
        game.NumColors(), game.NumRanks()));
  }
  const std::mdspan color_grid(hinted_colors, num_players, hand_size);  // [player, position]
  const std::mdspan rank_grid(hinted_ranks, num_players, hand_size);
  std::fill(hinted_colors, hinted_colors + color_grid.size(), -1);
  std::fill(hinted_ranks, hinted_ranks + rank_grid.size(), -1);
  const auto& hands = Observation(observation).Hands();
  for (int offset = 0; offset < num_players; ++offset) {
    const auto& knowledge = hands[offset].Knowledge();
    for (size_t index = 0; index < hands[offset].Cards().size(); ++index) {
      color_grid[offset, index] = static_cast<int8_t>(knowledge[index].Color());
      rank_grid[offset, index] = static_cast<int8_t>(knowledge[index].Rank());
    }
  }
  hanabi_learning_env::PlausibleCards(Observation(observation), plausible);
  return true;
}

int32_t pyhanabi_observation_discard_pile(const pyhanabi_observation_t* observation, int8_t* colors,
                                          int8_t* ranks, int32_t capacity) {
  REQUIRE(colors != nullptr && ranks != nullptr);
  const auto& discards = Observation(observation).DiscardPile();
  if (capacity < 0 || discards.size() > static_cast<size_t>(capacity)) {
    return FailInt(
        std::format("capacity {} is below the {} discarded cards", capacity, discards.size()));
  }
  for (size_t index = 0; index < discards.size(); ++index) {
    colors[index] = static_cast<int8_t>(discards[index].Color());
    ranks[index] = static_cast<int8_t>(discards[index].Rank());
  }
  return static_cast<int32_t>(discards.size());
}

bool pyhanabi_observation_fireworks(const pyhanabi_observation_t* observation, int32_t* heights,
                                    int32_t num_colors) {
  REQUIRE(heights != nullptr);
  const auto& fireworks = Observation(observation).Fireworks();
  if (num_colors < 0 || static_cast<size_t>(num_colors) != fireworks.size()) {
    return Fail(std::format("num_colors {} is not the game's {}", num_colors, fireworks.size()));
  }
  std::copy(fireworks.begin(), fireworks.end(), heights);
  return true;
}

int32_t pyhanabi_observation_legal_move_uids(const pyhanabi_observation_t* observation,
                                             int32_t* uids, int32_t capacity) {
  REQUIRE(uids != nullptr);
  const auto& game = Game(observation);
  const auto& moves = Observation(observation).LegalMoves();
  if (capacity < 0 || moves.size() > static_cast<size_t>(capacity)) {
    return FailInt(std::format("capacity {} is below the {} legal moves", capacity, moves.size()));
  }
  for (size_t index = 0; index < moves.size(); ++index) {
    uids[index] = game.GetMoveUid(moves[index]);
  }
  return static_cast<int32_t>(moves.size());
}

bool pyhanabi_observation_unseen_card_counts(const pyhanabi_observation_t* observation,
                                             int32_t* counts, int32_t counts_size) {
  REQUIRE(counts != nullptr);
  const auto& game = Game(observation);
  if (counts_size != game.NumColors() * game.NumRanks()) {
    return Fail(std::format("counts_size {} is not num_colors * num_ranks ({})", counts_size,
                            game.NumColors() * game.NumRanks()));
  }
  hanabi_learning_env::UnseenCardCounts(Observation(observation), counts);
  return true;
}

bool pyhanabi_observation_encode(const pyhanabi_observation_encoder_t* encoder,
                                 const pyhanabi_observation_t* observation, uint8_t* bits,
                                 int32_t size) {
  REQUIRE(encoder != nullptr);
  REQUIRE(encoder->encoder != nullptr);
  REQUIRE(bits != nullptr);
  // NewObservationEncoder only creates canonical encoders.
  const auto& obs_encoder =
      *static_cast<const hanabi_learning_env::CanonicalObservationEncoder*>(encoder->encoder);
  if (size != obs_encoder.Length()) {
    return Fail(std::format("size {} is not the encoding length {}", size, obs_encoder.Length()));
  }
  obs_encoder.EncodeInto(Observation(observation), bits);
  return true;
}

bool pyhanabi_state_encode_observation(const pyhanabi_observation_encoder_t* encoder,
                                       const pyhanabi_state_t* state, int32_t player, uint8_t* bits,
                                       int32_t size) {
  REQUIRE(encoder != nullptr);
  REQUIRE(encoder->encoder != nullptr);
  REQUIRE(state != nullptr);
  REQUIRE(state->state != nullptr);
  REQUIRE(bits != nullptr);
  const auto& hanabi_state = *static_cast<const hanabi_learning_env::HanabiState*>(state->state);
  if (!CheckIndex(player, hanabi_state.ParentGame()->NumPlayers(), "player")) return false;
  // NewObservationEncoder only creates canonical encoders.
  const auto& obs_encoder =
      *static_cast<const hanabi_learning_env::CanonicalObservationEncoder*>(encoder->encoder);
  if (size != obs_encoder.Length()) {
    return Fail(std::format("size {} is not the encoding length {}", size, obs_encoder.Length()));
  }
  obs_encoder.EncodeInto(hanabi_learning_env::HanabiObservation(hanabi_state, player), bits);
  return true;
}

}  // extern "C"
