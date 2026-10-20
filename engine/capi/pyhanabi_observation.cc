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

#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "hanabi_lib/canonical_encoders.h"
#include "hanabi_lib/hanabi_history_item.h"
#include "hanabi_lib/hanabi_move.h"
#include "hanabi_lib/hanabi_observation.h"
#include "hanabi_lib/hanabi_state.h"
#include "hanabi_lib/observation_encoder.h"
#include "hanabi_lib/util.h"
#include "pyhanabi.h"
#include "pyhanabi_internal.h"

using pyhanabi_internal::CopyCard;
using pyhanabi_internal::InRange;
using pyhanabi_internal::Observation;

extern "C" {

/* Wrapper definitions for HanabiObservation. */
bool NewObservation(pyhanabi_state_t* state, int player, pyhanabi_observation_t* observation) {
  REQUIRE(state != nullptr);
  REQUIRE(state->state != nullptr);
  REQUIRE(observation != nullptr);
  const auto& hanabi_state = *static_cast<hanabi_learning_env::HanabiState*>(state->state);
  if (player < 0 || player >= hanabi_state.ParentGame()->NumPlayers()) return false;
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
  return InRange(pid, hands.size()) ? static_cast<int>(hands[pid].Cards().size()) : -1;
}

bool ObsGetHandCard(pyhanabi_observation_t* observation, int pid, int index,
                    pyhanabi_card_t* card) {
  const auto& hands = Observation(observation).Hands();
  return InRange(pid, hands.size()) && CopyCard(hands[pid].Cards(), index, card);
}

bool ObsGetHandCardKnowledge(pyhanabi_observation_t* observation, int pid, int index,
                             pyhanabi_card_knowledge_t* knowledge) {
  REQUIRE(knowledge != nullptr);
  const auto& hands = Observation(observation).Hands();
  if (!InRange(pid, hands.size()) || !InRange(index, hands[pid].Knowledge().size())) return false;
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
  return InRange(color, fireworks.size()) ? fireworks[color] : -1;
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
  if (!InRange(index, last_moves.size())) return false;
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
  if (!InRange(index, legal_moves.size())) return false;
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
  if (type != hanabi_learning_env::ObservationEncoder::Type::kCanonical) return false;
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

}  // extern "C"
