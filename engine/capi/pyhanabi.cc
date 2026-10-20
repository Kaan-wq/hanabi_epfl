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

#include "pyhanabi.h"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <nlohmann/json.hpp>
#include <string>
#include <unordered_map>
#include <vector>
using json = nlohmann::json;

#include "hanabi_lib/hanabi_card.h"
#include "hanabi_lib/hanabi_game.h"
#include "hanabi_lib/hanabi_history_item.h"
#include "hanabi_lib/hanabi_move.h"
#include "hanabi_lib/hanabi_observation.h"
#include "hanabi_lib/hanabi_state.h"
#include "hanabi_lib/hand_sampler.h"
#include "hanabi_lib/util.h"
#include "pyhanabi_internal.h"

namespace {

using pyhanabi_internal::CopyCard;
using pyhanabi_internal::InRange;

// Serializes `object` into a malloc'd C string, or returns nullptr on failure.
template <typename T>
char* ToJsonString(const T& object) {
  try {
    return strdup(object.toJSON().dump().c_str());
  } catch (const std::exception&) {
    return nullptr;
  }
}

// Builds a T from `json_str` with T::fromJSON(json, extra_args...) and, on
// success, replaces the object `*handle` points to (deleting the old one).
// On any error, leaves `*handle` untouched and returns false.
template <typename T, typename... Args>
bool ReplaceFromJson(const char* json_str, void** handle, const Args&... extra_args) {
  try {
    auto parsed = std::make_unique<T>(T::fromJSON(json::parse(json_str), extra_args...));
    delete static_cast<T*>(*handle);
    *handle = parsed.release();
    return true;
  } catch (const std::exception&) {
    return false;
  }
}

bool FitsInt8(int value) { return value >= INT8_MIN && value <= INT8_MAX; }

// Builds a move for the Get*Move functions; false if a value does not fit.
bool NewMove(hanabi_learning_env::HanabiMove::Type type, int card_index, int target_offset,
             int color, int rank, pyhanabi_move_t* move) {
  REQUIRE(move != nullptr);
  if (!FitsInt8(card_index) || !FitsInt8(target_offset) || !FitsInt8(color) || !FitsInt8(rank)) {
    return false;
  }
  move->move = new hanabi_learning_env::HanabiMove(type, card_index, target_offset, color, rank);
  return true;
}

}  // namespace

extern "C" {

/* Helpers. */

void DeleteString(char* str) { free(str); }

/* Wrapper definitions for HanabiCard. */
int CardValid(pyhanabi_card_t* card) { return card->color >= 0; }

/* Wrapper definitions for HanabiCardKnowledge. */
char* CardKnowledgeToString(pyhanabi_card_knowledge_t* knowledge) {
  REQUIRE(knowledge != nullptr);
  REQUIRE(knowledge->knowledge != nullptr);
  std::string str =
      reinterpret_cast<const hanabi_learning_env::HanabiHand::CardKnowledge*>(knowledge->knowledge)
          ->ToString();
  return strdup(str.c_str());
}

int ColorWasHinted(pyhanabi_card_knowledge_t* knowledge) {
  REQUIRE(knowledge != nullptr);
  REQUIRE(knowledge->knowledge != nullptr);
  return reinterpret_cast<const hanabi_learning_env::HanabiHand::CardKnowledge*>(
             knowledge->knowledge)
      ->ColorHinted();
}

int KnownColor(pyhanabi_card_knowledge_t* knowledge) {
  REQUIRE(knowledge != nullptr);
  REQUIRE(knowledge->knowledge != nullptr);
  return reinterpret_cast<const hanabi_learning_env::HanabiHand::CardKnowledge*>(
             knowledge->knowledge)
      ->Color();
}

int ColorIsPlausible(pyhanabi_card_knowledge_t* knowledge, int color) {
  REQUIRE(knowledge != nullptr);
  REQUIRE(knowledge->knowledge != nullptr);
  const auto* card_knowledge =
      static_cast<const hanabi_learning_env::HanabiHand::CardKnowledge*>(knowledge->knowledge);
  if (color < 0 || color >= card_knowledge->NumColors()) return -1;
  return card_knowledge->ColorPlausible(color);
}

int RankWasHinted(pyhanabi_card_knowledge_t* knowledge) {
  REQUIRE(knowledge != nullptr);
  REQUIRE(knowledge->knowledge != nullptr);
  return reinterpret_cast<const hanabi_learning_env::HanabiHand::CardKnowledge*>(
             knowledge->knowledge)
      ->RankHinted();
}

int KnownRank(pyhanabi_card_knowledge_t* knowledge) {
  REQUIRE(knowledge != nullptr);
  REQUIRE(knowledge->knowledge != nullptr);
  return reinterpret_cast<const hanabi_learning_env::HanabiHand::CardKnowledge*>(
             knowledge->knowledge)
      ->Rank();
}

int RankIsPlausible(pyhanabi_card_knowledge_t* knowledge, int rank) {
  REQUIRE(knowledge != nullptr);
  REQUIRE(knowledge->knowledge != nullptr);
  const auto* card_knowledge =
      static_cast<const hanabi_learning_env::HanabiHand::CardKnowledge*>(knowledge->knowledge);
  if (rank < 0 || rank >= card_knowledge->NumRanks()) return -1;
  return card_knowledge->RankPlausible(rank);
}

/* Wrapper definitions for HanabiMove. */
void DeleteMoveList(void* movelist) {
  delete reinterpret_cast<std::vector<hanabi_learning_env::HanabiMove>*>(movelist);
}

int NumMoves(void* movelist) {
  return reinterpret_cast<std::vector<hanabi_learning_env::HanabiMove>*>(movelist)->size();
}

bool GetMove(void* movelist, int index, pyhanabi_move_t* move) {
  REQUIRE(movelist != nullptr);
  REQUIRE(move != nullptr);
  const auto& moves = *static_cast<std::vector<hanabi_learning_env::HanabiMove>*>(movelist);
  if (!InRange(index, moves.size())) return false;
  move->move = new hanabi_learning_env::HanabiMove(moves[index]);
  return true;
}

void DeleteMove(pyhanabi_move_t* move) {
  REQUIRE(move != nullptr);
  REQUIRE(move->move != nullptr);
  delete reinterpret_cast<hanabi_learning_env::HanabiMove*>(move->move);
  move->move = nullptr;
}

char* MoveToString(pyhanabi_move_t* move) {
  REQUIRE(move != nullptr);
  REQUIRE(move->move != nullptr);
  std::string str =
      reinterpret_cast<const hanabi_learning_env::HanabiMove*>(move->move)->ToString();
  return strdup(str.c_str());
}

int MoveType(pyhanabi_move_t* move) {
  return reinterpret_cast<const hanabi_learning_env::HanabiMove*>(move->move)->MoveType();
}

int CardIndex(pyhanabi_move_t* move) {
  return reinterpret_cast<const hanabi_learning_env::HanabiMove*>(move->move)->CardIndex();
}

int TargetOffset(pyhanabi_move_t* move) {
  return reinterpret_cast<const hanabi_learning_env::HanabiMove*>(move->move)->TargetOffset();
}

int MoveColor(pyhanabi_move_t* move) {
  return reinterpret_cast<const hanabi_learning_env::HanabiMove*>(move->move)->Color();
}

int MoveRank(pyhanabi_move_t* move) {
  return reinterpret_cast<const hanabi_learning_env::HanabiMove*>(move->move)->Rank();
}

bool GetDiscardMove(int card_index, pyhanabi_move_t* move) {
  return NewMove(hanabi_learning_env::HanabiMove::kDiscard, card_index, -1, -1, -1, move);
}

bool GetPlayMove(int card_index, pyhanabi_move_t* move) {
  return NewMove(hanabi_learning_env::HanabiMove::kPlay, card_index, -1, -1, -1, move);
}

// MB: Added Move
bool GetReturnMove(int card_index, int player, pyhanabi_move_t* move) {
  return NewMove(hanabi_learning_env::HanabiMove::kReturn, card_index, player, -1, -1, move);
}

bool GetDealSpecificMove(int card_index, int player, int color, int rank, pyhanabi_move_t* move) {
  return NewMove(hanabi_learning_env::HanabiMove::kDealSpecific, card_index, player, color, rank,
                 move);
}

bool GetRevealColorMove(int target_offset, int color, pyhanabi_move_t* move) {
  return NewMove(hanabi_learning_env::HanabiMove::kRevealColor, -1, target_offset, color, -1, move);
}

bool GetRevealRankMove(int target_offset, int rank, pyhanabi_move_t* move) {
  return NewMove(hanabi_learning_env::HanabiMove::kRevealRank, -1, target_offset, -1, rank, move);
}

/* Wrapper definitions for HanabiHistoryItem. */
void DeleteHistoryItem(pyhanabi_history_item_t* item) {
  REQUIRE(item != nullptr);
  REQUIRE(item->item != nullptr);
  delete reinterpret_cast<hanabi_learning_env::HanabiHistoryItem*>(item->item);
  item->item = nullptr;
}

char* HistoryItemToString(pyhanabi_history_item_t* item) {
  REQUIRE(item != nullptr);
  REQUIRE(item->item != nullptr);
  std::string str =
      reinterpret_cast<const hanabi_learning_env::HanabiHistoryItem*>(item->item)->ToString();
  return strdup(str.c_str());
}

void HistoryItemMove(pyhanabi_history_item_t* item, pyhanabi_move_t* move) {
  REQUIRE(item != nullptr);
  REQUIRE(item->item != nullptr);
  REQUIRE(move != nullptr);
  move->move = new hanabi_learning_env::HanabiMove(
      reinterpret_cast<const hanabi_learning_env::HanabiHistoryItem*>(item->item)->move);
}

int HistoryItemPlayer(pyhanabi_history_item_t* item) {
  REQUIRE(item != nullptr);
  REQUIRE(item->item != nullptr);
  return reinterpret_cast<const hanabi_learning_env::HanabiHistoryItem*>(item->item)->player;
}

int HistoryItemScored(pyhanabi_history_item_t* item) {
  REQUIRE(item != nullptr);
  REQUIRE(item->item != nullptr);
  return reinterpret_cast<const hanabi_learning_env::HanabiHistoryItem*>(item->item)->scored;
}

int HistoryItemInformationToken(pyhanabi_history_item_t* item) {
  REQUIRE(item != nullptr);
  REQUIRE(item->item != nullptr);
  return reinterpret_cast<const hanabi_learning_env::HanabiHistoryItem*>(item->item)
      ->information_token;
}

int HistoryItemColor(pyhanabi_history_item_t* item) {
  REQUIRE(item != nullptr);
  REQUIRE(item->item != nullptr);
  return reinterpret_cast<const hanabi_learning_env::HanabiHistoryItem*>(item->item)->color;
}

int HistoryItemRank(pyhanabi_history_item_t* item) {
  REQUIRE(item != nullptr);
  REQUIRE(item->item != nullptr);
  return reinterpret_cast<const hanabi_learning_env::HanabiHistoryItem*>(item->item)->rank;
}

int HistoryItemRevealBitmask(pyhanabi_history_item_t* item) {
  REQUIRE(item != nullptr);
  REQUIRE(item->item != nullptr);
  return reinterpret_cast<const hanabi_learning_env::HanabiHistoryItem*>(item->item)
      ->reveal_bitmask;
}

int HistoryItemNewlyRevealedBitmask(pyhanabi_history_item_t* item) {
  REQUIRE(item != nullptr);
  REQUIRE(item->item != nullptr);
  return reinterpret_cast<const hanabi_learning_env::HanabiHistoryItem*>(item->item)
      ->newly_revealed_bitmask;
}

int HistoryItemDealToPlayer(pyhanabi_history_item_t* item) {
  REQUIRE(item != nullptr);
  REQUIRE(item->item != nullptr);
  return reinterpret_cast<const hanabi_learning_env::HanabiHistoryItem*>(item->item)
      ->deal_to_player;
}

/* Wrapper definitions for HanabiState. */
void NewState(pyhanabi_game_t* game, pyhanabi_state_t* state) {
  REQUIRE(state != nullptr);
  REQUIRE(game != nullptr);
  REQUIRE(game->game != nullptr);
  state->state = new hanabi_learning_env::HanabiState(
      static_cast<hanabi_learning_env::HanabiGame*>(game->game));
}

void CopyState(const pyhanabi_state_t* src, pyhanabi_state_t* dest) {
  REQUIRE(src != nullptr);
  REQUIRE(src->state != nullptr);
  REQUIRE(dest != nullptr);
  dest->state = new hanabi_learning_env::HanabiState(
      *static_cast<hanabi_learning_env::HanabiState*>(src->state));
}

void DeleteState(pyhanabi_state_t* state) {
  REQUIRE(state != nullptr);
  REQUIRE(state->state != nullptr);
  delete reinterpret_cast<hanabi_learning_env::HanabiState*>(state->state);
  state->state = nullptr;
}

bool StateApplyMove(pyhanabi_state_t* state, pyhanabi_move_t* move) {
  REQUIRE(state != nullptr);
  REQUIRE(state->state != nullptr);
  REQUIRE(move != nullptr);
  REQUIRE(move->move != nullptr);
  auto hanabi_state = static_cast<hanabi_learning_env::HanabiState*>(state->state);
  const auto& hanabi_move = *static_cast<const hanabi_learning_env::HanabiMove*>(move->move);
  if (!hanabi_state->MoveIsLegal(hanabi_move)) return false;
  hanabi_state->ApplyMove(hanabi_move);
  return true;
}

int StateTurnsToPlay(pyhanabi_state_t* state) {
  return reinterpret_cast<hanabi_learning_env::HanabiState*>(state->state)->TurnsToPlay();
}

bool StateRemoveKnowledge(pyhanabi_state_t* state, int player, int card_index) {
  REQUIRE(state != nullptr);
  REQUIRE(state->state != nullptr);
  auto hanabi_state = static_cast<hanabi_learning_env::HanabiState*>(state->state);
  const auto& hands = hanabi_state->Hands();
  if (!InRange(player, hands.size()) || !InRange(card_index, hands[player].Knowledge().size())) {
    return false;
  }
  hanabi_state->RemoveKnowledge(player, card_index);
  return true;
}

int StateCurPlayer(pyhanabi_state_t* state) {
  REQUIRE(state != nullptr);
  REQUIRE(state->state != nullptr);
  return reinterpret_cast<hanabi_learning_env::HanabiState*>(state->state)->CurPlayer();
}

bool StateDealCard(pyhanabi_state_t* state) {
  REQUIRE(state != nullptr);
  REQUIRE(state->state != nullptr);
  auto hanabi_state = static_cast<hanabi_learning_env::HanabiState*>(state->state);
  if (hanabi_state->CurPlayer() != hanabi_learning_env::kChancePlayerId ||
      hanabi_state->Deck().Empty()) {
    return false;
  }
  hanabi_state->ApplyRandomChance();
  return true;
}

int StateDeckSize(pyhanabi_state_t* state) {
  REQUIRE(state != nullptr);
  REQUIRE(state->state != nullptr);
  return reinterpret_cast<hanabi_learning_env::HanabiState*>(state->state)->Deck().Size();
}

int StateFireworks(pyhanabi_state_t* state, int color) {
  REQUIRE(state != nullptr);
  REQUIRE(state->state != nullptr);
  const auto& fireworks = static_cast<hanabi_learning_env::HanabiState*>(state->state)->Fireworks();
  return InRange(color, fireworks.size()) ? fireworks[color] : -1;
}

int StateDiscardPileSize(pyhanabi_state_t* state) {
  REQUIRE(state != nullptr);
  REQUIRE(state->state != nullptr);
  return reinterpret_cast<hanabi_learning_env::HanabiState*>(state->state)->DiscardPile().size();
}

bool StateGetDiscard(pyhanabi_state_t* state, int index, pyhanabi_card_t* card) {
  REQUIRE(state != nullptr);
  REQUIRE(state->state != nullptr);
  return CopyCard(static_cast<hanabi_learning_env::HanabiState*>(state->state)->DiscardPile(),
                  index, card);
}

int StateGetHandSize(pyhanabi_state_t* state, int pid) {
  REQUIRE(state != nullptr);
  REQUIRE(state->state != nullptr);
  const auto& hands = static_cast<hanabi_learning_env::HanabiState*>(state->state)->Hands();
  return InRange(pid, hands.size()) ? static_cast<int>(hands[pid].Cards().size()) : -1;
}

bool StateGetHandCard(pyhanabi_state_t* state, int pid, int index, pyhanabi_card_t* card) {
  REQUIRE(state != nullptr);
  REQUIRE(state->state != nullptr);
  const auto& hands = static_cast<hanabi_learning_env::HanabiState*>(state->state)->Hands();
  return InRange(pid, hands.size()) && CopyCard(hands[pid].Cards(), index, card);
}

int StateInformationTokens(pyhanabi_state_t* state) {
  REQUIRE(state != nullptr);
  REQUIRE(state->state != nullptr);
  return reinterpret_cast<hanabi_learning_env::HanabiState*>(state->state)->InformationTokens();
}

int StateEndOfGameStatus(pyhanabi_state_t* state) {
  REQUIRE(state != nullptr);
  REQUIRE(state->state != nullptr);
  return reinterpret_cast<hanabi_learning_env::HanabiState*>(state->state)->EndOfGameStatus();
}

void* StateLegalMoves(pyhanabi_state_t* state) {
  REQUIRE(state != nullptr);
  REQUIRE(state->state != nullptr);
  auto hanabi_state = reinterpret_cast<hanabi_learning_env::HanabiState*>(state->state);
  std::vector<hanabi_learning_env::HanabiMove>* list =
      new std::vector<hanabi_learning_env::HanabiMove>(
          hanabi_state->LegalMoves(hanabi_state->CurPlayer()));
  return static_cast<void*>(list);
}

int StateLifeTokens(pyhanabi_state_t* state) {
  REQUIRE(state != nullptr);
  REQUIRE(state->state != nullptr);
  return reinterpret_cast<hanabi_learning_env::HanabiState*>(state->state)->LifeTokens();
}

int StateNumPlayers(pyhanabi_state_t* state) {
  REQUIRE(state != nullptr);
  REQUIRE(state->state != nullptr);
  return reinterpret_cast<hanabi_learning_env::HanabiState*>(state->state)
      ->ParentGame()
      ->NumPlayers();
}

int StateScore(pyhanabi_state_t* state) {
  REQUIRE(state != nullptr);
  REQUIRE(state->state != nullptr);
  return reinterpret_cast<hanabi_learning_env::HanabiState*>(state->state)->Score();
}

char* StateToString(pyhanabi_state_t* state) {
  REQUIRE(state != nullptr);
  REQUIRE(state->state != nullptr);
  std::string str = reinterpret_cast<hanabi_learning_env::HanabiState*>(state->state)->ToString();
  return strdup(str.c_str());
}

bool MoveIsLegal(const pyhanabi_state_t* state, const pyhanabi_move_t* move) {
  auto hanabi_state = reinterpret_cast<const hanabi_learning_env::HanabiState*>(state->state);
  auto hanabi_move = reinterpret_cast<const hanabi_learning_env::HanabiMove*>(move->move);
  return hanabi_state->MoveIsLegal(*hanabi_move);
}

bool CardPlayableOnFireworks(const pyhanabi_state_t* state, int color, int rank) {
  return reinterpret_cast<const hanabi_learning_env::HanabiState*>(state->state)
      ->CardPlayableOnFireworks(color, rank);
}

int StateLenMoveHistory(pyhanabi_state_t* state) {
  REQUIRE(state != nullptr);
  REQUIRE(state->state != nullptr);
  return reinterpret_cast<const hanabi_learning_env::HanabiState*>(state->state)
      ->MoveHistory()
      .size();
}

bool StateGetMoveHistory(pyhanabi_state_t* state, int index, pyhanabi_history_item_t* item) {
  REQUIRE(state != nullptr);
  REQUIRE(state->state != nullptr);
  REQUIRE(item != nullptr);
  const auto& history =
      static_cast<const hanabi_learning_env::HanabiState*>(state->state)->MoveHistory();
  if (!InRange(index, history.size())) return false;
  item->item = new hanabi_learning_env::HanabiHistoryItem(history[index]);
  return true;
}

/* Wrapper definitions for HanabiGame. */
void DeleteGame(pyhanabi_game_t* game) {
  REQUIRE(game != nullptr);
  REQUIRE(game->game != nullptr);
  delete reinterpret_cast<hanabi_learning_env::HanabiGame*>(game->game);
  game->game = nullptr;
}

void NewDefaultGame(pyhanabi_game_t* game) {
  std::unordered_map<std::string, std::string> params;
  REQUIRE(game != nullptr);
  game->game = static_cast<void*>(new hanabi_learning_env::HanabiGame(params));
  REQUIRE(game->game != nullptr);
}

bool NewGame(pyhanabi_game_t* game, int list_length, const char** param_list) {
  REQUIRE(game != nullptr);
  if (list_length < 0 || list_length % 2 != 0) return false;
  REQUIRE(list_length == 0 || param_list != nullptr);
  std::unordered_map<std::string, std::string> params;
  for (int p = 0; p < list_length; p += 2) {
    REQUIRE(param_list[p] != nullptr && param_list[p + 1] != nullptr);
    params[param_list[p]] = param_list[p + 1];
  }
  try {
    // The constructor throws on invalid parameters.
    game->game = new hanabi_learning_env::HanabiGame(params);
    return true;
  } catch (const std::exception&) {
    return false;
  }
}

char* GameParamString(pyhanabi_game_t* game) {
  REQUIRE(game != nullptr);
  REQUIRE(game->game != nullptr);
  std::string str;
  auto params = reinterpret_cast<hanabi_learning_env::HanabiGame*>(game->game)->Parameters();
  for (const auto& item : params) {
    str += item.first + '=' + item.second + '\n';
  }
  return strdup(str.c_str());
}

int NumPlayers(pyhanabi_game_t* game) {
  return reinterpret_cast<hanabi_learning_env::HanabiGame*>(game->game)->NumPlayers();
}

int NumColors(pyhanabi_game_t* game) {
  return reinterpret_cast<hanabi_learning_env::HanabiGame*>(game->game)->NumColors();
}

int NumRanks(pyhanabi_game_t* game) {
  return reinterpret_cast<hanabi_learning_env::HanabiGame*>(game->game)->NumRanks();
}

int HandSize(pyhanabi_game_t* game) {
  return reinterpret_cast<hanabi_learning_env::HanabiGame*>(game->game)->HandSize();
}

int MaxInformationTokens(pyhanabi_game_t* game) {
  return reinterpret_cast<hanabi_learning_env::HanabiGame*>(game->game)->MaxInformationTokens();
}

int MaxLifeTokens(pyhanabi_game_t* game) {
  return reinterpret_cast<hanabi_learning_env::HanabiGame*>(game->game)->MaxLifeTokens();
}

int ObservationType(pyhanabi_game_t* game) {
  return reinterpret_cast<hanabi_learning_env::HanabiGame*>(game->game)->ObservationType();
}

int NumCards(pyhanabi_game_t* game, int color, int rank) {
  return reinterpret_cast<hanabi_learning_env::HanabiGame*>(game->game)
      ->NumberCardInstances(color, rank);
}

int GetMoveUid(pyhanabi_game_t* game, pyhanabi_move_t* move) {
  return reinterpret_cast<hanabi_learning_env::HanabiGame*>(game->game)
      ->GetMoveUid(*reinterpret_cast<const hanabi_learning_env::HanabiMove*>(move->move));
}

bool GetMoveByUid(pyhanabi_game_t* game, int move_uid, pyhanabi_move_t* move) {
  REQUIRE(game != nullptr);
  REQUIRE(game->game != nullptr);
  REQUIRE(move != nullptr);
  auto hanabi_game = static_cast<hanabi_learning_env::HanabiGame*>(game->game);
  if (move_uid < 0 || move_uid >= hanabi_game->MaxMoves()) return false;
  move->move = new hanabi_learning_env::HanabiMove(hanabi_game->GetMove(move_uid));
  return true;
}

int MaxMoves(pyhanabi_game_t* game) {
  return reinterpret_cast<hanabi_learning_env::HanabiGame*>(game->game)->MaxMoves();
}

/*==============================================================================
 *                  Serialization and Deserialization
 *============================================================================*/

/* ============================= HanabiMove ============================= */

char* MoveToJson(pyhanabi_move_t* move) {
  REQUIRE(move != nullptr);
  REQUIRE(move->move != nullptr);
  return ToJsonString(*static_cast<hanabi_learning_env::HanabiMove*>(move->move));
}

bool MoveFromJson(const char* json_str, pyhanabi_move_t* move) {
  REQUIRE(json_str != nullptr);
  REQUIRE(move != nullptr);
  return ReplaceFromJson<hanabi_learning_env::HanabiMove>(json_str, &move->move);
}

/* ============================= HanabiGame ============================= */

char* GameToJson(pyhanabi_game_t* game) {
  REQUIRE(game != nullptr);
  REQUIRE(game->game != nullptr);
  return ToJsonString(*static_cast<hanabi_learning_env::HanabiGame*>(game->game));
}

bool GameFromJson(const char* json_str, pyhanabi_game_t* game) {
  REQUIRE(json_str != nullptr);
  REQUIRE(game != nullptr);
  return ReplaceFromJson<hanabi_learning_env::HanabiGame>(json_str, &game->game);
}

/* ============================= HanabiHistoryItem ============================= */

char* HistoryItemToJson(pyhanabi_history_item_t* item) {
  REQUIRE(item != nullptr);
  REQUIRE(item->item != nullptr);
  return ToJsonString(*static_cast<hanabi_learning_env::HanabiHistoryItem*>(item->item));
}

bool HistoryItemFromJson(const char* json_str, pyhanabi_history_item_t* item) {
  REQUIRE(json_str != nullptr);
  REQUIRE(item != nullptr);
  return ReplaceFromJson<hanabi_learning_env::HanabiHistoryItem>(json_str, &item->item);
}

/* ============================= HanabiState ============================= */

char* StateToJson(pyhanabi_state_t* state) {
  REQUIRE(state != nullptr);
  REQUIRE(state->state != nullptr);
  return ToJsonString(*static_cast<hanabi_learning_env::HanabiState*>(state->state));
}

bool StateFromJson(const char* json_str, pyhanabi_state_t* state, pyhanabi_game_t* game) {
  REQUIRE(json_str != nullptr);
  REQUIRE(state != nullptr);
  REQUIRE(game != nullptr);
  REQUIRE(game->game != nullptr);
  const auto* parent_game = static_cast<const hanabi_learning_env::HanabiGame*>(game->game);
  return ReplaceFromJson<hanabi_learning_env::HanabiState>(json_str, &state->state, parent_game);
}

/* ============================= HanabiObservation ============================= */

/*==============================================================================
 *                  Hidden-hand sampling
 *============================================================================*/

bool pyhanabi_state_candidate_counts(const pyhanabi_state_t* state, int32_t player,
                                     int32_t card_index, bool use_knowledge,
                                     const pyhanabi_card_t* excluded, int32_t num_excluded,
                                     int32_t* counts, int32_t counts_size) {
  REQUIRE(state != nullptr);
  REQUIRE(state->state != nullptr);
  REQUIRE(counts != nullptr);
  const auto& hanabi_state = *static_cast<const hanabi_learning_env::HanabiState*>(state->state);
  const auto& game = *hanabi_state.ParentGame();
  if (!InRange(player, hanabi_state.Hands().size()) || card_index < 0 ||
      card_index >= game.HandSize() || num_excluded < 0 ||
      counts_size != game.NumColors() * game.NumRanks()) {
    return false;
  }
  REQUIRE(num_excluded == 0 || excluded != nullptr);
  const auto& knowledge = hanabi_state.Hands()[player].Knowledge();
  if (use_knowledge && !InRange(card_index, knowledge.size())) return false;
  std::vector<hanabi_learning_env::HanabiCard> excluded_cards;
  for (int32_t i = 0; i < num_excluded; ++i) {
    if (!game.NumberCardInstances(excluded[i].color, excluded[i].rank)) return false;
    excluded_cards.emplace_back(excluded[i].color, excluded[i].rank);
  }
  const std::vector<int> candidates = hanabi_learning_env::HandSampler::CandidateCounts(
      hanabi_state, player, card_index, use_knowledge ? &knowledge[card_index] : nullptr,
      excluded_cards);
  std::copy(candidates.begin(), candidates.end(), counts);
  return true;
}

bool pyhanabi_state_resample_hand(pyhanabi_state_t* state, int32_t player, uint64_t seed) {
  REQUIRE(state != nullptr);
  REQUIRE(state->state != nullptr);
  return hanabi_learning_env::HandSampler(seed).ResampleHand(
      static_cast<hanabi_learning_env::HanabiState*>(state->state), player);
}

bool pyhanabi_state_restore_hand(pyhanabi_state_t* state, int32_t player,
                                 const pyhanabi_card_t* remembered, int32_t num_remembered,
                                 int32_t removed_color, int32_t removed_rank, int32_t removed_index,
                                 uint64_t seed) {
  REQUIRE(state != nullptr);
  REQUIRE(state->state != nullptr);
  auto* hanabi_state = static_cast<hanabi_learning_env::HanabiState*>(state->state);
  const auto& game = *hanabi_state->ParentGame();
  if (num_remembered < 0 || num_remembered > game.HandSize() || removed_index < -1 ||
      removed_index >= num_remembered) {
    return false;
  }
  REQUIRE(num_remembered == 0 || remembered != nullptr);
  std::vector<hanabi_learning_env::HanabiCard> remembered_cards;
  for (int32_t i = 0; i < num_remembered; ++i) {
    if (!game.NumberCardInstances(remembered[i].color, remembered[i].rank)) return false;
    remembered_cards.emplace_back(remembered[i].color, remembered[i].rank);
  }
  const bool has_removed_card = removed_color != -1 || removed_rank != -1;
  if (has_removed_card && !game.NumberCardInstances(removed_color, removed_rank)) return false;
  const hanabi_learning_env::HanabiCard removed_card =
      has_removed_card ? hanabi_learning_env::HanabiCard(removed_color, removed_rank)
                       : hanabi_learning_env::HanabiCard();
  return hanabi_learning_env::HandSampler(seed).RestoreHand(hanabi_state, player, remembered_cards,
                                                            removed_card, removed_index);
}

} /* extern "C" */
