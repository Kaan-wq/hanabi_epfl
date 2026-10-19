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

#ifndef __PYHANABI_H__
#define __PYHANABI_H__

/**
 * This is a pure C API to the C++ code.
 * All the declarations are loaded in src/hanabi_search/env/_pyhanabi.py.
 * The set of functions below is referred to as the 'cdef' throughout the code.
 */

#include <stdint.h>

extern "C" {

typedef struct PyHanabiCard {
  int color;
  int rank;
} pyhanabi_card_t;

typedef struct PyHanabiCardKnowledge {
  /* Points to a hanabi_learning_env::HanabiHand::CardKnowledge. */
  const void* knowledge;
} pyhanabi_card_knowledge_t;

typedef struct PyHanabiMove {
  /* Points to a hanabi_learning_env::HanabiMove. */
  void* move;
} pyhanabi_move_t;

typedef struct PyHanabiHistoryItem {
  /* Points to a hanabi_learning_env::HanabiHistoryItem. */
  void* item;
} pyhanabi_history_item_t;

typedef struct PyHanabiState {
  /* Points to a hanabi_learning_env::HanabiState. */
  void* state;
} pyhanabi_state_t;

typedef struct PyHanabiGame {
  /* Points to a hanabi_learning_env::HanabiGame. */
  void* game;
} pyhanabi_game_t;

typedef struct PyHanabiObservation {
  /* Points to a hanabi_learning_env::HanabiObservation. */
  void* observation;
} pyhanabi_observation_t;

typedef struct PyHanabiObservationEncoder {
  /* Points to a hanabi_learning_env::ObservationEncoder. */
  void* encoder;
} pyhanabi_observation_encoder_t;

/* Error convention. Functions that take an index, id or parameter from the
 * caller check it: on a bad value, a function returning bool returns false
 * and leaves its output handle untouched, and a getter returning int returns
 * -1. A NULL pointer argument is a caller bug and still aborts.
 *
 * Ownership: a function filling a pyhanabi_*_t handle allocates a new object
 * that the caller frees with the matching Delete* function. Strings returned
 * as char* are freed with DeleteString. */

/* Utility Functions. */
void DeleteString(char* str);

/* Card functions. */
int CardValid(pyhanabi_card_t* card);

/* CardKnowledge functions */
char* CardKnowledgeToString(pyhanabi_card_knowledge_t* knowledge);
int ColorWasHinted(pyhanabi_card_knowledge_t* knowledge);
int KnownColor(pyhanabi_card_knowledge_t* knowledge);
/* 1 if no hint rules out `color`, 0 if one does, -1 if `color` is out of range. */
int ColorIsPlausible(pyhanabi_card_knowledge_t* knowledge, int color);
int RankWasHinted(pyhanabi_card_knowledge_t* knowledge);
int KnownRank(pyhanabi_card_knowledge_t* knowledge);
/* 1 if no hint rules out `rank`, 0 if one does, -1 if `rank` is out of range. */
int RankIsPlausible(pyhanabi_card_knowledge_t* knowledge, int rank);

/* Move functions. */
void DeleteMoveList(void* movelist);
int NumMoves(void* movelist);
/* Copies move `index` of a list from StateLegalMoves; false if out of range. */
bool GetMove(void* movelist, int index, pyhanabi_move_t* move);
void DeleteMove(pyhanabi_move_t* move);
char* MoveToString(pyhanabi_move_t* move);
int MoveType(pyhanabi_move_t* move);
int CardIndex(pyhanabi_move_t* move);
int TargetOffset(pyhanabi_move_t* move);
int MoveColor(pyhanabi_move_t* move);
int MoveRank(pyhanabi_move_t* move);
/* Get*Move build a move without checking it against a state (use MoveIsLegal).
 * They return false if a value does not fit the move's 8-bit fields. Return
 * and deal-specific moves take an absolute player id. */
bool GetDiscardMove(int card_index, pyhanabi_move_t* move);
bool GetReturnMove(int card_index, int player, pyhanabi_move_t* move);
bool GetPlayMove(int card_index, pyhanabi_move_t* move);
bool GetRevealColorMove(int target_offset, int color, pyhanabi_move_t* move);
bool GetRevealRankMove(int target_offset, int rank, pyhanabi_move_t* move);
bool GetDealSpecificMove(int card_index, int player, int color, int rank, pyhanabi_move_t* move);

/* HistoryItem functions. */
void DeleteHistoryItem(pyhanabi_history_item_t* item);
char* HistoryItemToString(pyhanabi_history_item_t* item);
void HistoryItemMove(pyhanabi_history_item_t* item, pyhanabi_move_t* move);
int HistoryItemPlayer(pyhanabi_history_item_t* item);
int HistoryItemScored(pyhanabi_history_item_t* item);
int HistoryItemInformationToken(pyhanabi_history_item_t* item);
int HistoryItemColor(pyhanabi_history_item_t* item);
int HistoryItemRank(pyhanabi_history_item_t* item);
int HistoryItemRevealBitmask(pyhanabi_history_item_t* item);
int HistoryItemNewlyRevealedBitmask(pyhanabi_history_item_t* item);
int HistoryItemDealToPlayer(pyhanabi_history_item_t* item);

/* State functions. */
void NewState(pyhanabi_game_t* game, pyhanabi_state_t* state);
void CopyState(const pyhanabi_state_t* src, pyhanabi_state_t* dest);
void DeleteState(pyhanabi_state_t* state);
/* Applies `move`; returns false, changing nothing, if it is not legal. */
bool StateApplyMove(pyhanabi_state_t* state, pyhanabi_move_t* move);
/* Resets the hint knowledge of player `pid`'s card `index` (a knowledge slot,
 * which outlives its card across a return move); false if out of range. */
bool StateRemoveKnowledge(pyhanabi_state_t* state, int pid, int index);
int StateCurPlayer(pyhanabi_state_t* state);
/* Deals a random card; false if it is not a chance turn or the deck is empty. */
bool StateDealCard(pyhanabi_state_t* state);
int StateDeckSize(pyhanabi_state_t* state);
/* Height of firework `color`; -1 if out of range. */
int StateFireworks(pyhanabi_state_t* state, int color);
int StateDiscardPileSize(pyhanabi_state_t* state);
/* Discarded card `index`, oldest first; false if out of range. */
bool StateGetDiscard(pyhanabi_state_t* state, int index, pyhanabi_card_t* card);
/* Number of cards in player `pid`'s hand; -1 if out of range. */
int StateGetHandSize(pyhanabi_state_t* state, int pid);
/* Card `index` of player `pid`'s hand; false if either is out of range. */
bool StateGetHandCard(pyhanabi_state_t* state, int pid, int index, pyhanabi_card_t* card);
int StateEndOfGameStatus(pyhanabi_state_t* state);
int StateInformationTokens(pyhanabi_state_t* state);
void* StateLegalMoves(pyhanabi_state_t* state);
int StateLifeTokens(pyhanabi_state_t* state);
int StateNumPlayers(pyhanabi_state_t* state);
int StateScore(pyhanabi_state_t* state);
int StateTurnsToPlay(pyhanabi_state_t* state);
char* StateToString(pyhanabi_state_t* state);
/* Returns whether `move` can be applied to `state` now. Any field values are
 * accepted: out-of-range players, card indices, colors or ranks make the move
 * illegal. Return and deal-specific moves take an absolute player id in
 * TargetOffset. */
bool MoveIsLegal(const pyhanabi_state_t* state, const pyhanabi_move_t* move);
bool CardPlayableOnFireworks(const pyhanabi_state_t* state, int color, int rank);
int StateLenMoveHistory(pyhanabi_state_t* state);
/* History item `index`, oldest first; false if out of range. */
bool StateGetMoveHistory(pyhanabi_state_t* state, int index, pyhanabi_history_item_t* item);

/* Game functions. */
void DeleteGame(pyhanabi_game_t* game);
void NewDefaultGame(pyhanabi_game_t* game);
/* Builds a game from `list_length` strings: key, value, key, value, ...
 * Returns false if the list is malformed or a parameter is invalid (see
 * HanabiGame for the accepted ranges). */
bool NewGame(pyhanabi_game_t* game, int list_length, const char** param_list);
char* GameParamString(pyhanabi_game_t* game);
int NumPlayers(pyhanabi_game_t* game);
int NumColors(pyhanabi_game_t* game);
int NumRanks(pyhanabi_game_t* game);
int HandSize(pyhanabi_game_t* game);
int MaxInformationTokens(pyhanabi_game_t* game);
int MaxLifeTokens(pyhanabi_game_t* game);
int ObservationType(pyhanabi_game_t* game);
/* Copies of (color, rank) in the full deck; 0 if the card is not in the game. */
int NumCards(pyhanabi_game_t* game, int color, int rank);
int GetMoveUid(pyhanabi_game_t* game, pyhanabi_move_t* move);
/* The move with id `move_uid` in [0, MaxMoves); false if out of range. */
bool GetMoveByUid(pyhanabi_game_t* game, int move_uid, pyhanabi_move_t* move);
int MaxMoves(pyhanabi_game_t* game);

/* Observation functions. */
/* Player `player`'s view of `state`; false if `player` is out of range.
 * In an observation, players are offsets from the observer. */
bool NewObservation(pyhanabi_state_t* state, int player, pyhanabi_observation_t* observation);
void DeleteObservation(pyhanabi_observation_t* observation);
char* ObsToString(pyhanabi_observation_t* observation);
int ObsCurPlayerOffset(pyhanabi_observation_t* observation);
int ObsNumPlayers(pyhanabi_observation_t* observation);
/* Obs* getters taking an offset or index: -1 or false if out of range. */
int ObsGetHandSize(pyhanabi_observation_t* observation, int pid);
bool ObsGetHandCard(pyhanabi_observation_t* observation, int pid, int index, pyhanabi_card_t* card);
/* The knowledge points into `observation`, which must outlive it. */
bool ObsGetHandCardKnowledge(pyhanabi_observation_t* observation, int pid, int index,
                             pyhanabi_card_knowledge_t* knowledge);
int ObsDiscardPileSize(pyhanabi_observation_t* observation);
bool ObsGetDiscard(pyhanabi_observation_t* observation, int index, pyhanabi_card_t* card);
int ObsFireworks(pyhanabi_observation_t* observation, int color);
int ObsDeckSize(pyhanabi_observation_t* observation);
int ObsNumLastMoves(pyhanabi_observation_t* observation);
bool ObsGetLastMove(pyhanabi_observation_t* observation, int index, pyhanabi_history_item_t* item);
int ObsInformationTokens(pyhanabi_observation_t* observation);
int ObsLifeTokens(pyhanabi_observation_t* observation);
int ObsNumLegalMoves(pyhanabi_observation_t* observation);
bool ObsGetLegalMove(pyhanabi_observation_t* observation, int index, pyhanabi_move_t* move);
bool ObsCardPlayableOnFireworks(const pyhanabi_observation_t* observation, int color, int rank);

/* ObservationEncoder functions. */
/* Encoder of `type` (0: canonical); false if the type is unknown. */
bool NewObservationEncoder(pyhanabi_observation_encoder_t* encoder, pyhanabi_game_t* game,
                           int type);
void DeleteObservationEncoder(pyhanabi_observation_encoder_t* encoder);
char* ObservationShape(pyhanabi_observation_encoder_t* encoder);
char* EncodeObservation(pyhanabi_observation_encoder_t* encoder,
                        pyhanabi_observation_t* observation);

/* Serialization + Deserialization functions.
 *
 * *ToJson: returns the object as a JSON string, or NULL on failure. The caller
 * owns the string and frees it with DeleteString.
 *
 * *FromJson: parses `json_str` (a NUL-terminated string from the matching
 * *ToJson) into a new object and stores it in the handle, deleting the object
 * the handle held before (pass a zeroed handle to create a new one). The
 * caller owns the new object and frees it with the matching Delete* function.
 * Returns false, leaving the handle untouched, if the string is not valid
 * JSON, a field is missing or has the wrong type, or a value is out of range
 * (for a state: out of range for `game`). Whether a state's cards add up to
 * the deck is not checked. Aborts if a pointer argument is NULL.
 *
 * StateFromJson: the new state refers to `game`, which must outlive it. */
char* MoveToJson(pyhanabi_move_t* move);
bool MoveFromJson(const char* json_str, pyhanabi_move_t* move);

char* GameToJson(pyhanabi_game_t* game);
bool GameFromJson(const char* json_str, pyhanabi_game_t* game);

char* HistoryItemToJson(pyhanabi_history_item_t* item);
bool HistoryItemFromJson(const char* json_str, pyhanabi_history_item_t* item);

char* StateToJson(pyhanabi_state_t* state);
bool StateFromJson(const char* json_str, pyhanabi_state_t* state, pyhanabi_game_t* game);

/* Hidden-hand sampling for re-determinizing MCTS (hanabi_lib/hand_sampler.h).
 *
 * Card buffers use pyhanabi_card_t; count buffers are int32_t[num_colors *
 * num_ranks], indexed color * num_ranks + rank. The sampling functions draw
 * from a generator seeded by their `seed` argument, never from the game's, so
 * equal seeds give equal results on every platform. */

/* Writes to `counts` how many copies of each card could be at position
 * `card_index` (in [0, hand size)) of `player`'s hand given every other card
 * in the state: the deck, plus the card now at that position if any, minus
 * one copy per card in `excluded` (`num_excluded` cards; may be NULL if 0).
 * With `use_knowledge`, cards the player's hints rule out at that position
 * count 0 (the position must then hold card knowledge). Returns false if an
 * argument is out of range or `counts_size` != num_colors * num_ranks. */
bool pyhanabi_state_candidate_counts(const pyhanabi_state_t* state, int32_t player,
                                     int32_t card_index, bool use_knowledge,
                                     const pyhanabi_card_t* excluded, int32_t num_excluded,
                                     int32_t* counts, int32_t counts_size);

/* Replaces `player`'s hand with one sampled from what that player knows,
 * through return and deal-specific moves. Returns false, leaving `state`
 * unchanged, if `player` is out of range or no consistent hand is found. */
bool pyhanabi_state_resample_hand(pyhanabi_state_t* state, int32_t player, uint64_t seed);

/* Restores `player`'s hand after they acted: deals back the `num_remembered`
 * cards of `remembered` (their hand before acting) except position
 * `removed_index` (-1 after a hint), then samples a card for the position
 * left over. (`removed_color`, `removed_rank`) is the card they played or
 * discarded (-1, -1 after a hint); a remembered copy of it that is no longer
 * available is replaced by a sampled card. Returns false, leaving `state`
 * unchanged, if an argument is out of range or no consistent hand is found. */
bool pyhanabi_state_restore_hand(pyhanabi_state_t* state, int32_t player,
                                 const pyhanabi_card_t* remembered, int32_t num_remembered,
                                 int32_t removed_color, int32_t removed_rank, int32_t removed_index,
                                 uint64_t seed);

} /* extern "C" */

#endif