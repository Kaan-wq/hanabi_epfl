"""cffi binding to the engine's C API (`engine/capi/pyhanabi.h`).

This is the only module that loads the engine. `CDEF` must match the header's
`extern "C"` block exactly; `tests/unit/env/test_pyhanabi_binding.py` enforces it.

The library is searched for, in order: the path in the `HANABI_SEARCH_LIB`
environment variable, this package directory, and the repository's CMake build
directory (`build/engine/`).
"""

import os
from pathlib import Path
from typing import Any, Protocol, cast

from cffi import FFI

from hanabi_search.exceptions import EngineLoadError

type CData = FFI.CData

LIB_ENV_VAR = "HANABI_SEARCH_LIB"
_LIB_NAMES = ("libpyhanabi.dylib", "libpyhanabi.so")
_REPO_ROOT = Path(__file__).resolve().parents[3]

CDEF = """
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

/* Utility Functions. */
void DeleteString(char* str);

/* Card functions. */
int CardValid(pyhanabi_card_t* card);

/* CardKnowledge functions */
char* CardKnowledgeToString(pyhanabi_card_knowledge_t* knowledge);
int ColorWasHinted(pyhanabi_card_knowledge_t* knowledge);
int KnownColor(pyhanabi_card_knowledge_t* knowledge);
int ColorIsPlausible(pyhanabi_card_knowledge_t* knowledge, int color);
int RankWasHinted(pyhanabi_card_knowledge_t* knowledge);
int KnownRank(pyhanabi_card_knowledge_t* knowledge);
int RankIsPlausible(pyhanabi_card_knowledge_t* knowledge, int rank);

/* Move functions. */
void DeleteMoveList(void* movelist);
int NumMoves(void* movelist);
void GetMove(void* movelist, int index, pyhanabi_move_t* move);
void DeleteMove(pyhanabi_move_t* move);
char* MoveToString(pyhanabi_move_t* move);
int MoveType(pyhanabi_move_t* move);
int CardIndex(pyhanabi_move_t* move);
int TargetOffset(pyhanabi_move_t* move);
int MoveColor(pyhanabi_move_t* move);
int MoveRank(pyhanabi_move_t* move);
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
const void* StateParentGame(pyhanabi_state_t* state);
void StateApplyMove(pyhanabi_state_t* state, pyhanabi_move_t* move);
void StateRemoveKnowledge(pyhanabi_state_t* state, int pid, int index);
int StateCurPlayer(pyhanabi_state_t* state);
void StateDealCard(pyhanabi_state_t* state);
int StateDeckSize(pyhanabi_state_t* state);
int StateFireworks(pyhanabi_state_t* state, int color);
int StateDiscardPileSize(pyhanabi_state_t* state);
void StateGetDiscard(pyhanabi_state_t* state, int index, pyhanabi_card_t* card);
int StateGetHandSize(pyhanabi_state_t* state, int pid);
void StateGetHandCard(pyhanabi_state_t* state, int pid, int index, pyhanabi_card_t* card);
int StateEndOfGameStatus(pyhanabi_state_t* state);
int StateInformationTokens(pyhanabi_state_t* state);
void* StateLegalMoves(pyhanabi_state_t* state);
int StateLifeTokens(pyhanabi_state_t* state);
int StateNumPlayers(pyhanabi_state_t* state);
int StateScore(pyhanabi_state_t* state);
int StateTurnsToPlay(pyhanabi_state_t* state);
char* StateToString(pyhanabi_state_t* state);
bool MoveIsLegal(const pyhanabi_state_t* state, const pyhanabi_move_t* move);
bool CardPlayableOnFireworks(const pyhanabi_state_t* state, int color, int rank);
int StateLenMoveHistory(pyhanabi_state_t* state);
void StateGetMoveHistory(pyhanabi_state_t* state, int index, pyhanabi_history_item_t* item);

/* Game functions. */
void DeleteGame(pyhanabi_game_t* game);
void NewDefaultGame(pyhanabi_game_t* game);
void NewGame(pyhanabi_game_t* game, int list_length, const char** param_list);
char* GameParamString(pyhanabi_game_t* game);
int NumPlayers(pyhanabi_game_t* game);
int NumColors(pyhanabi_game_t* game);
int NumRanks(pyhanabi_game_t* game);
int HandSize(pyhanabi_game_t* game);
int MaxInformationTokens(pyhanabi_game_t* game);
int MaxLifeTokens(pyhanabi_game_t* game);
int ObservationType(pyhanabi_game_t* game);
int NumCards(pyhanabi_game_t* game, int color, int rank);
int GetMoveUid(pyhanabi_game_t* game, pyhanabi_move_t* move);
void GetMoveByUid(pyhanabi_game_t* game, int move_uid, pyhanabi_move_t* move);
int MaxMoves(pyhanabi_game_t* game);

/* Observation functions. */
void NewObservation(pyhanabi_state_t* state, int player, pyhanabi_observation_t* observation);
void DeleteObservation(pyhanabi_observation_t* observation);
char* ObsToString(pyhanabi_observation_t* observation);
int ObsCurPlayerOffset(pyhanabi_observation_t* observation);
int ObsNumPlayers(pyhanabi_observation_t* observation);
int ObsGetHandSize(pyhanabi_observation_t* observation, int pid);
void ObsGetHandCard(pyhanabi_observation_t* observation, int pid, int index, pyhanabi_card_t* card);
void ObsGetHandCardKnowledge(pyhanabi_observation_t* observation, int pid, int index,
                             pyhanabi_card_knowledge_t* knowledge);
int ObsDiscardPileSize(pyhanabi_observation_t* observation);
void ObsGetDiscard(pyhanabi_observation_t* observation, int index, pyhanabi_card_t* card);
int ObsFireworks(pyhanabi_observation_t* observation, int color);
int ObsDeckSize(pyhanabi_observation_t* observation);
int ObsNumLastMoves(pyhanabi_observation_t* observation);
void ObsGetLastMove(pyhanabi_observation_t* observation, int index, pyhanabi_history_item_t* item);
int ObsInformationTokens(pyhanabi_observation_t* observation);
int ObsLifeTokens(pyhanabi_observation_t* observation);
int ObsNumLegalMoves(pyhanabi_observation_t* observation);
void ObsGetLegalMove(pyhanabi_observation_t* observation, int index, pyhanabi_move_t* move);
bool ObsCardPlayableOnFireworks(const pyhanabi_observation_t* observation, int color, int rank);

/* ObservationEncoder functions. */
void NewObservationEncoder(pyhanabi_observation_encoder_t* encoder, pyhanabi_game_t* game,
                           int type);
void DeleteObservationEncoder(pyhanabi_observation_encoder_t* encoder);
char* ObservationShape(pyhanabi_observation_encoder_t* encoder);
char* EncodeObservation(pyhanabi_observation_encoder_t* encoder,
                        pyhanabi_observation_t* observation);

/* Serialization + Deserialization functions. */
char* MoveToJson(pyhanabi_move_t* move);
bool MoveFromJson(const char* json_str, pyhanabi_move_t* move);

char* GameToJson(pyhanabi_game_t* game);
bool GameFromJson(const char* json_str, pyhanabi_game_t* game);

char* HistoryItemToJson(pyhanabi_history_item_t* item);
bool HistoryItemFromJson(const char* json_str, pyhanabi_history_item_t* item);

char* StateToJson(pyhanabi_state_t* state);
bool StateFromJson(const char* json_str, pyhanabi_state_t* state, pyhanabi_game_t* game);

"""

ffi = FFI()
ffi.cdef(CDEF)


def _candidate_paths() -> list[Path]:
    override = os.environ.get(LIB_ENV_VAR)
    if override:
        return [Path(override)]
    directories = [Path(__file__).resolve().parent, _REPO_ROOT / "build" / "engine"]
    return [directory / name for directory in directories for name in _LIB_NAMES]


def _load_library() -> tuple[Any, Path]:
    candidates = _candidate_paths()
    for path in candidates:
        if path.is_file():
            try:
                return ffi.dlopen(str(path)), path
            except OSError as error:
                raise EngineLoadError(f"Found {path} but could not load it: {error}") from error
    searched = "\n  ".join(str(path) for path in candidates)
    raise EngineLoadError(
        "Engine library libpyhanabi not found. Build it with\n"
        "  cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build\n"
        f"or set {LIB_ENV_VAR}. Searched:\n  {searched}"
    )


lib, LIBRARY_PATH = _load_library()


def take_string(c_string: CData) -> str:
    """Converts a C string returned by the engine to `str` and frees it.

    Args:
        c_string: A `char*` allocated by the engine.

    Returns:
        The decoded string.
    """
    try:
        value = ffi.string(c_string)
        return value.decode("ascii") if isinstance(value, bytes) else value
    finally:
        lib.DeleteString(c_string)


class _CardStruct(Protocol):
    color: int
    rank: int


def card_values(c_card: CData) -> tuple[int, int]:
    """Returns (color, rank) of a `pyhanabi_card_t*`."""
    card = cast(_CardStruct, c_card)
    return int(card.color), int(card.rank)
