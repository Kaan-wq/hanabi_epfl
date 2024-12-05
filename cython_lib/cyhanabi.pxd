from libc.stdint cimport uint8_t
from libcpp cimport bool

cdef extern from "pyhanabi.h":
    ctypedef struct pyhanabi_card_t:
        int color
        int rank

    ctypedef struct pyhanabi_card_knowledge_t:
        const void* knowledge

    ctypedef struct pyhanabi_move_t:
        void* move

    ctypedef struct pyhanabi_history_item_t:
        void* item

    ctypedef struct pyhanabi_state_t:
        void* state

    ctypedef struct pyhanabi_game_t:
        void* game

    ctypedef struct pyhanabi_observation_t:
        void* observation

    ctypedef struct pyhanabi_observation_encoder_t:
        void* encoder

    # Utility functions
    cdef void DeleteString(char* str)

    # Card functions
    cdef int CardValid(pyhanabi_card_t* card)

    # Card knowledge functions
    cdef char* CardKnowledgeToString(pyhanabi_card_knowledge_t* knowledge)
    cdef int ColorWasHinted(pyhanabi_card_knowledge_t* knowledge)
    cdef int KnownColor(pyhanabi_card_knowledge_t* knowledge)
    cdef int ColorIsPlausible(pyhanabi_card_knowledge_t* knowledge, int color)
    cdef int RankWasHinted(pyhanabi_card_knowledge_t* knowledge)
    cdef int KnownRank(pyhanabi_card_knowledge_t* knowledge)
    cdef int RankIsPlausible(pyhanabi_card_knowledge_t* knowledge, int rank)

    # Move functions
    cdef void DeleteMoveList(void* movelist)
    cdef int NumMoves(void* movelist)
    cdef void GetMove(void* movelist, int index, pyhanabi_move_t* move)
    cdef void DeleteMove(pyhanabi_move_t* move)
    cdef char* MoveToString(pyhanabi_move_t* move)
    cdef int MoveType(pyhanabi_move_t* move)
    cdef int CardIndex(pyhanabi_move_t* move)
    cdef int TargetOffset(pyhanabi_move_t* move)
    cdef int MoveColor(pyhanabi_move_t* move)
    cdef int MoveRank(pyhanabi_move_t* move)
    cdef bool GetDiscardMove(int card_index, pyhanabi_move_t* move)
    cdef bool GetReturnMove(int card_index, int player, pyhanabi_move_t* move)
    cdef bool GetPlayMove(int card_index, pyhanabi_move_t* move)
    cdef bool GetRevealColorMove(int target_offset, int color, pyhanabi_move_t* move)
    cdef bool GetRevealRankMove(int target_offset, int rank, pyhanabi_move_t* move)
    cdef bool GetDealSpecificMove(int card_index, int player, int color, int rank, pyhanabi_move_t* move)

    # HistoryItem functions
    cdef void DeleteHistoryItem(pyhanabi_history_item_t* item)
    cdef char* HistoryItemToString(pyhanabi_history_item_t* item)
    cdef void HistoryItemMove(pyhanabi_history_item_t* item, pyhanabi_move_t* move)
    cdef int HistoryItemPlayer(pyhanabi_history_item_t* item)
    cdef int HistoryItemScored(pyhanabi_history_item_t* item)
    cdef int HistoryItemInformationToken(pyhanabi_history_item_t* item)
    cdef int HistoryItemColor(pyhanabi_history_item_t* item)
    cdef int HistoryItemRank(pyhanabi_history_item_t* item)
    cdef int HistoryItemRevealBitmask(pyhanabi_history_item_t* item)
    cdef int HistoryItemNewlyRevealedBitmask(pyhanabi_history_item_t* item)
    cdef int HistoryItemDealToPlayer(pyhanabi_history_item_t* item)

    # Serialization + Deserialization functions
    cdef char* MoveToJson(pyhanabi_move_t* move)
    cdef bool MoveFromJson(const char* json_str, pyhanabi_move_t* move)
    cdef char* GameToJson(pyhanabi_game_t* game)
    cdef bool GameFromJson(const char* json_str, pyhanabi_game_t* game)
    cdef char* HistoryItemToJson(pyhanabi_history_item_t* item)
    cdef bool HistoryItemFromJson(const char* json_str, pyhanabi_history_item_t* item)
    cdef char* StateToJson(pyhanabi_state_t* state)
    cdef bool StateFromJson(const char* json_str, pyhanabi_state_t* state, pyhanabi_game_t* game)