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

    # State functions
    cdef void NewState(pyhanabi_game_t* game, pyhanabi_state_t* state)
    cdef void CopyState(const pyhanabi_state_t* src, pyhanabi_state_t* dest)
    cdef void DeleteState(pyhanabi_state_t* state)
    cdef const void* StateParentGame(pyhanabi_state_t* state)
    cdef void StateApplyMove(pyhanabi_state_t* state, pyhanabi_move_t* move)
    cdef void StateRemoveKnowledge(pyhanabi_state_t* state, int pid, int index)
    cdef int StateCurPlayer(pyhanabi_state_t* state)
    cdef void StateDealCard(pyhanabi_state_t* state)
    cdef int StateDeckSize(pyhanabi_state_t* state)
    cdef int StateFireworks(pyhanabi_state_t* state, int color)
    cdef int StateDiscardPileSize(pyhanabi_state_t* state)
    cdef void StateGetDiscard(pyhanabi_state_t* state, int index, pyhanabi_card_t* card)
    cdef int StateGetHandSize(pyhanabi_state_t* state, int pid)
    cdef void StateGetHandCard(pyhanabi_state_t* state, int pid, int index, pyhanabi_card_t* card)
    cdef int StateEndOfGameStatus(pyhanabi_state_t* state)
    cdef int StateInformationTokens(pyhanabi_state_t* state)
    cdef void* StateLegalMoves(pyhanabi_state_t* state)
    cdef int StateLifeTokens(pyhanabi_state_t* state)
    cdef int StateNumPlayers(pyhanabi_state_t* state)
    cdef int StateScore(pyhanabi_state_t* state)
    cdef int StateTurnsToPlay(pyhanabi_state_t* state)
    cdef char* StateToString(pyhanabi_state_t* state)
    cdef bool MoveIsLegal(const pyhanabi_state_t* state, const pyhanabi_move_t* move)
    cdef bool CardPlayableOnFireworks(const pyhanabi_state_t* state, int color, int rank)
    cdef int StateLenMoveHistory(pyhanabi_state_t* state)
    cdef void StateGetMoveHistory(pyhanabi_state_t* state, int index, pyhanabi_history_item_t* item)

    # Game functions
    cdef void DeleteGame(pyhanabi_game_t* game)
    cdef void NewDefaultGame(pyhanabi_game_t* game)
    cdef void NewGame(pyhanabi_game_t* game, int list_length, const char** param_list)
    cdef char* GameParamString(pyhanabi_game_t* game)
    cdef int NumPlayers(pyhanabi_game_t* game)
    cdef int NumColors(pyhanabi_game_t* game)
    cdef int NumRanks(pyhanabi_game_t* game)
    cdef int HandSize(pyhanabi_game_t* game)
    cdef int MaxInformationTokens(pyhanabi_game_t* game)
    cdef int MaxLifeTokens(pyhanabi_game_t* game)
    cdef int ObservationType(pyhanabi_game_t* game)
    cdef int NumCards(pyhanabi_game_t* game, int color, int rank)
    cdef int GetMoveUid(pyhanabi_game_t* game, pyhanabi_move_t* move)
    cdef void GetMoveByUid(pyhanabi_game_t* game, int move_uid, pyhanabi_move_t* move)
    cdef int MaxMoves(pyhanabi_game_t* game)

    # Serialization + Deserialization functions
    cdef char* MoveToJson(pyhanabi_move_t* move)
    cdef bool MoveFromJson(const char* json_str, pyhanabi_move_t* move)
    cdef char* GameToJson(pyhanabi_game_t* game)
    cdef bool GameFromJson(const char* json_str, pyhanabi_game_t* game)
    cdef char* HistoryItemToJson(pyhanabi_history_item_t* item)
    cdef bool HistoryItemFromJson(const char* json_str, pyhanabi_history_item_t* item)
    cdef char* StateToJson(pyhanabi_state_t* state)
    cdef bool StateFromJson(const char* json_str, pyhanabi_state_t* state, pyhanabi_game_t* game)