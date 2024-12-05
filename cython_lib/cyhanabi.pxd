from libcpp cimport bool
from libcpp.string cimport string
from libcpp.vector cimport vector
from libcpp.unordered_map cimport unordered_map
from libcpp.memory cimport unique_ptr, shared_ptr
from libcpp.pair cimport pair
from libc.stdint cimport int8_t, uint8_t
from libcpp.random cimport mt19937

# Forward declare nlohmann::json
cdef extern from "nlohmann/json.hpp" namespace "nlohmann":
    cppclass json:
        pass

# Declare C++ enums 
cdef extern from "hanabi_lib/hanabi_game.h" namespace "hanabi_learning_env":
    cdef enum AgentObservationType:
        kMinimal
        kCardKnowledge
        kSeer

cdef extern from "hanabi_lib/hanabi_state.h" namespace "hanabi_learning_env":
    cdef enum EndOfGameType:
        kNotFinished
        kOutOfLifeTokens
        kOutOfCards
        kCompletedFireworks

# HanabiCard declaration
cdef extern from "hanabi_lib/hanabi_card.h" namespace "hanabi_learning_env":
    cppclass HanabiCard:
        HanabiCard() except +
        HanabiCard(int color, int rank) except +
        bool operator==(const HanabiCard&) const
        bool IsValid() const
        string ToString() const
        int Color() const
        int Rank() const
        json toJSON() const
        @staticmethod
        HanabiCard fromJSON(const json& j)

# HanabiHand declarations
cdef extern from "hanabi_lib/hanabi_hand.h" namespace "hanabi_learning_env":
    cppclass HanabiHand:
        # ValueKnowledge inner class
        cppclass ValueKnowledge:
            ValueKnowledge(int value_range) except +
            int Range() const
            bool ValueHinted() const
            int Value() const
            bool IsPlausible(int value) const
            void ApplyIsValueHint(int value)
            void ApplyIsNotValueHint(int value)
            json toJSON() const
            @staticmethod
            ValueKnowledge fromJSON(const json& j)

        # CardKnowledge inner class
        cppclass CardKnowledge:
            CardKnowledge(int num_colors, int num_ranks) except +
            int NumColors() const
            bool ColorHinted() const
            int Color() const
            bool ColorPlausible(int color) const
            void ApplyIsColorHint(int color)
            void ApplyIsNotColorHint(int color)
            int NumRanks() const
            bool RankHinted() const
            int Rank() const
            bool RankPlausible(int rank) const
            void ApplyIsRankHint(int rank)
            void ApplyIsNotRankHint(int rank)
            string ToString() const
            json toJSON() const
            @staticmethod
            CardKnowledge fromJSON(const json& j)

        # Main HanabiHand methods
        HanabiHand() except +
        const vector[HanabiCard]& Cards() const
        const vector[CardKnowledge]& Knowledge() const
        void AddCard(HanabiCard card, const CardKnowledge& initial_knowledge)
        void InsertCard(HanabiCard card, int card_index)
        void RemoveFromHand(int card_index, vector[HanabiCard]* discard_pile)
        void ReturnFromHand(int card_index)
        void RemoveKnowledge(int card_index, const CardKnowledge& initial_knowledge)
        uint8_t RevealRank(int rank)
        uint8_t RevealColor(int color)
        string ToString() const
        json toJSON() const
        @staticmethod
        HanabiHand fromJSON(const json& j)

# HanabiMove declaration
cdef extern from "hanabi_lib/hanabi_move.h" namespace "hanabi_learning_env":
    cppclass HanabiMove:
        enum Type:
            kInvalid "hanabi_learning_env::HanabiMove::kInvalid"
            kPlay "hanabi_learning_env::HanabiMove::kPlay"
            kDiscard "hanabi_learning_env::HanabiMove::kDiscard"
            kRevealColor "hanabi_learning_env::HanabiMove::kRevealColor"
            kRevealRank "hanabi_learning_env::HanabiMove::kRevealRank"
            kDeal "hanabi_learning_env::HanabiMove::kDeal"
            kReturn "hanabi_learning_env::HanabiMove::kReturn"
            kDealSpecific "hanabi_learning_env::HanabiMove::kDealSpecific"
        
        HanabiMove(Type move_type, int8_t card_index, int8_t target_offset,
                  int8_t color, int8_t rank) except +
        bool operator==(const HanabiMove& other_move) const
        string ToString() const
        Type MoveType() const
        bool IsValid() const
        int8_t CardIndex() const
        int8_t TargetOffset() const
        int8_t Color() const
        int8_t Rank() const
        json toJSON() const
        @staticmethod
        HanabiMove fromJSON(const json& j)

# HanabiHistoryItem declaration
cdef extern from "hanabi_lib/hanabi_history_item.h" namespace "hanabi_learning_env":
    cppclass HanabiHistoryItem:
        HanabiHistoryItem(HanabiMove move_made) except +
        string ToString() const
        json toJSON() const
        @staticmethod
        HanabiHistoryItem fromJSON(const json& j)
        HanabiMove move
        int8_t player
        bool scored
        bool information_token
        int8_t color
        int8_t rank
        uint8_t reveal_bitmask
        uint8_t newly_revealed_bitmask
        int8_t deal_to_player

# HanabiState declarations
cdef extern from "hanabi_lib/hanabi_state.h" namespace "hanabi_learning_env":
    cppclass HanabiDeck:
        HanabiDeck(const HanabiGame& game) except +
        void ReturnCard(int color, int rank)
        HanabiCard DealCard(int color, int rank)
        HanabiCard DealCard(mt19937* rng)
        int Size() const
        bool Empty() const
        int CardCount(int color, int rank) const
        json toJSON() const
        void fromJSON(const json& j)

    cppclass HanabiState:
        HanabiState(const HanabiGame* parent_game, int start_player) except +
        bool MoveIsLegal(HanabiMove move) const
        void ApplyMove(HanabiMove move)
        vector[HanabiMove] LegalMoves(int player) const
        bool CardPlayableOnFireworks(int color, int rank) const
        bool CardPlayableOnFireworks(HanabiCard card) const
        double ChanceOutcomeProb(HanabiMove move) const
        void ApplyChanceOutcome(HanabiMove move)
        void RemoveKnowledge(int player, int card_index)
        void ApplyRandomChance()
        pair[vector[HanabiMove], vector[double]] ChanceOutcomes() const
        EndOfGameType EndOfGameStatus() const
        bool IsTerminal() const
        int Score() const
        string ToString() const
        int CurPlayer() const
        int LifeTokens() const
        int InformationTokens() const
        int TurnsToPlay() const
        const vector[HanabiHand]& Hands() const
        const vector[int]& Fireworks() const
        const HanabiGame* ParentGame() const
        const HanabiDeck& Deck() const
        const vector[HanabiCard]& DiscardPile() const
        const vector[HanabiHistoryItem]& MoveHistory() const
        json toJSON() const
        @staticmethod
        HanabiState fromJSON(const json& j, const HanabiGame* game)

# HanabiGame declaration
cdef extern from "hanabi_lib/hanabi_game.h" namespace "hanabi_learning_env":
    cppclass HanabiGame:
        HanabiGame(const unordered_map[string, string]& params) except +
        int MaxMoves() const
        HanabiMove GetMove(int uid) const
        int GetMoveUid(HanabiMove move) const
        int MaxChanceOutcomes() const
        HanabiMove GetChanceOutcome(int uid) const
        int GetChanceOutcomeUid(HanabiMove move) const
        unordered_map[string, string] Parameters() const
        int NumColors() const
        int NumRanks() const
        int NumPlayers() const
        int HandSize() const
        int MaxInformationTokens() const
        int MaxLifeTokens() const
        int MaxDeckSize() const
        int NumberCardInstances(int color, int rank) const
        AgentObservationType ObservationType() const
        int GetSampledStartPlayer() const
        json toJSON() const
        @staticmethod
        HanabiGame fromJSON(const json& j)

# HanabiObservation declaration
cdef extern from "hanabi_lib/hanabi_observation.h" namespace "hanabi_learning_env":
    cppclass HanabiObservation:
        HanabiObservation() except +
        HanabiObservation(const HanabiState& state, int observing_player) except +
        string ToString() const
        json toJSON() const
        @staticmethod
        HanabiObservation fromJSON(const json& j)
        int CurPlayerOffset() const
        const vector[HanabiHand]& Hands() const
        const vector[HanabiCard]& DiscardPile() const
        const vector[int]& Fireworks() const
        int DeckSize() const
        const HanabiGame* ParentGame() const
        const vector[HanabiHistoryItem]& LastMoves() const
        int InformationTokens() const
        int LifeTokens() const
        const vector[HanabiMove]& LegalMoves() const
        bool CardPlayableOnFireworks(int color, int rank) const
        bool CardPlayableOnFireworks(HanabiCard card) const

# Add after HanabiObservation declaration and before the function declarations
cdef extern from "hanabi_lib/observation_encoder.h" namespace "hanabi_learning_env":
    cdef cppclass ObservationEncoder:
        enum Type:
            kCanonical "hanabi_learning_env::ObservationEncoder::kCanonical"

        vector[int] Shape() const
        vector[int] Encode(const HanabiObservation& obs) const
        Type type() const

    cppclass CanonicalObservationEncoder(ObservationEncoder):
        CanonicalObservationEncoder(const HanabiGame* parent_game) except +
        vector[int] Shape() const
        vector[int] Encode(const HanabiObservation& obs) const
        Type type() const