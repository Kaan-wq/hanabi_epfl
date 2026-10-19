// Shared by the golden-data recorder (built against upstream HLE) and the
// parity test (built against our engine). Only uses API present in both.

#ifndef HANABI_PARITY_TRACE_H_
#define HANABI_PARITY_TRACE_H_

#include <cstdint>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#include "canonical_encoders.h"
#include "hanabi_game.h"
#include "hanabi_move.h"
#include "hanabi_observation.h"
#include "hanabi_state.h"

namespace hanabi_parity {

namespace hle = hanabi_learning_env;

// One game variant: engine parameters plus how many games to record with it.
struct Variant {
  std::string name;
  std::unordered_map<std::string, std::string> params;
  int num_games;
};

// FNV-1a, 64-bit. Stable across platforms and compilers.
inline uint64_t Fnv1a(const std::string& data) {
  uint64_t hash = 14695981039346656037ULL;
  for (unsigned char c : data) {
    hash ^= c;
    hash *= 1099511628211ULL;
  }
  return hash;
}

inline std::string Hex(uint64_t value) {
  std::ostringstream out;
  out << std::hex << value;
  return out.str();
}

// Compact, lossless move encoding: "type,card_index,target_offset,color,rank".
inline std::string EncodeMove(const hle::HanabiMove& move) {
  std::ostringstream out;
  // The getters return int8_t, which ostream would print as a character.
  out << static_cast<int>(move.MoveType()) << ',' << static_cast<int>(move.CardIndex()) << ','
      << static_cast<int>(move.TargetOffset()) << ',' << static_cast<int>(move.Color()) << ','
      << static_cast<int>(move.Rank());
  return out.str();
}

inline hle::HanabiMove DecodeMove(const std::string& encoded) {
  std::istringstream in(encoded);
  int fields[5];
  char sep;
  in >> fields[0];
  for (int i = 1; i < 5; ++i) in >> sep >> fields[i];
  return hle::HanabiMove(static_cast<hle::HanabiMove::Type>(fields[0]),
                         static_cast<int8_t>(fields[1]), static_cast<int8_t>(fields[2]),
                         static_cast<int8_t>(fields[3]), static_cast<int8_t>(fields[4]));
}

// Digests of everything observable in a state, taken before each move.
struct StepDigest {
  std::string state;         // state string, current player, scores, legal moves
  std::string observations;  // every player's observation string
  std::string encodings;     // every player's canonical observation vector

  bool operator==(const StepDigest& other) const {
    return state == other.state && observations == other.observations &&
           encodings == other.encodings;
  }
};

inline StepDigest DigestState(const hle::HanabiState& state,
                              const hle::CanonicalObservationEncoder& encoder) {
  StepDigest digest;
  std::ostringstream state_text;
  state_text << state.ToString() << "|cur=" << state.CurPlayer() << "|score=" << state.Score()
             << "|life=" << state.LifeTokens() << "|info=" << state.InformationTokens()
             << "|deck=" << state.Deck().Size()
             << "|end=" << static_cast<int>(state.EndOfGameStatus()) << "|legal=";
  if (state.CurPlayer() != hle::kChancePlayerId) {
    for (const auto& move : state.LegalMoves(state.CurPlayer())) {
      state_text << EncodeMove(move) << ';';
    }
  }
  digest.state = Hex(Fnv1a(state_text.str()));

  std::ostringstream observation_text;
  std::ostringstream encoding_text;
  for (int player = 0; player < state.ParentGame()->NumPlayers(); ++player) {
    hle::HanabiObservation observation(state, player);
    observation_text << observation.ToString() << '#';
    for (int bit : encoder.Encode(observation)) encoding_text << bit;
    encoding_text << '#';
  }
  digest.observations = Hex(Fnv1a(observation_text.str()));
  digest.encodings = Hex(Fnv1a(encoding_text.str()));
  return digest;
}

// The variants recorded in the golden file. Covers every player count, all
// observation types, and reduced games (fewer colors, smaller hands, fewer tokens).
inline std::vector<Variant> GoldenVariants() {
  std::vector<Variant> variants;
  for (int players = 2; players <= 5; ++players) {
    const std::string p = std::to_string(players);
    variants.push_back({"full_card_knowledge_p" + p,
                        {{"players", p}, {"observation_type", "1"}},
                        8});
    variants.push_back({"full_minimal_p" + p,
                        {{"players", p}, {"observation_type", "0"}},
                        2});
    variants.push_back({"full_seer_p" + p,
                        {{"players", p}, {"observation_type", "2"}},
                        2});
    variants.push_back({"small_p" + p,
                        {{"players", p},
                         {"colors", "2"},
                         {"ranks", "5"},
                         {"hand_size", "2"},
                         {"max_information_tokens", "3"},
                         {"max_life_tokens", "1"},
                         {"observation_type", "1"}},
                        2});
  }
  return variants;
}

}  // namespace hanabi_parity

#endif  // HANABI_PARITY_TRACE_H_
