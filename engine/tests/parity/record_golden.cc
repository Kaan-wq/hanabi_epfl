// Records golden parity data from UNMODIFIED upstream HLE.
//
// Not part of the normal build: compile it against upstream's hanabi_lib at
// the commit named in engine/UPSTREAM.md (see engine/tests/parity/README.md).

#include <cstdint>
#include <fstream>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "parity_trace.h"

namespace hle = hanabi_learning_env;
namespace hp = hanabi_parity;

namespace {

constexpr int kSeedsPerVariantStride = 1000;

// Mostly-sensible policy with full-state knowledge, so games run long enough
// to reach discards, hints, end-game turns and life losses. Uses raw mt19937
// output (not std:: distributions) to stay identical across standard libraries.
hle::HanabiMove ChooseMove(const hle::HanabiState& state, std::mt19937* rng) {
  const int player = state.CurPlayer();
  const std::vector<hle::HanabiMove> legal = state.LegalMoves(player);
  const auto& cards = state.Hands()[player].Cards();

  std::vector<hle::HanabiMove> good_plays;
  std::vector<hle::HanabiMove> not_bad;
  for (const auto& move : legal) {
    const bool is_play = move.MoveType() == hle::HanabiMove::kPlay;
    const bool playable =
        is_play && state.CardPlayableOnFireworks(cards[move.CardIndex()].Color(),
                                                 cards[move.CardIndex()].Rank());
    if (playable) good_plays.push_back(move);
    if (!is_play || playable) not_bad.push_back(move);
  }

  const uint32_t roll = (*rng)() % 100;
  if (!good_plays.empty() && roll < 60) return good_plays[(*rng)() % good_plays.size()];
  if (!not_bad.empty() && roll < 95) return not_bad[(*rng)() % not_bad.size()];
  return legal[(*rng)() % legal.size()];
}

nlohmann::json RecordGame(const hp::Variant& variant, int seed, int start_player) {
  auto params = variant.params;
  params["seed"] = std::to_string(seed);
  hle::HanabiGame game(params);
  hle::CanonicalObservationEncoder encoder(&game);
  hle::HanabiState state(&game, start_player);
  std::mt19937 policy_rng(static_cast<uint32_t>(seed) * 7919U + 1U);

  nlohmann::json steps = nlohmann::json::array();
  while (!state.IsTerminal()) {
    nlohmann::json step;
    if (state.CurPlayer() == hle::kChancePlayerId) {
      state.ApplyRandomChance();
      const hle::HanabiMove dealt = state.MoveHistory().back().move;
      if (dealt.MoveType() != hle::HanabiMove::kDeal) {
        throw std::runtime_error("chance move was not recorded as a deal");
      }
      step["move"] = hp::EncodeMove(dealt);
    } else {
      const hp::StepDigest digest = hp::DigestState(state, encoder);
      const hle::HanabiMove move = ChooseMove(state, &policy_rng);
      step["move"] = hp::EncodeMove(move);
      step["digest"] = {digest.state, digest.observations, digest.encodings};
      state.ApplyMove(move);
    }
    steps.push_back(step);
  }
  const hp::StepDigest final_digest = hp::DigestState(state, encoder);
  return {{"seed", seed},
          {"start_player", start_player},
          {"steps", steps},
          {"final_score", state.Score()},
          {"end_status", static_cast<int>(state.EndOfGameStatus())},
          {"final_digest",
           {final_digest.state, final_digest.observations, final_digest.encodings}}};
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 3) {
    std::cerr << "usage: record_golden <upstream-commit> <output.json>\n";
    return 2;
  }
  nlohmann::json root;
  root["upstream_commit"] = argv[1];
  root["variants"] = nlohmann::json::array();

  const std::vector<hp::Variant> variants = hp::GoldenVariants();
  for (size_t v = 0; v < variants.size(); ++v) {
    const hp::Variant& variant = variants[v];
    const int num_players = std::stoi(variant.params.at("players"));
    nlohmann::json games = nlohmann::json::array();
    for (int g = 0; g < variant.num_games; ++g) {
      const int seed = static_cast<int>(v) * kSeedsPerVariantStride + g;
      games.push_back(RecordGame(variant, seed, g % num_players));
    }
    root["variants"].push_back({{"name", variant.name}, {"params", variant.params}, {"games", games}});
  }

  std::ofstream out(argv[2]);
  out << root.dump() << '\n';
  return out ? 0 : 1;
}
