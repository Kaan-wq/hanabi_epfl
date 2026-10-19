// Replays golden games recorded from upstream HLE and checks that our engine
// produces identical states, observations, encodings and scores at every step.

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>
#include <string>
#include <unordered_map>

#include "parity_trace.h"

namespace hle = hanabi_learning_env;
namespace hp = hanabi_parity;

namespace {

constexpr int kMaxReportedFailures = 10;

hp::StepDigest DigestFromJson(const nlohmann::json& digest) {
  return {digest.at(0).get<std::string>(), digest.at(1).get<std::string>(),
          digest.at(2).get<std::string>()};
}

std::string DescribeMismatch(const hp::StepDigest& expected, const hp::StepDigest& actual) {
  std::string parts;
  if (expected.state != actual.state) parts += " state";
  if (expected.observations != actual.observations) parts += " observations";
  if (expected.encodings != actual.encodings) parts += " encodings";
  return parts;
}

// Returns an empty string on success, otherwise a description of the first mismatch.
std::string ReplayGame(const std::unordered_map<std::string, std::string>& variant_params,
                       const nlohmann::json& record) {
  auto params = variant_params;
  params["seed"] = std::to_string(record.at("seed").get<int>());
  hle::HanabiGame game(params);
  hle::CanonicalObservationEncoder encoder(&game);
  hle::HanabiState state(&game, record.at("start_player").get<int>());

  const auto& steps = record.at("steps");
  for (size_t i = 0; i < steps.size(); ++i) {
    if (state.IsTerminal()) return "game ended early at step " + std::to_string(i);
    const auto& step = steps[i];
    if (step.contains("digest")) {
      const hp::StepDigest actual = hp::DigestState(state, encoder);
      const hp::StepDigest expected = DigestFromJson(step.at("digest"));
      if (!(actual == expected)) {
        return "step " + std::to_string(i) + " differs in" + DescribeMismatch(expected, actual) +
               "\n" + state.ToString();
      }
    }
    const hle::HanabiMove move = hp::DecodeMove(step.at("move").get<std::string>());
    if (!state.MoveIsLegal(move)) {
      return "step " + std::to_string(i) + ": recorded move " + move.ToString() + " is illegal";
    }
    state.ApplyMove(move);
  }

  if (!state.IsTerminal()) return "game did not end after the recorded moves";
  if (state.Score() != record.at("final_score").get<int>()) {
    return "final score " + std::to_string(state.Score()) +
           " != " + std::to_string(record.at("final_score").get<int>());
  }
  if (static_cast<int>(state.EndOfGameStatus()) != record.at("end_status").get<int>()) {
    return "end-of-game status differs";
  }
  const hp::StepDigest final_actual = hp::DigestState(state, encoder);
  const hp::StepDigest final_expected = DigestFromJson(record.at("final_digest"));
  if (!(final_actual == final_expected)) {
    return "final state differs in" + DescribeMismatch(final_expected, final_actual);
  }
  return "";
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 2) {
    std::cerr << "usage: parity_test <golden.json>\n";
    return 2;
  }
  std::ifstream in(argv[1]);
  if (!in) {
    std::cerr << "cannot open " << argv[1] << '\n';
    return 2;
  }
  const nlohmann::json root = nlohmann::json::parse(in);

  int games = 0;
  int failures = 0;
  for (const auto& variant : root.at("variants")) {
    const auto params = variant.at("params").get<std::unordered_map<std::string, std::string>>();
    for (const auto& record : variant.at("games")) {
      ++games;
      const std::string error = ReplayGame(params, record);
      if (error.empty()) continue;
      if (++failures <= kMaxReportedFailures) {
        std::cerr << "FAIL " << variant.at("name").get<std::string>()
                  << " seed=" << record.at("seed").get<int>() << ": " << error << '\n';
      }
    }
  }

  std::cout << games - failures << '/' << games << " golden games match upstream "
            << root.at("upstream_commit").get<std::string>() << '\n';
  return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
