// C API for search: re-determinizing play, the Van den Bergh policy,
// rollouts and MCTS (declared in pyhanabi.h).

#include <cstdint>
#include <format>
#include <span>

#include "hanabi_lib/hanabi_card.h"
#include "hanabi_lib/hanabi_game.h"
#include "hanabi_lib/hanabi_move.h"
#include "hanabi_lib/hanabi_state.h"
#include "hanabi_lib/hand_sampler.h"
#include "hanabi_lib/util.h"
#include "pyhanabi.h"
#include "pyhanabi_internal.h"
#include "search/mcts.h"
#include "search/van_den_bergh.h"

using pyhanabi_internal::CheckIndex;
using pyhanabi_internal::FailInt;
using pyhanabi_internal::Observation;

namespace {

namespace hle = hanabi_learning_env;

// Policy fallbacks draw from a generator seeded apart from the hand sampler's.
constexpr uint64_t kPolicySeedMask = 0x5DEECE66DULL;

hle::HanabiState& State(pyhanabi_state_t* state) {
  REQUIRE(state != nullptr);
  REQUIRE(state->state != nullptr);
  return *static_cast<hle::HanabiState*>(state->state);
}

const hle::HanabiState& State(const pyhanabi_state_t* state) {
  REQUIRE(state != nullptr);
  REQUIRE(state->state != nullptr);
  return *static_cast<const hle::HanabiState*>(state->state);
}

// Reads the caller's remembered hand (count -1: nothing remembered) into
// `hand`; false, with the error recorded, if a count or card is out of range.
bool ReadRememberedHand(const hle::HanabiGame& game, const pyhanabi_card_t* remembered,
                        int32_t num_remembered, hle::RememberedHand* hand) {
  REQUIRE(remembered != nullptr);
  if (num_remembered < -1 || num_remembered > game.HandSize()) {
    FailInt(std::format("num_remembered {} is not in [-1, {}]", num_remembered, game.HandSize()));
    return false;
  }
  hand->valid = num_remembered >= 0;
  for (int32_t i = 0; i < num_remembered; ++i) {
    if (!game.NumberCardInstances(remembered[i].color, remembered[i].rank)) {
      FailInt(std::format("remembered card (color {}, rank {}) is not in the game",
                          remembered[i].color, remembered[i].rank));
      return false;
    }
    hand->cards.push_back(hle::HanabiCard(remembered[i].color, remembered[i].rank));
  }
  return true;
}

// Writes the remembered hand back for the caller's next call.
void WriteRememberedHand(const hle::RememberedHand& hand, pyhanabi_card_t* remembered,
                         int32_t* num_remembered) {
  *num_remembered = hand.valid ? static_cast<int32_t>(hand.cards.size()) : -1;
  for (size_t i = 0; i < hand.cards.size(); ++i) {
    remembered[i] = {hand.cards[i].Color(), hand.cards[i].Rank()};
  }
}

}  // namespace

extern "C" {

int32_t pyhanabi_state_play_determinized(pyhanabi_state_t* state, const int32_t* move_uids,
                                         int32_t num_moves, int32_t searching_player,
                                         int32_t determinization, pyhanabi_card_t* remembered,
                                         int32_t* num_remembered, uint64_t seed) {
  hle::HanabiState& hanabi_state = State(state);
  REQUIRE(num_moves == 0 || move_uids != nullptr);
  REQUIRE(num_remembered != nullptr);
  const auto& game = *hanabi_state.ParentGame();
  if (num_moves < 0) return FailInt(std::format("num_moves {} is negative", num_moves));
  if (!CheckIndex(searching_player, game.NumPlayers(), "searching player") ||
      !CheckIndex(determinization, 3, "determinization")) {
    return -1;
  }
  hle::RememberedHand hand;
  if (!ReadRememberedHand(game, remembered, *num_remembered, &hand)) return -1;

  hle::HandSampler sampler(seed);
  int32_t applied = 0;
  for (; applied < num_moves && !hanabi_state.IsTerminal(); ++applied) {
    if (!CheckIndex(move_uids[applied], game.MaxMoves(), "move uid")) {
      WriteRememberedHand(hand, remembered, num_remembered);
      return -1;
    }
    const hle::HanabiMove move = game.GetMove(move_uids[applied]);
    if (!hanabi_state.MoveIsLegal(move)) break;
    const auto status = hle::AdvanceDeterminized(&hanabi_state, move, searching_player,
                                                 static_cast<hle::Determinization>(determinization),
                                                 &hand, &sampler);
    if (!status) {
      WriteRememberedHand(hand, remembered, num_remembered);
      return FailInt(status.error());
    }
  }
  WriteRememberedHand(hand, remembered, num_remembered);
  return applied;
}

int32_t pyhanabi_observation_van_den_bergh_move(const pyhanabi_observation_t* observation) {
  return hle::VanDenBerghRuleMove(Observation(observation));
}

int32_t pyhanabi_state_play_rollout(pyhanabi_state_t* state, int32_t max_steps,
                                    int32_t searching_player, int32_t determinization,
                                    pyhanabi_card_t* remembered, int32_t* num_remembered,
                                    int32_t reward_kind, uint64_t seed) {
  hle::HanabiState& hanabi_state = State(state);
  REQUIRE(num_remembered != nullptr);
  const auto& game = *hanabi_state.ParentGame();
  if (max_steps < 0) return FailInt(std::format("max_steps {} is negative", max_steps));
  if (!CheckIndex(searching_player, game.NumPlayers(), "searching player") ||
      !CheckIndex(determinization, 3, "determinization") ||
      !CheckIndex(reward_kind, 2, "reward kind")) {
    return -1;
  }
  if (!hanabi_state.IsTerminal() && hanabi_state.CurPlayer() == hle::kChancePlayerId) {
    return FailInt("a card must be dealt before the rollout");
  }
  hle::RememberedHand hand;
  if (!ReadRememberedHand(game, remembered, *num_remembered, &hand)) return -1;

  hle::HandSampler sampler(seed);
  hle::SplitMix64 rng(seed ^ kPolicySeedMask);
  const auto status =
      hle::PlayRollout(&hanabi_state, max_steps, searching_player,
                       static_cast<hle::Determinization>(determinization), &hand, &sampler, &rng);
  WriteRememberedHand(hand, remembered, num_remembered);
  if (!status) return FailInt(status.error());
  return hle::Reward(hanabi_state, static_cast<hle::RolloutReward>(reward_kind));
}

int32_t pyhanabi_state_mcts_search(const pyhanabi_state_t* state, int32_t iterations,
                                   int32_t max_depth, int32_t rollout_steps,
                                   double exploration_weight, int32_t determinization,
                                   int32_t reward_kind, uint64_t seed, int32_t* root_visits,
                                   int32_t num_root_visits) {
  const hle::HanabiState& root = State(state);
  REQUIRE(root_visits != nullptr);
  const auto& game = *root.ParentGame();
  if (iterations < 1) return FailInt(std::format("iterations {} is not positive", iterations));
  if (max_depth < 1) return FailInt(std::format("max_depth {} is not positive", max_depth));
  if (rollout_steps < 0) {
    return FailInt(std::format("rollout_steps {} is negative", rollout_steps));
  }
  if (!(exploration_weight >= 0)) {
    return FailInt(std::format("exploration_weight {} is negative", exploration_weight));
  }
  if (!CheckIndex(determinization, 3, "determinization") ||
      !CheckIndex(reward_kind, 2, "reward kind")) {
    return -1;
  }
  if (num_root_visits < game.MaxMoves()) {
    return FailInt(
        std::format("num_root_visits {} is below {} moves", num_root_visits, game.MaxMoves()));
  }
  if (root.IsTerminal()) return FailInt("the game is over");
  if (root.CurPlayer() == hle::kChancePlayerId) {
    return FailInt("a card must be dealt before the search");
  }
  const hle::MctsConfig config{
      .iterations = iterations,
      .max_depth = max_depth,
      .rollout_steps = rollout_steps,
      .exploration = exploration_weight,
      .determinization = static_cast<hle::Determinization>(determinization),
      .reward = static_cast<hle::RolloutReward>(reward_kind)};
  const auto move = hle::MctsSearch(root, config, seed,
                                    std::span(root_visits, static_cast<size_t>(num_root_visits)));
  if (!move) return FailInt(move.error());
  return *move;
}

}  // extern "C"
