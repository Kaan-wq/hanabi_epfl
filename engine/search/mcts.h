// Re-determinizing information-set MCTS with Van den Bergh rollouts.
//
// The engine port of MCTSAgent (src/hanabi_search/agents/mcts/) without
// expansion rules: every legal move of a leaf becomes a child. Each iteration
//   1. copies the root and resamples the searcher's own hand,
//   2. descends by UCT to an unexpanded node, or stops at the first unvisited
//      child, replaying the path's moves through AdvanceDeterminized,
//   3. values the leaf with the reward reached if the path is impossible here
//      (a move is illegal, or the game ended on the way); otherwise expands it
//      (below `max_depth`) and plays a Van den Bergh rollout from it,
//   4. adds one visit and the reward to every node on the path.
// Nodes stand for move sequences, so one node gathers the statistics of every
// determinization that reaches it. Children are ordered by move id, and ties
// go to the first child.

#ifndef ENGINE_SEARCH_MCTS_H_
#define ENGINE_SEARCH_MCTS_H_

#include <cstdint>
#include <expected>
#include <span>
#include <string>

#include "hanabi_lib/hanabi_state.h"
#include "hanabi_lib/hand_sampler.h"
#include "search/van_den_bergh.h"

namespace hanabi_learning_env {

struct MctsConfig {
  int iterations = 50;
  int max_depth = 60;        // Leaves this many moves below the root are not expanded.
  int rollout_steps = 3;     // Van den Bergh moves played from each leaf.
  double exploration = 2.5;  // UCT exploration weight.
  Determinization determinization = Determinization::kRestore;
  RolloutReward reward = RolloutReward::kScore;
};

// Searches for the acting player of `root` (not terminal, not a chance turn)
// and returns the id of the root move with the best mean reward among those
// visited more than once (a random legal move if the root has no children).
// Writes the visits of each root move into `root_visits`, indexed by move id
// (at least game.MaxMoves() entries; others are 0). Draws every random number
// from `seed`. Fails if a hand cannot be sampled.
std::expected<int, std::string> MctsSearch(const HanabiState& root, const MctsConfig& config,
                                           uint64_t seed, std::span<int32_t> root_visits);

}  // namespace hanabi_learning_env

#endif  // ENGINE_SEARCH_MCTS_H_
