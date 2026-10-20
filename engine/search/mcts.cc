#include "mcts.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <vector>

#include "hanabi_lib/hanabi_game.h"
#include "hanabi_lib/hanabi_move.h"
#include "hanabi_lib/util.h"

namespace hanabi_learning_env {

namespace {

// Policy fallbacks draw from a generator seeded apart from the hand sampler's.
constexpr uint64_t kPolicySeedMask = 0x5DEECE66DULL;

struct Node {
  int32_t move_uid = -1;     // The move from the parent; -1 at the root.
  int32_t first_child = -1;  // Children are contiguous: [first_child, first_child + num_children).
  int32_t num_children = 0;
  bool expanded = false;
  int32_t visits = 0;
  double total_reward = 0.0;
};

class Search {
 public:
  Search(const HanabiState& root, const MctsConfig& config, uint64_t seed)
      : root_(root),
        game_(*root.ParentGame()),
        config_(config),
        searcher_(root.CurPlayer()),
        state_(root),
        sampler_(seed),
        rng_(seed ^ kPolicySeedMask) {
    nodes_.emplace_back();
    path_.reserve(static_cast<size_t>(config.max_depth) + 2);
  }

  Status Run() {
    for (int i = 0; i < config_.iterations; ++i) {
      if (auto status = Iterate(); !status) return status;
    }
    return {};
  }

  // The root move with the best mean reward among those visited more than once.
  int BestMove() {
    const Node& root = nodes_[0];
    if (root.num_children == 0) {
      const std::vector<HanabiMove> legal = root_.LegalMoves(searcher_);
      return game_.GetMoveUid(legal[rng_() % legal.size()]);
    }
    int best = root.first_child;
    double best_mean = -std::numeric_limits<double>::infinity();
    for (int child = root.first_child; child < root.first_child + root.num_children; ++child) {
      const Node& node = nodes_[child];
      const double mean = node.visits <= 1 ? -std::numeric_limits<double>::infinity()
                                           : node.total_reward / node.visits;
      if (mean > best_mean) {
        best = child;
        best_mean = mean;
      }
    }
    return nodes_[best].move_uid;
  }

  void WriteRootVisits(std::span<int32_t> root_visits) const {
    std::ranges::fill(root_visits, 0);
    const Node& root = nodes_[0];
    for (int child = root.first_child; child < root.first_child + root.num_children; ++child) {
      root_visits[nodes_[child].move_uid] = nodes_[child].visits;
    }
  }

 private:
  Status Iterate() {
    state_ = root_;
    if (auto status = sampler_.ResampleHand(&state_, searcher_); !status) return status;
    Select();

    RememberedHand remembered;
    bool impossible = false;
    int depth = 0;
    for (size_t i = 1; i < path_.size(); ++i) {
      const HanabiMove move = game_.GetMove(nodes_[path_[i]].move_uid);
      if (state_.IsTerminal() || !state_.MoveIsLegal(move)) {
        impossible = true;
        break;
      }
      if (auto status = AdvanceDeterminized(&state_, move, searcher_, config_.determinization,
                                            &remembered, &sampler_);
          !status) {
        return status;
      }
      ++depth;
    }

    if (!impossible) {
      if (depth < config_.max_depth) Expand(path_.back());
      if (auto status = PlayRollout(&state_, config_.rollout_steps, searcher_,
                                    config_.determinization, &remembered, &sampler_, &rng_);
          !status) {
        return status;
      }
    }
    const double reward = Reward(state_, config_.reward);
    for (const int32_t index : path_) {
      Node& node = nodes_[index];
      ++node.visits;
      node.total_reward += reward;
    }
    return {};
  }

  // Fills path_ from the root down to an unexpanded or childless node, or to
  // the first unvisited child, choosing by UCT in between.
  void Select() {
    path_.clear();
    int32_t index = 0;
    while (true) {
      path_.push_back(index);
      const Node& node = nodes_[index];
      if (!node.expanded || node.num_children == 0) return;
      const int32_t end = node.first_child + node.num_children;
      int32_t best = -1;
      double best_score = -std::numeric_limits<double>::infinity();
      const double log_visits = std::log(static_cast<double>(node.visits));
      for (int32_t child = node.first_child; child < end; ++child) {
        const Node& candidate = nodes_[child];
        if (candidate.visits == 0) {
          path_.push_back(child);
          return;
        }
        const double visits = candidate.visits;
        const double score =
            candidate.total_reward / visits + config_.exploration * std::sqrt(log_visits / visits);
        if (score > best_score) {
          best = child;
          best_score = score;
        }
      }
      index = best;
    }
  }

  // Gives `index` one child per legal move of the acting player in state_.
  void Expand(int32_t index) {
    if (nodes_[index].expanded) return;
    nodes_[index].expanded = true;
    if (state_.IsTerminal()) return;
    const std::vector<HanabiMove> legal = state_.LegalMoves(state_.CurPlayer());
    const auto first = static_cast<int32_t>(nodes_.size());
    for (const HanabiMove& move : legal) {
      nodes_.push_back(Node{.move_uid = game_.GetMoveUid(move)});
    }
    std::sort(nodes_.begin() + first, nodes_.end(),
              [](const Node& a, const Node& b) { return a.move_uid < b.move_uid; });
    nodes_[index].first_child = first;
    nodes_[index].num_children = static_cast<int32_t>(legal.size());
  }

  const HanabiState& root_;
  const HanabiGame& game_;
  const MctsConfig& config_;
  const int searcher_;
  HanabiState state_;  // The determinization of the current iteration.
  HandSampler sampler_;
  SplitMix64 rng_;
  std::vector<Node> nodes_;
  std::vector<int32_t> path_;  // Node indices from the root.
};

}  // namespace

std::expected<int, std::string> MctsSearch(const HanabiState& root, const MctsConfig& config,
                                           uint64_t seed, std::span<int32_t> root_visits) {
  REQUIRE(!root.IsTerminal() && root.CurPlayer() >= 0);
  REQUIRE(root_visits.size() >= static_cast<size_t>(root.ParentGame()->MaxMoves()));
  Search search(root, config, seed);
  if (auto status = search.Run(); !status) return std::unexpected(status.error());
  search.WriteRootVisits(root_visits);
  return search.BestMove();
}

}  // namespace hanabi_learning_env
