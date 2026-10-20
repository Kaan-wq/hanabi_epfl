#include "van_den_bergh.h"

#include <algorithm>
#include <array>
#include <numeric>
#include <vector>

#include "hanabi_lib/card_mask.h"
#include "hanabi_lib/hanabi_card.h"
#include "hanabi_lib/hanabi_game.h"
#include "hanabi_lib/hanabi_hand.h"
#include "hanabi_lib/hanabi_move.h"

namespace hanabi_learning_env {

namespace {

constexpr int kMaxPlayers = 5;

// Per hand position.
using HandProbabilities = std::array<double, kMaxHandSize>;

constexpr double kPlayThreshold = 0.6;
constexpr double kDiscardThreshold = 0.99;

// What the acting player observes; hands are indexed by offset from them.
struct View {
  const HanabiGame& game;
  std::array<const HanabiHand*, kMaxPlayers> hands;
  // Whether the player's own cards count as seen (kSeer). Hidden cards are
  // invalid and never counted either way.
  bool own_cards_seen;
  // False under kMinimal: every identity is plausible and nothing is hinted.
  bool use_knowledge;
  const std::vector<int>& fireworks;
  const std::vector<HanabiCard>& discards;
  int information_tokens;
  int life_tokens;
};

View StateView(const HanabiState& state) {
  const HanabiGame& game = *state.ParentGame();
  const int player = state.CurPlayer();
  REQUIRE(player >= 0);
  View view{game,
            {},
            game.ObservationType() == HanabiGame::kSeer,
            game.ObservationType() != HanabiGame::kMinimal,
            state.Fireworks(),
            state.DiscardPile(),
            state.InformationTokens(),
            state.LifeTokens()};
  for (int offset = 0; offset < game.NumPlayers(); ++offset) {
    view.hands[offset] = &state.Hands()[(player + offset) % game.NumPlayers()];
  }
  return view;
}

View ObservationView(const HanabiObservation& observation) {
  const HanabiGame& game = *observation.ParentGame();
  // The observation already hides what the player cannot see.
  View view{game,
            {},
            true,
            true,
            observation.Fireworks(),
            observation.DiscardPile(),
            observation.InformationTokens(),
            observation.LifeTokens()};
  for (int offset = 0; offset < game.NumPlayers(); ++offset) {
    view.hands[offset] = &observation.Hands()[offset];
  }
  return view;
}

int HintedColor(const View& view, const HanabiHand::CardKnowledge& knowledge) {
  return view.use_knowledge ? knowledge.Color() : -1;
}

int HintedRank(const View& view, const HanabiHand::CardKnowledge& knowledge) {
  return view.use_knowledge ? knowledge.Rank() : -1;
}

// Index of the largest probability, the first on ties (as numpy.argmax).
int ArgMax(const HandProbabilities& probabilities, int hand_size) {
  int best = 0;
  for (int index = 1; index < hand_size; ++index) {
    if (probabilities[index] > probabilities[best]) best = index;
  }
  return best;
}

// The rules, on one view. Card classes and probabilities are computed once,
// as masks of card identities (see card_mask.h).
class Rules {
 public:
  explicit Rules(const View& view)
      : view_(view),
        game_(view_.game),
        num_colors_(game_.NumColors()),
        num_ranks_(game_.NumRanks()),
        own_(*view_.hands[0]),
        hand_size_(static_cast<int>(own_.Cards().size())) {
    ClassifyCards();
    CountUnseen();
    FindPlausibleCards();
    ComputeProbabilities();
  }

  // The move id of the first rule that applies, or -1.
  int Move() const {
    const bool can_discard = view_.information_tokens < game_.MaxInformationTokens();
    const bool can_hint = view_.information_tokens > 0;
    if (hand_size_ > 0) {
      const int likely_playable = ArgMax(playable_probability_, hand_size_);
      if (view_.life_tokens > 1 && playable_probability_[likely_playable] >= kPlayThreshold) {
        return Play(likely_playable);
      }
    }
    if (const int safe = SafeCard(); safe >= 0) return Play(safe);
    const int likely_useless = ArgMax(useless_probability_, hand_size_);
    if (can_discard && hand_size_ > 0 &&
        useless_probability_[likely_useless] >= kDiscardThreshold) {
      return Discard(likely_useless);
    }
    if (can_hint) {
      if (const int hint = HintPlayableCard(); hint >= 0) return hint;
      if (const int hint = HintUselessCard(); hint >= 0) return hint;
      if (const int hint = HintMostCards(); hint >= 0) return hint;
    }
    if (can_discard && hand_size_ > 0) return Discard(likely_useless);
    return -1;
  }

 private:
  int Index(int color, int rank) const { return color * num_ranks_ + rank; }

  // Playable now; useless if already played or no longer reachable because
  // every copy of a lower card of its color is discarded.
  void ClassifyCards() {
    CardCounts discarded{};
    for (const HanabiCard& card : view_.discards) ++discarded[Index(card.Color(), card.Rank())];
    for (int color = 0; color < num_colors_; ++color) {
      int reachable = num_ranks_;
      for (int rank = 0; rank < num_ranks_; ++rank) {
        if (discarded[Index(color, rank)] >= game_.NumberCardInstances(color, rank)) {
          reachable = rank;
          break;
        }
      }
      const int height = view_.fireworks[color];
      for (int rank = 0; rank < num_ranks_; ++rank) {
        const CardMask card = CardMask{1} << Index(color, rank);
        if (rank == height) playable_ |= card;
        if (rank < height || rank >= reachable) useless_ |= card;
        unseen_[Index(color, rank)] = game_.NumberCardInstances(color, rank) -
                                      discarded[Index(color, rank)] - (rank < height ? 1 : 0);
      }
    }
  }

  // Removes the cards in the hands the player sees.
  void CountUnseen() {
    for (int offset = view_.own_cards_seen ? 0 : 1; offset < game_.NumPlayers(); ++offset) {
      for (const HanabiCard& card : view_.hands[offset]->Cards()) {
        if (card.IsValid()) --unseen_[Index(card.Color(), card.Rank())];
      }
    }
    for (int& count : unseen_) count = std::max(count, 0);
  }

  // The identities the hints allow for each own card (all under kMinimal).
  void FindPlausibleCards() {
    for (int index = 0; index < hand_size_; ++index) {
      plausible_[index] = view_.use_knowledge
                              ? PlausibleCards(own_.Knowledge()[index], num_colors_, num_ranks_)
                              : AllCards(num_colors_, num_ranks_);
    }
  }

  void ComputeProbabilities() {
    for (int index = 0; index < hand_size_; ++index) {
      const int total = Total(unseen_, plausible_[index]);
      const int playable = Total(unseen_, plausible_[index] & playable_);
      const int useless = Total(unseen_, plausible_[index] & useless_);
      playable_probability_[index] = total > 0 ? static_cast<double>(playable) / total : 0.0;
      useless_probability_[index] = total > 0 ? static_cast<double>(useless) / total : 0.0;
    }
  }

  // The first card that every identity the hints allow makes playable, or -1.
  int SafeCard() const {
    for (int index = 0; index < hand_size_; ++index) {
      if (plausible_[index] != 0 && (plausible_[index] & ~playable_) == 0) return index;
    }
    return -1;
  }

  int HintPlayableCard() const {
    for (int offset = 1; offset < game_.NumPlayers(); ++offset) {
      const HanabiHand& hand = *view_.hands[offset];
      for (size_t index = 0; index < hand.Cards().size(); ++index) {
        const HanabiCard& card = hand.Cards()[index];
        if (!Contains(playable_, Index(card.Color(), card.Rank()))) continue;
        if (HintedRank(view_, hand.Knowledge()[index]) < 0) return RevealRank(offset, card.Rank());
        if (HintedColor(view_, hand.Knowledge()[index]) < 0) {
          return RevealColor(offset, card.Color());
        }
      }
    }
    return -1;
  }

  int HintUselessCard() const {
    for (int offset = 1; offset < game_.NumPlayers(); ++offset) {
      const HanabiHand& hand = *view_.hands[offset];
      for (size_t index = 0; index < hand.Cards().size(); ++index) {
        const HanabiCard& card = hand.Cards()[index];
        if (!Contains(useless_, Index(card.Color(), card.Rank()))) continue;
        if (HintedColor(view_, hand.Knowledge()[index]) < 0) {
          return RevealColor(offset, card.Color());
        }
        if (HintedRank(view_, hand.Knowledge()[index]) < 0) return RevealRank(offset, card.Rank());
      }
    }
    return -1;
  }

  // The hint touching the most cards of one hand; colors win ties with ranks,
  // and earlier partners and cards win ties among themselves.
  int HintMostCards() const {
    int most = -1;
    int best = -1;
    for (int offset = 1; offset < game_.NumPlayers(); ++offset) {
      const auto& cards = view_.hands[offset]->Cards();
      std::array<int, kMaxNumColors> colors{};
      std::array<int, kMaxNumRanks> ranks{};
      for (const HanabiCard& card : cards) {
        ++colors[card.Color()];
        ++ranks[card.Rank()];
      }
      for (const HanabiCard& card : cards) {
        if (colors[card.Color()] > most) {
          most = colors[card.Color()];
          best = RevealColor(offset, card.Color());
        }
        if (ranks[card.Rank()] > most) {
          most = ranks[card.Rank()];
          best = RevealRank(offset, card.Rank());
        }
      }
    }
    return best;
  }

  int Play(int index) const { return game_.GetMoveUid(HanabiMove::kPlay, index, -1, -1, -1); }
  int Discard(int index) const { return game_.GetMoveUid(HanabiMove::kDiscard, index, -1, -1, -1); }
  int RevealColor(int offset, int color) const {
    return game_.GetMoveUid(HanabiMove::kRevealColor, -1, offset, color, -1);
  }
  int RevealRank(int offset, int rank) const {
    return game_.GetMoveUid(HanabiMove::kRevealRank, -1, offset, -1, rank);
  }

  const View view_;
  const HanabiGame& game_;
  const int num_colors_;
  const int num_ranks_;
  const HanabiHand& own_;
  const int hand_size_;
  CardMask playable_ = 0;
  CardMask useless_ = 0;
  CardCounts unseen_{};  // Copies the player cannot see, at least 0.
  std::array<CardMask, kMaxHandSize> plausible_{};
  HandProbabilities playable_probability_{};
  HandProbabilities useless_probability_{};
};

}  // namespace

int VanDenBerghRuleMove(const HanabiState& state) { return Rules(StateView(state)).Move(); }

int VanDenBerghRuleMove(const HanabiObservation& observation) {
  if (observation.CurPlayerOffset() != 0) return -1;
  return Rules(ObservationView(observation)).Move();
}

int VanDenBerghMove(const HanabiState& state, SplitMix64* rng) {
  const int uid = VanDenBerghRuleMove(state);
  if (uid >= 0) return uid;
  const std::vector<HanabiMove> legal = state.LegalMoves(state.CurPlayer());
  REQUIRE(!legal.empty());
  return state.ParentGame()->GetMoveUid(legal[(*rng)() % legal.size()]);
}

int Reward(const HanabiState& state, RolloutReward reward) {
  if (reward == RolloutReward::kProgress) {
    return std::accumulate(state.Fireworks().begin(), state.Fireworks().end(), 0);
  }
  return state.Score();
}

Status PlayRollout(HanabiState* state, int max_steps, int searching_player,
                   Determinization determinization, RememberedHand* remembered,
                   HandSampler* sampler, SplitMix64* rng) {
  const HanabiGame& game = *state->ParentGame();
  for (int step = 0; step < max_steps && !state->IsTerminal(); ++step) {
    const HanabiMove move = game.GetMove(VanDenBerghMove(*state, rng));
    if (auto status = AdvanceDeterminized(state, move, searching_player, determinization,
                                          remembered, sampler);
        !status) {
      return status;
    }
  }
  return {};
}

}  // namespace hanabi_learning_env
