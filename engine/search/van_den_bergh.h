// The Van den Bergh rule-based policy, and rollouts that play it.
//
// A port of VanDenBerghAgent in src/hanabi_search/agents/rule_based/: the
// acting player takes the move of the first rule that applies.
//   1. Play the card most likely playable, if P >= 0.6 and more than one life is left.
//   2. Play a card that every identity the hints allow makes playable.
//   3. Discard the card most likely useless, if P >= 0.99 and hint tokens are below the maximum.
//   4. Hint a partner's playable card: its rank, else its color, if not hinted yet.
//   5. Hint a partner's useless card: its color, else its rank, if not hinted yet.
//   6. Give the hint that touches the most cards of one partner's hand.
//   7. Discard the card most likely useless, if hint tokens are below the maximum.
// Partners are scanned from the next player on, cards oldest first; ties go
// to the first candidate. Probabilities weight each identity the hints allow
// by its copies the player cannot see, and are compared as the Python rules
// compare them, so both pick the same move.

#ifndef ENGINE_SEARCH_VAN_DEN_BERGH_H_
#define ENGINE_SEARCH_VAN_DEN_BERGH_H_

#include "hanabi_lib/hanabi_observation.h"
#include "hanabi_lib/hanabi_state.h"
#include "hanabi_lib/hand_sampler.h"
#include "hanabi_lib/util.h"

namespace hanabi_learning_env {

// Returns the id (HanabiGame::GetMoveUid) of the move of the first rule that
// applies for the acting player of `state` (not terminal, not a chance turn),
// or -1 if none does. Reads only what that player observes, as the game's
// observation type defines it.
int VanDenBerghRuleMove(const HanabiState& state);

// The same for the observer of `observation`, or -1 if it is not their turn.
int VanDenBerghRuleMove(const HanabiObservation& observation);

// Like VanDenBerghRuleMove, but never -1: if no rule applies, a legal move
// drawn uniformly with `rng`.
int VanDenBerghMove(const HanabiState& state, SplitMix64* rng);

// What a rollout returns.
enum class RolloutReward {
  kScore = 0,     // The game score: 0 once all lives are lost.
  kProgress = 1,  // Cards on the fireworks, even after losing all lives.
};

// Returns `state`'s reward of kind `reward`.
int Reward(const HanabiState& state, RolloutReward reward);

// Plays up to `max_steps` Van den Bergh moves in `state`, each through
// AdvanceDeterminized (see hand_sampler.h) with `remembered` and `sampler`,
// stopping once the game is over. Policy fallbacks draw from `rng`.
// Fails if a hand swap fails; the state then holds the moves played so far.
Status PlayRollout(HanabiState* state, int max_steps, int searching_player,
                   Determinization determinization, RememberedHand* remembered,
                   HandSampler* sampler, SplitMix64* rng);

}  // namespace hanabi_learning_env

#endif  // ENGINE_SEARCH_VAN_DEN_BERGH_H_
