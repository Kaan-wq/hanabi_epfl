// Copyright 2018 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <iostream>
#include <numeric>
#include <vector>

#include "canonical_encoders.h"

namespace hanabi_learning_env {

namespace {

// Computes the product of dimensions in shape, i.e., how many individual
// pieces of data the encoded observation requires.
inline int FlatLength(const std::vector<int>& shape) {
  return std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>());
}

// Returns the last non-deal move from the past moves.
const HanabiHistoryItem* GetLastNonDealMove(
    const std::vector<HanabiHistoryItem>& past_moves) {
  for (auto it = past_moves.rbegin(); it != past_moves.rend(); ++it) {
    if (it->move.MoveType() != HanabiMove::Type::kDeal) {
      return &(*it);
    }
  }
  return nullptr;
}

// Precompute bits per card.
inline int BitsPerCard(const HanabiGame& game) {
  return game.NumColors() * game.NumRanks();
}

// Precompute card index.
inline int CardIndex(int color, int rank, int num_ranks) {
  return color * num_ranks + rank;
}

int HandsSectionLength(const HanabiGame& game) {
  return (game.NumPlayers() - 1) * game.HandSize() * BitsPerCard(game) +
         game.NumPlayers();
}

// Encodes cards in all other player's hands (excluding our unknown hand),
// and whether the hand is missing a card for all players (when deck is empty.)
// Each card in a hand is encoded with a one-hot representation using
// <num_colors> * <num_ranks> bits (25 bits in a standard game) per card.
// Returns the number of entries written to the encoding.
int EncodeHands(const HanabiGame& game, const HanabiObservation& obs,
                int start_offset, std::vector<int>* encoding) {
  const int bits_per_card = BitsPerCard(game);
  const int num_ranks = game.NumRanks();
  const int num_players = game.NumPlayers();
  const int hand_size = game.HandSize();

  int offset = start_offset;
  const std::vector<HanabiHand>& hands = obs.Hands();
  assert(hands.size() == num_players);
  for (int player = 1; player < num_players; ++player) {
    const std::vector<HanabiCard>& cards = hands[player].Cards();
    const int num_cards = static_cast<int>(cards.size());

    for (const HanabiCard& card : cards) {
      // Only a player's own cards can be invalid/unobserved.
      assert(card.IsValid());
      assert(card.Color() < game.NumColors());
      assert(card.Rank() < num_ranks);
      (*encoding)[offset + CardIndex(card.Color(), card.Rank(), num_ranks)] = 1;
      offset += bits_per_card;
    }

    // A player's hand can have fewer cards than the initial hand size.
    // Leave the bits for the absent cards empty (adjust the offset to skip
    // bits for the missing cards).
    offset += (hand_size - num_cards) * bits_per_card;
  }

  // For each player, set a bit if their hand is missing a card.
  for (int player = 0; player < num_players; ++player) {
    if (hands[player].Cards().size() < game.HandSize()) {
      (*encoding)[offset + player] = 1;
    }
  }
  offset += num_players;

  assert(offset - start_offset == HandsSectionLength(game));
  return offset - start_offset;
}

int BoardSectionLength(const HanabiGame& game) {
  return game.MaxDeckSize() - game.NumPlayers() * game.HandSize() +  // deck
         game.NumColors() * game.NumRanks() +                        // fireworks
         game.MaxInformationTokens() +                               // info tokens
         game.MaxLifeTokens();                                       // life tokens
}

// Encode the board, including:
//   - remaining deck size
//     (max_deck_size - num_players * hand_size bits; thermometer)
//   - state of the fireworks (<num_ranks> bits per color; one-hot)
//   - information tokens remaining (max_information_tokens bits; thermometer)
//   - life tokens remaining (max_life_tokens bits; thermometer)
// We note several features use a thermometer representation instead of one-hot.
// For example, life tokens could be: 000 (0), 100 (1), 110 (2), 111 (3).
// Returns the number of entries written to the encoding.
int EncodeBoard(const HanabiGame& game, const HanabiObservation& obs,
                int start_offset, std::vector<int>* encoding) {
  const int num_colors = game.NumColors();
  const int num_ranks = game.NumRanks();
  const int num_players = game.NumPlayers();
  const int hand_size = game.HandSize();
  const int max_deck_size = game.MaxDeckSize();

  int offset = start_offset;
  const int deck_size = obs.DeckSize();
  // Encode the deck size using a thermometer representation
  std::fill_n(encoding->begin() + offset, deck_size, 1);
  offset += (max_deck_size - hand_size * num_players);

  // Fireworks
  const std::vector<int>& fireworks = obs.Fireworks();
  for (int c = 0; c < num_colors; ++c) {
    int firework_level = fireworks[c];
    if (firework_level > 0) {
      (*encoding)[offset + firework_level - 1] = 1;
    }
    offset += num_ranks;
  }

  // Information tokens
  std::fill_n(encoding->begin() + offset, obs.InformationTokens(), 1);
  offset += game.MaxInformationTokens();

  // Life tokens
  std::fill_n(encoding->begin() + offset, obs.LifeTokens(), 1);
  offset += game.MaxLifeTokens();

  assert(offset - start_offset == BoardSectionLength(game));
  return offset - start_offset;
}

int DiscardSectionLength(const HanabiGame& game) {
  int length = 0;
  for (int c = 0; c < game.NumColors(); ++c) {
    for (int r = 0; r < game.NumRanks(); ++r) {
      length += game.NumberCardInstances(c, r);
    }
  }
  return length;
}

// Encode the discard pile. (variable length bits based on number of card instances)
// Encoding is in color-major ordering, as in kColorStr ("RYGWB"), with each
// color and rank using a thermometer to represent the number of cards
// discarded.
// Returns the number of entries written to the encoding.
int EncodeDiscards(const HanabiGame& game, const HanabiObservation& obs,
                   int start_offset, std::vector<int>* encoding) {
  const int num_colors = game.NumColors();
  const int num_ranks = game.NumRanks();

  int offset = start_offset;
  std::vector<int> discard_counts(num_colors * num_ranks, 0);
  for (const HanabiCard& card : obs.DiscardPile()) {
    ++discard_counts[CardIndex(card.Color(), card.Rank(), num_ranks)];
  }

  for (int idx = 0; idx < num_colors * num_ranks; ++idx) {
    int num_discarded = discard_counts[idx];
    int num_instances = game.NumberCardInstances(idx / num_ranks, idx % num_ranks);
    std::fill_n(encoding->begin() + offset, num_discarded, 1);
    offset += num_instances;
  }

  assert(offset - start_offset == DiscardSectionLength(game));
  return offset - start_offset;
}

int LastActionSectionLength(const HanabiGame& game) {
  return game.NumPlayers() +  // player id
         4 +                  // move types (play, discard, reveal color, reveal rank)
         game.NumPlayers() +  // target player id (if hint action)
         game.NumColors() +   // color (if hint action)
         game.NumRanks() +    // rank (if hint action)
         game.HandSize() +    // outcome (if hint action)
         game.HandSize() +    // position (if play/discard action)
         BitsPerCard(game) +  // card (if play or discard action)
         2;                   // play (successful, added information token)
}

// Encode the last player action (not chance's deal of cards). This encodes:
//  - Acting player index, relative to ourself (<num_players> bits; one-hot)
//  - The MoveType (4 bits; one-hot)
//  - Target player index, relative to acting player, if a reveal move
//    (<num_players> bits; one-hot)
//  - Color revealed, if a reveal color move (<num_colors> bits; one-hot)
//  - Rank revealed, if a reveal rank move (<num_ranks> bits; one-hot)
//  - Reveal outcome (<hand_size> bits; each bit is 1 if the card was hinted at)
//  - Position played/discarded (<hand_size> bits; one-hot)
//  - Card played/discarded (<num_colors> * <num_ranks> bits; one-hot)
// Returns the number of entries written to the encoding.
int EncodeLastAction(const HanabiGame& game, const HanabiObservation& obs,
                     int start_offset, std::vector<int>* encoding) {
  const int num_colors = game.NumColors();
  const int num_ranks = game.NumRanks();
  const int num_players = game.NumPlayers();
  const int hand_size = game.HandSize();

  int offset = start_offset;
  const HanabiHistoryItem* last_move = GetLastNonDealMove(obs.LastMoves());
  if (last_move == nullptr) {
    offset += LastActionSectionLength(game);
  } else {
    HanabiMove::Type last_move_type = last_move->move.MoveType();

    // Player ID
    (*encoding)[offset + last_move->player] = 1;
    offset += num_players;

    // Move Type
    switch (last_move_type) {
      case HanabiMove::Type::kPlay:
        (*encoding)[offset + 0] = 1;
        break;
      case HanabiMove::Type::kDiscard:
        (*encoding)[offset + 1] = 1;
        break;
      case HanabiMove::Type::kRevealColor:
        (*encoding)[offset + 2] = 1;
        break;
      case HanabiMove::Type::kRevealRank:
        (*encoding)[offset + 3] = 1;
        break;
      default:
        break;
    }
    offset += 4;

    // Target player (if hint action)
    if (last_move_type == HanabiMove::Type::kRevealColor ||
        last_move_type == HanabiMove::Type::kRevealRank) {
      int8_t observer_relative_target =
          (last_move->player + last_move->move.TargetOffset()) % num_players;
      (*encoding)[offset + observer_relative_target] = 1;
    }
    offset += num_players;

    // Color (if hint action)
    if (last_move_type == HanabiMove::Type::kRevealColor) {
      (*encoding)[offset + last_move->move.Color()] = 1;
    }
    offset += num_colors;

    // Rank (if hint action)
    if (last_move_type == HanabiMove::Type::kRevealRank) {
      (*encoding)[offset + last_move->move.Rank()] = 1;
    }
    offset += num_ranks;

    // Outcome (if hinted action)
    if (last_move_type == HanabiMove::Type::kRevealColor ||
        last_move_type == HanabiMove::Type::kRevealRank) {
      uint8_t reveal_bitmask = last_move->reveal_bitmask;
      for (int i = 0; i < hand_size; ++i) {
        if (reveal_bitmask & (1 << i)) {
          (*encoding)[offset + i] = 1;
        }
      }
    }
    offset += hand_size;

    // Position (if play or discard action)
    if (last_move_type == HanabiMove::Type::kPlay ||
        last_move_type == HanabiMove::Type::kDiscard) {
      (*encoding)[offset + last_move->move.CardIndex()] = 1;
    }
    offset += hand_size;

    // Card (if play or discard action)
    if (last_move_type == HanabiMove::Type::kPlay ||
        last_move_type == HanabiMove::Type::kDiscard) {
      assert(last_move->color >= 0);
      assert(last_move->rank >= 0);
      (*encoding)[offset +
                  CardIndex(last_move->color, last_move->rank, num_ranks)] = 1;
    }
    offset += BitsPerCard(game);

    // Was successful and/or added information token (if play action)
    if (last_move_type == HanabiMove::Type::kPlay) {
      if (last_move->scored) {
        (*encoding)[offset] = 1;
      }
      if (last_move->information_token) {
        (*encoding)[offset + 1] = 1;
      }
    }
    offset += 2;
  }

  assert(offset - start_offset == LastActionSectionLength(game));
  return offset - start_offset;
}

int CardKnowledgeSectionLength(const HanabiGame& game) {
  return game.NumPlayers() * game.HandSize() *
         (BitsPerCard(game) + game.NumColors() + game.NumRanks());
}

// Encode the common card knowledge.
// For each card/position in each player's hand, including the observing player,
// encode the possible cards that could be in that position and whether the
// color and rank were directly revealed by a Reveal action. Possible card
// values are in color-major order, using <num_colors> * <num_ranks> bits per
// card.
// Returns the number of entries written to the encoding.
int EncodeCardKnowledge(const HanabiGame& game, const HanabiObservation& obs,
                        int start_offset, std::vector<int>* encoding) {
  const int bits_per_card = BitsPerCard(game);
  const int num_colors = game.NumColors();
  const int num_ranks = game.NumRanks();
  const int num_players = game.NumPlayers();
  const int hand_size = game.HandSize();

  int offset = start_offset;
  const std::vector<HanabiHand>& hands = obs.Hands();
  assert(hands.size() == num_players);

  for (int player = 0; player < num_players; ++player) {
    const std::vector<HanabiHand::CardKnowledge>& knowledge =
        hands[player].Knowledge();
    const int num_cards = static_cast<int>(knowledge.size());

    for (const HanabiHand::CardKnowledge& card_knowledge : knowledge) {
      // Flattened loop over possible cards
      for (int idx = 0; idx < bits_per_card; ++idx) {
        int color = idx / num_ranks;
        int rank = idx % num_ranks;
        if (card_knowledge.ColorPlausible(color) &&
            card_knowledge.RankPlausible(rank)) {
          (*encoding)[offset + idx] = 1;
        }
      }
      offset += bits_per_card;

      // Add bits for explicitly revealed colors and ranks.
      if (card_knowledge.ColorHinted()) {
        (*encoding)[offset + card_knowledge.Color()] = 1;
      }
      offset += num_colors;
      if (card_knowledge.RankHinted()) {
        (*encoding)[offset + card_knowledge.Rank()] = 1;
      }
      offset += num_ranks;
    }

    // Adjust offset for missing cards
    offset += (hand_size - num_cards) * (bits_per_card + num_colors + num_ranks);
  }

  assert(offset - start_offset == CardKnowledgeSectionLength(game));
  return offset - start_offset;
}

}  // namespace

// Constructor with precomputed values
CanonicalObservationEncoder::CanonicalObservationEncoder(const HanabiGame* parent_game)
    : parent_game_(parent_game) {
  bits_per_card_ = BitsPerCard(*parent_game_);
  hands_section_length_ = HandsSectionLength(*parent_game_);
  board_section_length_ = BoardSectionLength(*parent_game_);
  discard_section_length_ = DiscardSectionLength(*parent_game_);
  last_action_section_length_ = LastActionSectionLength(*parent_game_);
  card_knowledge_section_length_ =
      (parent_game_->ObservationType() == HanabiGame::kMinimal)
          ? 0
          : CardKnowledgeSectionLength(*parent_game_);
  total_encoding_length_ = hands_section_length_ + board_section_length_ +
                           discard_section_length_ + last_action_section_length_ +
                           card_knowledge_section_length_;
  encoding_.resize(total_encoding_length_, 0);
}

std::vector<int> CanonicalObservationEncoder::Shape() const {
  return {total_encoding_length_};
}

std::vector<int> CanonicalObservationEncoder::Encode(
    const HanabiObservation& obs) const {
  // Reset the encoding vector
  std::fill(encoding_.begin(), encoding_.end(), 0);

  // This offset is an index to the start of each section of the bit vector.
  // It is incremented at the end of each section.
  int offset = 0;
  offset += EncodeHands(*parent_game_, obs, offset, &encoding_);
  offset += EncodeBoard(*parent_game_, obs, offset, &encoding_);
  offset += EncodeDiscards(*parent_game_, obs, offset, &encoding_);
  offset += EncodeLastAction(*parent_game_, obs, offset, &encoding_);
  if (card_knowledge_section_length_ > 0) {
    offset += EncodeCardKnowledge(*parent_game_, obs, offset, &encoding_);
  }

  assert(offset == total_encoding_length_);
  return encoding_;
}

}  // namespace hanabi_learning_env
