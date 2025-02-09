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

#include "hanabi_state.h"
#include <iostream>
#include <algorithm>
#include <cassert>
#include <sstream>
#include <nlohmann/json.hpp>
using json = nlohmann::json;
#include <numeric>
#include "util.h"

namespace hanabi_learning_env {

namespace {
// Returns bitmask of card indices which match color.
uint8_t HandColorBitmask(const HanabiHand& hand, int color) {
  uint8_t mask = 0;
  const auto& cards = hand.Cards();
  assert(cards.size() <= 8);  // More than 8 cards is not supported.
  for (int i = 0; i < cards.size(); ++i) {
    if (cards[i].Color() == color) {
      mask |= static_cast<uint8_t>(1) << i;
    }
  }
  return mask;
}

// Returns bitmask of card indices which match color.
uint8_t HandRankBitmask(const HanabiHand& hand, int rank) {
  uint8_t mask = 0;
  const auto& cards = hand.Cards();
  assert(cards.size() <= 8);  // More than 8 cards is not supported.
  for (int i = 0; i < cards.size(); ++i) {
    if (cards[i].Rank() == rank) {
      mask |= static_cast<uint8_t>(1) << i;
    }
  }
  return mask;
}
}  // namespace

HanabiState::HanabiDeck::HanabiDeck(const HanabiGame& game)
    : card_count_(game.NumColors() * game.NumRanks(), 0),
      total_count_(0),
      num_ranks_(game.NumRanks()) {
  for (int color = 0; color < game.NumColors(); ++color) {
    for (int rank = 0; rank < game.NumRanks(); ++rank) {
      auto count = game.NumberCardInstances(color, rank);
      card_count_[CardToIndex(color, rank)] = count;
      total_count_ += count;
    }
  }
}

HanabiCard HanabiState::HanabiDeck::DealCard(std::mt19937* rng) {
  // MB: DealCard function. Need a AddCard option?
  if (Empty()) {
    return HanabiCard();
  }
  std::discrete_distribution<std::mt19937::result_type> dist(
      card_count_.begin(), card_count_.end());
  int index = dist(*rng);
  assert(card_count_[index] > 0);
  --card_count_[index];
  --total_count_;
  return HanabiCard(IndexToColor(index), IndexToRank(index));
}

HanabiCard HanabiState::HanabiDeck::DealCard(int color, int rank) {
  // MB: This is a promising option. Check for validity elsewhere
  int index = CardToIndex(color, rank);
  if (card_count_[index] <= 0) {
    return HanabiCard();
  }
  assert(card_count_[index] > 0);
  --card_count_[index];
  --total_count_;
  //std::cout<<"After DealCard count of that card colorrank ";
  //std::cout<<color;
  //std::cout<<rank;
  //std::cout<<" index ";
  //std::cout<<index;
  //std::cout<<" was ";
  //std::cout<<card_count_[index];
  return HanabiCard(IndexToColor(index), IndexToRank(index));
}

void HanabiState::HanabiDeck::ReturnCard(int color, int rank) {
  // MB: When a card is being Returned from a Hand to the deck
  // Do we need to check for if it's valid to return the card? Probs not.
  int index = CardToIndex(color, rank);
  card_count_[index] += 1;
  total_count_ += 1;
  //std::cout<<"After ReturnCard, count of that card colorrank ";
  //std::cout<<color;
  //std::cout<<rank;
  //std::cout<<" index ";
  //std::cout<<index;
  //std::cout<<" was ";
  //std::cout<<card_count_[index];
}

HanabiState::HanabiState(const HanabiGame* parent_game, int start_player)
    : parent_game_(parent_game),
      deck_(*parent_game),
      hands_(parent_game->NumPlayers()),
      cur_player_(kChancePlayerId),
      next_non_chance_player_(start_player >= 0 &&
                                      start_player < parent_game->NumPlayers()
                                  ? start_player
                                  : parent_game->GetSampledStartPlayer()),
      information_tokens_(parent_game->MaxInformationTokens()),
      life_tokens_(parent_game->MaxLifeTokens()),
      fireworks_(parent_game->NumColors(), 0),
      turns_to_play_(parent_game->NumPlayers()) {}

void HanabiState::RemoveKnowledge(int player, int card_index) {
    // MB: Define the default card knowledge structure
    HanabiHand::CardKnowledge card_knowledge(ParentGame()->NumColors(),
                                      ParentGame()->NumRanks());
    hands_[player].RemoveKnowledge(card_index, card_knowledge);
}

void HanabiState::AdvanceToNextPlayer(bool stayOnPlayer) {

  if (!deck_.Empty() && PlayerToDeal() >= 0) {
    cur_player_ = kChancePlayerId;
  } else if (stayOnPlayer == true){
    //MB: Expression neccesary to ensure modulo wraps -1 > 2 (C++ % implementation)
    cur_player_ = ((next_non_chance_player_-1) + hands_.size()) % hands_.size();
  } else {
    cur_player_ = next_non_chance_player_;
    next_non_chance_player_ = (cur_player_ + 1) % hands_.size();
  }
}

bool HanabiState::IncrementInformationTokens() {
  if (information_tokens_ < ParentGame()->MaxInformationTokens()) {
    ++information_tokens_;
    return true;
  } else {
    return false;
  }
}

void HanabiState::DecrementInformationTokens() {
  assert(information_tokens_ > 0);
  --information_tokens_;
}

void HanabiState::DecrementLifeTokens() {
  assert(life_tokens_ > 0);
  --life_tokens_;
}

std::pair<bool, bool> HanabiState::AddToFireworks(HanabiCard card) {
  if (CardPlayableOnFireworks(card)) {
    ++fireworks_[card.Color()];
    // Check if player completed a stack.
    if (fireworks_[card.Color()] == ParentGame()->NumRanks()) {
      return {true, IncrementInformationTokens()};
    }
    return {true, false};
  } else {
    DecrementLifeTokens();
    return {false, false};
  }
}

bool HanabiState::HintingIsLegal(HanabiMove move) const {
  if (InformationTokens() <= 0) {
    return false;
  }
  if (move.TargetOffset() < 1 ||
      move.TargetOffset() >= ParentGame()->NumPlayers()) {
    return false;
  }
  return true;
}

int HanabiState::PlayerToDeal() const {
  for (int i = 0; i < hands_.size(); ++i) {
    if (hands_[i].Cards().size() < ParentGame()->HandSize()) {
      return i;
    }
  }
  return -1;
}

bool HanabiState::MoveIsLegal(HanabiMove move) const {
  switch (move.MoveType()) {
    case HanabiMove::kDeal:
      if (cur_player_ != kChancePlayerId) {
        return false;
      }
      if (deck_.CardCount(move.Color(), move.Rank()) == 0) {
        return false;
      }
      break;
    //MB: Copy of Deal for now
    case HanabiMove::kDealSpecific:
      if (cur_player_ != kChancePlayerId) {
        return false;
      }
      if (deck_.CardCount(move.Color(), move.Rank()) == 0) {
        std::cout<<"MB Error: Card color ";
        std::cout<<move.Color();
        std::cout<<" rank ";
        std::cout<<move.Rank();
        std::cout<<" tried to be dealt specifically but it's not in core deck";
        return false;
      }
      break;
    case HanabiMove::kDiscard:
      if (InformationTokens() >= ParentGame()->MaxInformationTokens()) {
        return false;
      }
      if (move.CardIndex() >= hands_[cur_player_].Cards().size()) {
        return false;
      }
      break;
    case HanabiMove::kReturn:
      if (move.CardIndex() >= hands_[move.TargetOffset()].Cards().size()) {
        return false;
      }
      break;
    case HanabiMove::kPlay:
      if (move.CardIndex() >= hands_[cur_player_].Cards().size()) {
        return false;
      }
      break;
    case HanabiMove::kRevealColor: {
      if (!HintingIsLegal(move)) {
        return false;
      }
      const auto& cards = HandByOffset(move.TargetOffset()).Cards();
      if (!std::any_of(cards.begin(), cards.end(),
                       [move](const HanabiCard& card) {
                         return card.Color() == move.Color();
                       })) {
        return false;
      }
      break;
    }
    case HanabiMove::kRevealRank: {
      if (!HintingIsLegal(move)) {
        return false;
      }
      const auto& cards = HandByOffset(move.TargetOffset()).Cards();
      if (!std::any_of(cards.begin(), cards.end(),
                       [move](const HanabiCard& card) {
                         return card.Rank() == move.Rank();
                       })) {
        return false;
      }
      break;
    }
    default:
      return false;
  }
  return true;
}

void HanabiState::ApplyMove(HanabiMove move) {
  REQUIRE(MoveIsLegal(move));
  //MB: DealSpecific and Return can happen freely. Others mean it is now end game turns.
  if (deck_.Empty() && move.MoveType() != HanabiMove::kDealSpecific && move.MoveType()!= HanabiMove::kReturn) {
    --turns_to_play_;
  }
  // MB: Do we really want a RETURN or DEALSPECFIC move to add to history? Might have to deal with this in history
  HanabiHistoryItem history(move);
  history.player = cur_player_;
  switch (move.MoveType()) {
    case HanabiMove::kDeal: {
        //MB: Usuallly, deal returns an empty card
        history.deal_to_player = PlayerToDeal();
        HanabiHand::CardKnowledge card_knowledge(ParentGame()->NumColors(),
                                      ParentGame()->NumRanks());
        if (parent_game_->ObservationType() == HanabiGame::kSeer){
          card_knowledge.ApplyIsColorHint(move.Color());
          card_knowledge.ApplyIsRankHint(move.Rank());
        }
        hands_[history.deal_to_player].AddCard(
            deck_.DealCard(move.Color(), move.Rank()),
            card_knowledge);
      }
      break;
    case HanabiMove::kDealSpecific: {
        //MB: Important note: TargetOffset is bastardised here; really its just the absolute player id dealt to
        history.deal_to_player = move.TargetOffset();
        hands_[history.deal_to_player].InsertCard(
            deck_.DealCard(move.Color(), move.Rank()), move.CardIndex());
      }
      break;
    case HanabiMove::kDiscard:
      history.information_token = IncrementInformationTokens();
      history.color = hands_[cur_player_].Cards()[move.CardIndex()].Color();
      history.rank = hands_[cur_player_].Cards()[move.CardIndex()].Rank();
      hands_[cur_player_].RemoveFromHand(move.CardIndex(), &discard_pile_);
      break;
    case HanabiMove::kReturn:
      //MB: Return bastardises framework and uses TargetOffset to specify which hand to remove from
      history.player = move.TargetOffset();
      history.color = hands_[history.player].Cards()[move.CardIndex()].Color();
      history.rank = hands_[history.player].Cards()[move.CardIndex()].Rank();
      deck_.ReturnCard(hands_[history.player].Cards()[move.CardIndex()].Color()
                      ,hands_[history.player].Cards()[move.CardIndex()].Rank());
      hands_[history.player].ReturnFromHand(move.CardIndex());
      break;
    case HanabiMove::kPlay:
      history.color = hands_[cur_player_].Cards()[move.CardIndex()].Color();
      history.rank = hands_[cur_player_].Cards()[move.CardIndex()].Rank();
      std::tie(history.scored, history.information_token) =
          AddToFireworks(hands_[cur_player_].Cards()[move.CardIndex()]);
      hands_[cur_player_].RemoveFromHand(
          move.CardIndex(), history.scored ? nullptr : &discard_pile_);
      break;
    case HanabiMove::kRevealColor:
      DecrementInformationTokens();
      history.reveal_bitmask =
          HandColorBitmask(*HandByOffset(move.TargetOffset()), move.Color());
      history.newly_revealed_bitmask =
          HandByOffset(move.TargetOffset())->RevealColor(move.Color());
      break;
    case HanabiMove::kRevealRank:
      DecrementInformationTokens();
      history.reveal_bitmask =
          HandRankBitmask(*HandByOffset(move.TargetOffset()), move.Rank());
      history.newly_revealed_bitmask =
          HandByOffset(move.TargetOffset())->RevealRank(move.Rank());
      break;
    default:
      std::abort();  // Should not be possible.
  }
  move_history_.push_back(history);
  //MB WARNING: DealSpecific skips this step.
  AdvanceToNextPlayer(move.MoveType() == HanabiMove::kDealSpecific);
}

double HanabiState::ChanceOutcomeProb(HanabiMove move) const {
  return static_cast<double>(deck_.CardCount(move.Color(), move.Rank())) /
         static_cast<double>(deck_.Size());
}

void HanabiState::ApplyRandomChance() {
  auto chance_outcomes = ChanceOutcomes();
  REQUIRE(!chance_outcomes.second.empty());
  ApplyMove(ParentGame()->PickRandomChance(chance_outcomes));
}

std::vector<HanabiMove> HanabiState::LegalMoves(int player) const {
  std::vector<HanabiMove> movelist;
  // kChancePlayer=-1 must be handled by ChanceOutcome.
  REQUIRE(player >= 0 && player < ParentGame()->NumPlayers());
  if (player != cur_player_) {
    // Turn-based game. Empty move list for other players.
    return movelist;
  }
  int max_move_uid = ParentGame()->MaxMoves();
  for (int uid = 0; uid < max_move_uid; ++uid) {
    HanabiMove move = ParentGame()->GetMove(uid);
    if (MoveIsLegal(move)) {
      movelist.push_back(move);
    }
  }
  return movelist;
}

bool HanabiState::CardPlayableOnFireworks(int color, int rank) const {
  if (color < 0 || color >= ParentGame()->NumColors()) {
    return false;
  }
  return rank == fireworks_[color];
}

std::pair<std::vector<HanabiMove>, std::vector<double>>
HanabiState::ChanceOutcomes() const {
  std::pair<std::vector<HanabiMove>, std::vector<double>> rv;
  int max_outcome_uid = ParentGame()->MaxChanceOutcomes();
  for (int uid = 0; uid < max_outcome_uid; ++uid) {
    HanabiMove move = ParentGame()->GetChanceOutcome(uid);
    if (MoveIsLegal(move)) {
      rv.first.push_back(move);
      rv.second.push_back(ChanceOutcomeProb(move));
    }
  }
  return rv;
}

// Format:  <life tokens>:<info tokens>:
//           <fireworks color 1>-....::
//            <player 1 card>-.... || <player 1 hint>-...
//            :....
//            ::<discard card 1>-...
std::string HanabiState::ToString() const {
  std::string result;
  result += "Life tokens: " + std::to_string(LifeTokens()) + "\n";
  result += "Info tokens: " + std::to_string(InformationTokens()) + "\n";
  result += "Fireworks: ";
  for (int i = 0; i < ParentGame()->NumColors(); ++i) {
    result += ColorIndexToChar(i);
    result += std::to_string(fireworks_[i]) + " ";
  }
  result += "\nHands:\n";
  for (int i = 0; i < hands_.size(); ++i) {
    if (i > 0) {
      result += "-----\n";
    }
    if (i == CurPlayer()) {
      result += "Cur player\n";
    }
    result += hands_[i].ToString();
  }
  result += "Deck size: " + std::to_string(Deck().Size()) + "\n";
  result += "Discards:";
  for (int i = 0; i < discard_pile_.size(); ++i) {
    result += " " + discard_pile_[i].ToString();
  }
  return result;
}

int HanabiState::Score() const {
  if (LifeTokens() <= 0) {
    return 0;
  }
  return std::accumulate(fireworks_.begin(), fireworks_.end(), 0);
}

HanabiState::EndOfGameType HanabiState::EndOfGameStatus() const {
  if (LifeTokens() < 1) {
    return kOutOfLifeTokens;
  }
  if (Score() >= ParentGame()->NumColors() * ParentGame()->NumRanks()) {
    return kCompletedFireworks;
  }
  if (turns_to_play_ <= 0) {
    return kOutOfCards;
  }
  return kNotFinished;
}

/*=================================================================================
                 HanabiDeck Serialization + Deserialization
=================================================================================*/

json HanabiState::HanabiDeck::toJSON() const {
    json j;
    j["card_count"] = card_count_;
    j["total_count"] = total_count_;
    j["num_ranks"] = num_ranks_;
    return j;
}

void HanabiState::HanabiDeck::fromJSON(const json& j) {
    // Ensure the JSON contains all necessary fields
    assert(j.contains("card_count"));
    assert(j.contains("total_count"));
    assert(j.contains("num_ranks"));

    card_count_ = j.at("card_count").get<std::vector<int>>();
    total_count_ = j.at("total_count").get<int>();
    num_ranks_ = j.at("num_ranks").get<int>();
}

/*=================================================================================
                 HanabiState Serialization + Deserialization
=================================================================================*/

json HanabiState::toJSON() const {
    json j;

    // Serialize deck
    j["deck"] = deck_.toJSON();

    // Serialize discard_pile_
    j["discard_pile"] = json::array();
    for (const auto& card : discard_pile_) {
        j["discard_pile"].push_back(card.toJSON());
    }

    // Serialize hands_
    j["hands"] = json::array();
    for (const auto& hand : hands_) {
        j["hands"].push_back(hand.toJSON());
    }

    // Serialize move_history_
    j["move_history"] = json::array();
    for (const auto& history_item : move_history_) {
        j["move_history"].push_back(history_item.toJSON());
    }

    // Serialize other member variables
    j["cur_player"] = cur_player_;
    j["next_non_chance_player"] = next_non_chance_player_;
    j["information_tokens"] = information_tokens_;
    j["life_tokens"] = life_tokens_;
    j["fireworks"] = fireworks_;
    j["turns_to_play"] = turns_to_play_;

    return j;
}

HanabiState HanabiState::fromJSON(const json& j, const HanabiGame* game) {
    // Validate required fields
    assert(j.contains("deck"));
    assert(j.contains("discard_pile"));
    assert(j.contains("hands"));
    assert(j.contains("move_history"));
    assert(j.contains("cur_player"));
    assert(j.contains("next_non_chance_player"));
    assert(j.contains("information_tokens"));
    assert(j.contains("life_tokens"));
    assert(j.contains("fireworks"));
    assert(j.contains("turns_to_play"));

    // Initialize a new HanabiState with the parent_game and default start_player
    HanabiState state(game);

    // Deserialize deck
    state.deck_.fromJSON(j.at("deck"));

    // Deserialize discard_pile_
    state.discard_pile_.clear();
    for (const auto& card_json : j.at("discard_pile")) {
        state.discard_pile_.push_back(HanabiCard::fromJSON(card_json));
    }

    // Deserialize hands_
    state.hands_.clear();
    for (const auto& hand_json : j.at("hands")) {
        state.hands_.emplace_back(HanabiHand::fromJSON(hand_json));
    }

    // Deserialize move_history_
    state.move_history_.clear();
    for (const auto& history_json : j.at("move_history")) {
        state.move_history_.push_back(HanabiHistoryItem::fromJSON(history_json));
    }

    // Deserialize other member variables
    state.cur_player_ = j.at("cur_player").get<int>();
    state.next_non_chance_player_ = j.at("next_non_chance_player").get<int>();
    state.information_tokens_ = j.at("information_tokens").get<int>();
    state.life_tokens_ = j.at("life_tokens").get<int>();
    state.fireworks_ = j.at("fireworks").get<std::vector<int>>();
    state.turns_to_play_ = j.at("turns_to_play").get<int>();

    return state;
}

}  // namespace hanabi_learning_env
