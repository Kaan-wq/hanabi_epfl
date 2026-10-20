// Tests that the C API rejects bad input with a status instead of crashing:
// malformed or out-of-range JSON in the *FromJson functions, out-of-range moves
// in MoveIsLegal, and out-of-range indices, ids and parameters everywhere else.
// Run them under ASan + UBSan (Debug build).

#include <functional>
#include <iostream>
#include <nlohmann/json.hpp>
#include <sstream>
#include <string>
#include <vector>

#include "check.h"
#include "pyhanabi.h"

using json = nlohmann::json;

namespace {

constexpr int kNumPlayers = 3;

// Owns the C API objects of one test so every path frees them.
struct Fixture {
  pyhanabi_game_t game{};
  pyhanabi_state_t state{};

  Fixture() {
    const char* params[] = {"players", "3", "seed", "7"};
    CHECK(NewGame(&game, 4, params));
    NewState(&game, &state);
    DealUntilPlayerTurn();
  }
  ~Fixture() {
    DeleteState(&state);
    DeleteGame(&game);
  }
  Fixture(const Fixture&) = delete;
  Fixture& operator=(const Fixture&) = delete;

  void DealUntilPlayerTurn() {
    while (StateCurPlayer(&state) == -1 && StateDealCard(&state)) {
    }
  }

  // Applies a move built by `make`, which fills a zeroed pyhanabi_move_t.
  void Apply(const std::function<bool(pyhanabi_move_t*)>& make) {
    pyhanabi_move_t move{};
    make(&move);
    CHECK(StateApplyMove(&state, &move));
    DeleteMove(&move);
  }

  bool IsLegal(const std::function<bool(pyhanabi_move_t*)>& make) {
    pyhanabi_move_t move{};
    make(&move);
    const bool legal = MoveIsLegal(&state, &move);
    DeleteMove(&move);
    return legal;
  }
};

// Takes ownership of a string returned by the C API.
std::string Take(char* c_string) {
  std::string result = c_string == nullptr ? "" : c_string;
  DeleteString(c_string);
  return result;
}

std::string StateJson(Fixture& f) { return Take(StateToJson(&f.state)); }

const std::vector<std::string> kMalformed = {"", "{", "[]", "null", "42", "\"text\"", "{\"a\": }"};

// Every key of `j` removed in turn, and every value replaced by a string.
std::vector<json> BrokenVariants(const json& j) {
  std::vector<json> variants;
  for (const auto& item : j.items()) {
    json missing = j;
    missing.erase(item.key());
    variants.push_back(missing);
    json wrong_type = j;
    wrong_type[item.key()] = "wrong";
    variants.push_back(wrong_type);
  }
  return variants;
}

// Checks that loading `text` into a state fails and leaves the old state intact.
void CheckStateRejected(Fixture& f, const std::string& text) {
  const std::string before = StateJson(f);
  void* const handle = f.state.state;
  if (StateFromJson(text.c_str(), &f.state, &f.game)) {
    ++hanabi_test::FailureCount();
    std::cerr << "StateFromJson accepted: " << text.substr(0, 200) << '\n';
  }
  CHECK(f.state.state == handle);
  CHECK_EQ(StateJson(f), before);
}

}  // namespace

HANABI_TEST(FromJsonRejectsMalformedJson) {
  Fixture f;
  for (const std::string& text : kMalformed) {
    pyhanabi_move_t move{};
    CHECK(!MoveFromJson(text.c_str(), &move));
    CHECK(move.move == nullptr);
    pyhanabi_game_t game{};
    CHECK(!GameFromJson(text.c_str(), &game));
    CHECK(game.game == nullptr);
    pyhanabi_history_item_t item{};
    CHECK(!HistoryItemFromJson(text.c_str(), &item));
    CHECK(item.item == nullptr);
    CheckStateRejected(f, text);
  }
}

HANABI_TEST(FromJsonRoundTripsValidObjects) {
  Fixture f;
  f.Apply([](pyhanabi_move_t* m) { return GetRevealRankMove(1, 0, m); });
  f.DealUntilPlayerTurn();

  pyhanabi_history_item_t item{};
  StateGetMoveHistory(&f.state, StateLenMoveHistory(&f.state) - 1, &item);
  const std::string item_json = Take(HistoryItemToJson(&item));
  pyhanabi_history_item_t loaded_item{};
  CHECK(HistoryItemFromJson(item_json.c_str(), &loaded_item));
  CHECK_EQ(Take(HistoryItemToJson(&loaded_item)), item_json);

  pyhanabi_move_t move{};
  HistoryItemMove(&item, &move);
  const std::string move_json = Take(MoveToJson(&move));
  pyhanabi_move_t loaded_move{};
  CHECK(MoveFromJson(move_json.c_str(), &loaded_move));
  CHECK_EQ(Take(MoveToJson(&loaded_move)), move_json);

  const std::string game_json = Take(GameToJson(&f.game));
  pyhanabi_game_t loaded_game{};
  CHECK(GameFromJson(game_json.c_str(), &loaded_game));
  CHECK_EQ(Take(GameToJson(&loaded_game)), game_json);

  // Loading into a handle that already holds an object replaces it.
  const std::string state_json = StateJson(f);
  pyhanabi_state_t loaded_state{};
  NewState(&f.game, &loaded_state);
  CHECK(StateFromJson(state_json.c_str(), &loaded_state, &loaded_game));
  CHECK_EQ(Take(StateToJson(&loaded_state)), state_json);

  DeleteState(&loaded_state);
  DeleteGame(&loaded_game);
  DeleteMove(&loaded_move);
  DeleteMove(&move);
  DeleteHistoryItem(&loaded_item);
  DeleteHistoryItem(&item);
}

HANABI_TEST(FromJsonRejectsMissingFieldsAndWrongTypes) {
  Fixture f;
  f.Apply([](pyhanabi_move_t* m) { return GetRevealColorMove(1, 0, m); });
  f.DealUntilPlayerTurn();
  const json state = json::parse(StateJson(f));
  const json game = json::parse(Take(GameToJson(&f.game)));
  const json item = state["move_history"].back();
  const json& move = item["move"];

  for (const json& variant : BrokenVariants(move)) {
    pyhanabi_move_t loaded{};
    CHECK(!MoveFromJson(variant.dump().c_str(), &loaded));
  }
  for (const json& variant : BrokenVariants(item)) {
    pyhanabi_history_item_t loaded{};
    CHECK(!HistoryItemFromJson(variant.dump().c_str(), &loaded));
  }
  for (const json& variant : BrokenVariants(game)) {
    pyhanabi_game_t loaded{};
    if (GameFromJson(variant.dump().c_str(), &loaded)) {
      ++hanabi_test::FailureCount();
      std::cerr << "GameFromJson accepted: " << variant.dump().substr(0, 200) << '\n';
      DeleteGame(&loaded);
    }
  }
  std::vector<json> state_variants = BrokenVariants(state);
  for (const char* part : {"deck"}) {
    for (const json& sub : BrokenVariants(state[part])) {
      state_variants.push_back(state);
      state_variants.back()[part] = sub;
    }
  }
  for (const json& sub : BrokenVariants(state["hands"][0])) {
    state_variants.push_back(state);
    state_variants.back()["hands"][0] = sub;
  }
  for (const json& sub : BrokenVariants(state["hands"][0]["card_knowledge"][0]["color"])) {
    state_variants.push_back(state);
    state_variants.back()["hands"][0]["card_knowledge"][0]["color"] = sub;
  }
  for (const json& sub : BrokenVariants(state["move_history"][0])) {
    state_variants.push_back(state);
    state_variants.back()["move_history"][0] = sub;
  }
  // A float where an integer is expected.
  state_variants.push_back(state);
  state_variants.back()["life_tokens"] = 1.5;
  for (const json& variant : state_variants) CheckStateRejected(f, variant.dump());
}

HANABI_TEST(FromJsonRejectsOutOfRangeValues) {
  Fixture f;
  const json state = json::parse(StateJson(f));
  std::vector<std::function<void(json&)>> edits = {
      [](json& j) { j["hands"].erase(0); },                  // too few hands
      [](json& j) { j["hands"].push_back(j["hands"][0]); },  // too many hands
      [](json& j) { j["hands"][0]["cards"].push_back(j["hands"][0]["cards"][0]); },  // hand > size
      [](json& j) { j["hands"][0]["card_knowledge"].erase(0); },  // fewer knowledge than cards
      [](json& j) { j["hands"][0]["cards"][0]["color"] = 5; },
      [](json& j) { j["hands"][0]["cards"][0]["rank"] = -1; },
      [](json& j) { j["hands"][0]["card_knowledge"][0]["rank"]["value"] = 5; },
      [](json& j) { j["hands"][0]["card_knowledge"][0]["rank"]["value_plausible"].erase(0); },
      [](json& j) { j["deck"]["card_count"].erase(0); },
      [](json& j) { j["deck"]["card_count"][0] = -1; },
      [](json& j) { j["deck"]["total_count"] = j["deck"]["total_count"].get<int>() + 1; },
      [](json& j) { j["deck"]["num_ranks"] = 4; },
      [](json& j) { j["discard_pile"].push_back({{"color", 0}, {"rank", 9}}); },
      [](json& j) { j["fireworks"].erase(0); },
      [](json& j) { j["fireworks"][0] = 6; },
      [](json& j) { j["information_tokens"] = 9; },
      [](json& j) { j["life_tokens"] = -1; },
      [](json& j) { j["cur_player"] = kNumPlayers; },
      [](json& j) { j["cur_player"] = 1LL << 40; },
      [](json& j) { j["next_non_chance_player"] = -1; },
      [](json& j) { j["move_history"][0]["deal_to_player"] = kNumPlayers; },
      [](json& j) { j["move_history"][0]["move"]["move_type"] = 8; },
      [](json& j) { j["move_history"][0]["move"]["target_offset"] = kNumPlayers; },
      [](json& j) { j["move_history"][0]["reveal_bitmask"] = 256; },
  };
  for (const auto& edit : edits) {
    json variant = state;
    edit(variant);
    CheckStateRejected(f, variant.dump());
  }

  const json game = json::parse(Take(GameToJson(&f.game)));
  std::vector<std::function<void(json&)>> game_edits = {
      [](json& j) { j["num_players"] = 1; },
      [](json& j) { j["num_players"] = 6; },
      [](json& j) { j["num_colors"] = 0; },
      [](json& j) { j["num_ranks"] = 6; },
      [](json& j) { j["hand_size"] = 0; },
      [](json& j) { j["hand_size"] = 20; },  // more cards in hands than in the deck
      [](json& j) { j["hand_size"] = 9; },   // fits the deck; hands hold at most 8 cards
      [](json& j) { j["cards_per_color"] = 11; },
      [](json& j) { j["max_life_tokens"] = 0; },
      [](json& j) { j["max_information_tokens"] = -1; },
      [](json& j) { j["observation_type"] = 3; },
      [](json& j) { j["rng_state"] = "not an rng state"; },
      [](json& j) { j["params"] = {{"players", 3}}; },
      [](json& j) { j["moves"].erase(0); },
      [](json& j) {
        j["chance_outcomes"][0]["rank"] = 4 - j["chance_outcomes"][0]["rank"].get<int>();
      },
  };
  for (const auto& edit : game_edits) {
    json variant = game;
    edit(variant);
    pyhanabi_game_t loaded{};
    if (GameFromJson(variant.dump().c_str(), &loaded)) {
      ++hanabi_test::FailureCount();
      std::cerr << "GameFromJson accepted: " << variant.dump().substr(0, 200) << '\n';
      DeleteGame(&loaded);
    }
  }

  const json move = {
      {"move_type", 1}, {"card_index", 0}, {"target_offset", -1}, {"color", -1}, {"rank", -1}};
  std::vector<std::function<void(json&)>> move_edits = {
      [](json& j) { j["move_type"] = -1; },
      [](json& j) { j["move_type"] = 8; },
      [](json& j) { j["card_index"] = 128; },
      [](json& j) { j["color"] = 5; },
  };
  for (const auto& edit : move_edits) {
    json variant = move;
    edit(variant);
    pyhanabi_move_t loaded{};
    CHECK(!MoveFromJson(variant.dump().c_str(), &loaded));
  }
}

HANABI_TEST(FailedLoadKeepsExistingObjects) {
  Fixture f;
  const std::string game_json = Take(GameToJson(&f.game));
  void* const game_handle = f.game.game;
  CHECK(!GameFromJson("{}", &f.game));
  CHECK(f.game.game == game_handle);
  CHECK_EQ(Take(GameToJson(&f.game)), game_json);

  pyhanabi_move_t move{};
  GetPlayMove(0, &move);
  void* const move_handle = move.move;
  CHECK(!MoveFromJson("{\"move_type\": 1}", &move));
  CHECK(move.move == move_handle);
  CHECK_EQ(MoveType(&move), 1);
  DeleteMove(&move);
}

HANABI_TEST(MoveIsLegalRejectsOutOfRangeReturnMoves) {
  Fixture f;
  const int hand_size = StateGetHandSize(&f.state, 0);
  CHECK(f.IsLegal([](pyhanabi_move_t* m) { return GetReturnMove(0, 2, m); }));
  for (int player : {-1, kNumPlayers, 100}) {
    CHECK(!f.IsLegal([player](pyhanabi_move_t* m) { return GetReturnMove(0, player, m); }));
  }
  for (int index : {-1, hand_size, 100}) {
    CHECK(!f.IsLegal([index](pyhanabi_move_t* m) { return GetReturnMove(index, 0, m); }));
  }
}

HANABI_TEST(MoveIsLegalRejectsOutOfRangeDealSpecificMoves) {
  Fixture f;
  // Not a chance turn yet.
  CHECK(!f.IsLegal([](pyhanabi_move_t* m) { return GetDealSpecificMove(0, 1, 0, 0, m); }));

  // Returning player 1's first card makes it chance's turn to deal it back.
  pyhanabi_card_t card{};
  StateGetHandCard(&f.state, 1, 0, &card);
  f.Apply([](pyhanabi_move_t* m) { return GetReturnMove(0, 1, m); });
  CHECK_EQ(StateCurPlayer(&f.state), -1);
  const int color = card.color;
  const int rank = card.rank;
  const int cards_left = StateGetHandSize(&f.state, 1);

  CHECK(f.IsLegal([&](pyhanabi_move_t* m) { return GetDealSpecificMove(0, 1, color, rank, m); }));
  CHECK(f.IsLegal(
      [&](pyhanabi_move_t* m) { return GetDealSpecificMove(cards_left, 1, color, rank, m); }));

  std::ostringstream captured;
  std::streambuf* const old_cout = std::cout.rdbuf(captured.rdbuf());
  for (int player : {-1, kNumPlayers}) {  // bad player
    CHECK(!f.IsLegal(
        [&](pyhanabi_move_t* m) { return GetDealSpecificMove(0, player, color, rank, m); }));
  }
  for (int index : {-1, cards_left + 1}) {  // bad insert position
    CHECK(!f.IsLegal(
        [&](pyhanabi_move_t* m) { return GetDealSpecificMove(index, 1, color, rank, m); }));
  }
  // Player 0's hand is full.
  CHECK(!f.IsLegal([&](pyhanabi_move_t* m) { return GetDealSpecificMove(0, 0, color, rank, m); }));
  for (int bad_color : {-1, 5}) {
    CHECK(!f.IsLegal(
        [&](pyhanabi_move_t* m) { return GetDealSpecificMove(0, 1, bad_color, rank, m); }));
  }
  for (int bad_rank : {-1, 5}) {
    CHECK(!f.IsLegal(
        [&](pyhanabi_move_t* m) { return GetDealSpecificMove(0, 1, color, bad_rank, m); }));
  }
  std::cout.rdbuf(old_cout);
  CHECK(captured.str().empty());  // Library code does not print.
}

HANABI_TEST(MoveIsLegalRejectsOutOfRangePlayerMoves) {
  Fixture f;
  // A hint spends an information token, so discarding becomes legal.
  pyhanabi_card_t card{};
  StateGetHandCard(&f.state, 1, 0, &card);
  const int rank = card.rank;
  f.Apply([rank](pyhanabi_move_t* m) { return GetRevealRankMove(1, rank, m); });

  CHECK(f.IsLegal([](pyhanabi_move_t* m) { return GetPlayMove(0, m); }));
  CHECK(f.IsLegal([](pyhanabi_move_t* m) { return GetDiscardMove(0, m); }));
  CHECK(!f.IsLegal([](pyhanabi_move_t* m) { return GetPlayMove(-1, m); }));
  CHECK(!f.IsLegal([](pyhanabi_move_t* m) { return GetDiscardMove(-1, m); }));

  // On a chance turn, no player can play or discard.
  f.Apply([](pyhanabi_move_t* m) { return GetReturnMove(0, 2, m); });
  CHECK_EQ(StateCurPlayer(&f.state), -1);
  CHECK(!f.IsLegal([](pyhanabi_move_t* m) { return GetPlayMove(0, m); }));
  CHECK(!f.IsLegal([](pyhanabi_move_t* m) { return GetDiscardMove(0, m); }));
}

HANABI_TEST(MoveFunctionsRejectBadArguments) {
  pyhanabi_move_t move{};
  // Values that do not fit the move's 8-bit fields.
  CHECK(!GetPlayMove(128, &move));
  CHECK(!GetDiscardMove(-129, &move));
  CHECK(!GetReturnMove(0, 300, &move));
  CHECK(!GetDealSpecificMove(0, 0, 1000, 0, &move));
  CHECK(!GetRevealColorMove(1, 128, &move));
  CHECK(!GetRevealRankMove(200, 0, &move));
  CHECK(move.move == nullptr);

  Fixture f;
  void* list = StateLegalMoves(&f.state);
  CHECK(!GetMove(list, -1, &move));
  CHECK(!GetMove(list, NumMoves(list), &move));
  CHECK(move.move == nullptr);
  CHECK(GetMove(list, 0, &move));
  DeleteMove(&move);
  DeleteMoveList(list);

  CHECK(!GetMoveByUid(&f.game, -1, &move));
  CHECK(!GetMoveByUid(&f.game, MaxMoves(&f.game), &move));
  CHECK(move.move == nullptr);
  CHECK(GetMoveByUid(&f.game, MaxMoves(&f.game) - 1, &move));
  DeleteMove(&move);
}

HANABI_TEST(StateFunctionsRejectBadArguments) {
  Fixture f;
  pyhanabi_card_t card{};
  const int hand_size = StateGetHandSize(&f.state, 0);
  CHECK_EQ(hand_size, 5);
  for (int player : {-1, kNumPlayers}) {
    CHECK_EQ(StateGetHandSize(&f.state, player), -1);
    CHECK(!StateGetHandCard(&f.state, player, 0, &card));
    CHECK(!StateRemoveKnowledge(&f.state, player, 0));
  }
  for (int index : {-1, hand_size}) {
    CHECK(!StateGetHandCard(&f.state, 0, index, &card));
    CHECK(!StateRemoveKnowledge(&f.state, 0, index));
  }
  CHECK(StateGetHandCard(&f.state, 0, hand_size - 1, &card));
  CHECK(StateRemoveKnowledge(&f.state, 0, hand_size - 1));

  for (int color : {-1, 5}) CHECK_EQ(StateFireworks(&f.state, color), -1);
  CHECK_EQ(StateFireworks(&f.state, 4), 0);
  CHECK(!StateGetDiscard(&f.state, 0, &card));  // The discard pile is empty.
  CHECK(!StateGetDiscard(&f.state, -1, &card));

  pyhanabi_history_item_t item{};
  const int history_length = StateLenMoveHistory(&f.state);
  CHECK(!StateGetMoveHistory(&f.state, history_length, &item));
  CHECK(!StateGetMoveHistory(&f.state, -1, &item));
  CHECK(item.item == nullptr);
  CHECK(StateGetMoveHistory(&f.state, history_length - 1, &item));
  DeleteHistoryItem(&item);
}

HANABI_TEST(StateApplyMoveAndDealCardRejectIllegalCalls) {
  Fixture f;
  const std::string before = StateJson(f);
  pyhanabi_move_t move{};
  CHECK(GetPlayMove(7, &move));  // Beyond the hand.
  CHECK(!StateApplyMove(&f.state, &move));
  DeleteMove(&move);
  CHECK(!StateDealCard(&f.state));  // Not a chance turn.
  CHECK_EQ(StateJson(f), before);

  // A chance turn with an empty deck (a state only JSON can produce).
  json state = json::parse(before);
  state["cur_player"] = -1;
  for (auto& count : state["deck"]["card_count"]) count = 0;
  state["deck"]["total_count"] = 0;
  CHECK(StateFromJson(state.dump().c_str(), &f.state, &f.game));
  CHECK(!StateDealCard(&f.state));
}

HANABI_TEST(NewGameRejectsBadParameters) {
  const std::vector<std::vector<const char*>> bad_params = {
      {"players", "1"},
      {"players", "6"},
      {"players", "two"},
      {"colors", "0"},
      {"ranks", "6"},
      {"hand_size", "0"},
      {"hand_size", "26"},  // 2 x 26 > 50 cards
      {"max_life_tokens", "0"},
      {"max_information_tokens", "-1"},
      {"observation_type", "3"},
      {"players"},  // A key without a value.
  };
  for (auto params : bad_params) {  // A copy: NewGame takes a non-const array.
    pyhanabi_game_t game{};
    CHECK(!NewGame(&game, static_cast<int>(params.size()), params.data()));
    CHECK(game.game == nullptr);
  }
  pyhanabi_game_t game{};
  CHECK(!NewGame(&game, -2, nullptr));
  const char* good_params[] = {"players", "4"};
  CHECK(NewGame(&game, 2, good_params));
  CHECK_EQ(NumPlayers(&game), 4);

  pyhanabi_observation_encoder_t encoder{};
  CHECK(!NewObservationEncoder(&encoder, &game, 1));
  CHECK(!NewObservationEncoder(&encoder, &game, -1));
  CHECK(encoder.encoder == nullptr);
  CHECK(NewObservationEncoder(&encoder, &game, 0));
  DeleteObservationEncoder(&encoder);
  DeleteGame(&game);
}

HANABI_TEST(ObservationFunctionsRejectBadArguments) {
  Fixture f;
  pyhanabi_observation_t obs{};
  CHECK(!NewObservation(&f.state, -1, &obs));
  CHECK(!NewObservation(&f.state, kNumPlayers, &obs));
  CHECK(obs.observation == nullptr);
  CHECK(NewObservation(&f.state, 0, &obs));

  pyhanabi_card_t card{};
  pyhanabi_card_knowledge_t knowledge{};
  for (int offset : {-1, kNumPlayers}) {
    CHECK_EQ(ObsGetHandSize(&obs, offset), -1);
    CHECK(!ObsGetHandCard(&obs, offset, 0, &card));
    CHECK(!ObsGetHandCardKnowledge(&obs, offset, 0, &knowledge));
  }
  for (int index : {-1, ObsGetHandSize(&obs, 1)}) {
    CHECK(!ObsGetHandCard(&obs, 1, index, &card));
    CHECK(!ObsGetHandCardKnowledge(&obs, 1, index, &knowledge));
  }
  CHECK(knowledge.knowledge == nullptr);
  CHECK(ObsGetHandCardKnowledge(&obs, 1, 0, &knowledge));
  for (int value : {-1, 5}) {
    CHECK_EQ(ColorIsPlausible(&knowledge, value), -1);
    CHECK_EQ(RankIsPlausible(&knowledge, value), -1);
  }
  CHECK_EQ(ColorIsPlausible(&knowledge, 4), 1);
  CHECK_EQ(RankIsPlausible(&knowledge, 4), 1);

  for (int color : {-1, 5}) CHECK_EQ(ObsFireworks(&obs, color), -1);
  CHECK(!ObsGetDiscard(&obs, 0, &card));

  pyhanabi_history_item_t item{};
  CHECK(!ObsGetLastMove(&obs, ObsNumLastMoves(&obs), &item));
  CHECK(!ObsGetLastMove(&obs, -1, &item));
  CHECK(item.item == nullptr);
  pyhanabi_move_t move{};
  CHECK(!ObsGetLegalMove(&obs, ObsNumLegalMoves(&obs), &move));
  CHECK(!ObsGetLegalMove(&obs, -1, &move));
  CHECK(move.move == nullptr);
  CHECK(ObsGetLegalMove(&obs, 0, &move));
  DeleteMove(&move);
  DeleteObservation(&obs);
}

HANABI_TEST(HandSamplingFunctionsRejectBadArguments) {
  Fixture f;
  std::vector<int32_t> counts(25, -1);
  const auto candidates = [&](int player, int index, bool knowledge,
                              const pyhanabi_card_t* excluded, int num_excluded, int size) {
    return pyhanabi_state_candidate_counts(&f.state, player, index, knowledge, excluded,
                                           num_excluded, counts.data(), size);
  };
  CHECK(candidates(1, 0, true, nullptr, 0, 25));
  int total = 0;
  for (int32_t count : counts) total += count;
  CHECK_EQ(total, StateDeckSize(&f.state) + 1);  // The deck plus the card at position 0.

  const pyhanabi_card_t bad_card = {5, 0};
  CHECK(!candidates(-1, 0, true, nullptr, 0, 25));
  CHECK(!candidates(kNumPlayers, 0, true, nullptr, 0, 25));
  CHECK(!candidates(1, -1, false, nullptr, 0, 25));
  CHECK(!candidates(1, 5, false, nullptr, 0, 25));  // Hand size is 5.
  CHECK(!candidates(1, 0, true, nullptr, 0, 24));
  CHECK(!candidates(1, 0, true, &bad_card, 1, 25));
  CHECK(!candidates(1, 0, true, nullptr, -1, 25));

  const std::string before = StateJson(f);
  CHECK(!pyhanabi_state_resample_hand(&f.state, kNumPlayers, 1));
  pyhanabi_card_t remembered[5];
  for (int i = 0; i < 5; ++i) StateGetHandCard(&f.state, 1, i, &remembered[i]);
  CHECK(!pyhanabi_state_restore_hand(&f.state, -1, remembered, 5, -1, -1, -1, 1));
  CHECK(!pyhanabi_state_restore_hand(&f.state, 1, remembered, 6, -1, -1, -1, 1));
  CHECK(!pyhanabi_state_restore_hand(&f.state, 1, remembered, 5, -1, -1, 5, 1));
  CHECK(!pyhanabi_state_restore_hand(&f.state, 1, remembered, 5, 7, 0, 0, 1));
  CHECK(!pyhanabi_state_restore_hand(&f.state, 1, &bad_card, 1, -1, -1, -1, 1));
  CHECK_EQ(StateJson(f), before);

  CHECK(pyhanabi_state_resample_hand(&f.state, 1, 1));
  CHECK(pyhanabi_state_restore_hand(&f.state, 1, remembered, 5, -1, -1, -1, 1));
  for (int i = 0; i < 5; ++i) {
    pyhanabi_card_t card{};
    StateGetHandCard(&f.state, 1, i, &card);
    CHECK(card.color == remembered[i].color && card.rank == remembered[i].rank);
  }
}

HANABI_TEST(PlayDeterminizedRejectsBadArgumentsAndStopsAtIllegalMoves) {
  Fixture f;
  pyhanabi_card_t remembered[5] = {};
  int32_t num_remembered = -1;
  const auto play = [&](const std::vector<int32_t>& uids, int searcher, int determinization) {
    return pyhanabi_state_play_determinized(&f.state, uids.data(),
                                            static_cast<int32_t>(uids.size()), searcher,
                                            determinization, remembered, &num_remembered, 1);
  };
  const std::string before = StateJson(f);
  CHECK_EQ(play({0}, kNumPlayers, 0), -1);
  CHECK_EQ(play({0}, 0, 3), -1);
  CHECK_EQ(play({-1}, 0, 0), -1);
  num_remembered = 6;
  CHECK_EQ(play({0}, 0, 0), -1);
  num_remembered = -1;
  CHECK_EQ(StateJson(f), before);

  // Discarding (uid 0) is illegal while every information token is left.
  CHECK_EQ(play({0}, 0, 0), 0);
  CHECK_EQ(StateJson(f), before);
  // The first legal hint (hints follow the 5 discards and 5 plays).
  const auto hint = [&] {
    std::vector<int32_t> uids(64);
    const int32_t count = pyhanabi_state_legal_move_uids(&f.state, uids.data(), 64);
    for (int32_t i = 0; i < count; ++i) {
      if (uids[i] >= 10) return uids[i];
    }
    return int32_t{-1};
  };
  CHECK_EQ(play({hint()}, 0, 0), 1);
  CHECK_EQ(num_remembered, 5);  // Player 1 acts next: their hand was remembered.
  CHECK_EQ(play({hint()}, 0, 0), 1);

  // Restoring before anything was remembered is an error.
  num_remembered = -1;
  CHECK_EQ(play({hint()}, 0, 0), -1);
}

// Whether pyhanabi_last_error() mentions `text`.
bool LastErrorMentions(const std::string& text) {
  const std::string error = pyhanabi_last_error();
  if (error.find(text) != std::string::npos) return true;
  std::cerr << "last error \"" << error << "\" does not mention \"" << text << "\"\n";
  return false;
}

HANABI_TEST(FailuresExplainThemselves) {
  Fixture f;

  pyhanabi_game_t game{};
  const char* bad_params[] = {"players", "3", "hand_size", "9"};
  CHECK(!NewGame(&game, 4, bad_params));
  CHECK(LastErrorMentions("hand_size must be in [1, 8]"));
  CHECK(!NewGame(&game, 3, bad_params));
  CHECK(LastErrorMentions("must be even"));

  pyhanabi_card_t card{};
  CHECK(!StateGetHandCard(&f.state, kNumPlayers, 0, &card));
  CHECK(LastErrorMentions("player 3 is not in [0, 3)"));
  CHECK(!StateGetHandCard(&f.state, 0, 9, &card));
  CHECK(LastErrorMentions("card index 9 is not in [0, 5)"));
  CHECK_EQ(StateFireworks(&f.state, 7), -1);
  CHECK(LastErrorMentions("color 7"));

  pyhanabi_move_t move{};
  CHECK(GetPlayMove(0, &move));
  pyhanabi_state_t copy{};
  CopyState(&f.state, &copy);
  while (StateEndOfGameStatus(&copy) == 0 && StateCurPlayer(&copy) != -1) {
    StateApplyMove(&copy, &move);  // Plays until a deal is pending or the game ends.
  }
  if (StateCurPlayer(&copy) == -1) {
    CHECK(!StateApplyMove(&copy, &move));
    CHECK(LastErrorMentions("is not legal"));
  }
  DeleteState(&copy);
  DeleteMove(&move);
  CHECK(!StateDealCard(&f.state));
  CHECK(LastErrorMentions("not a chance turn"));

  CHECK(!StateFromJson("{\"cur_player\": 1}", &f.state, &f.game));
  CHECK(LastErrorMentions("invalid JSON"));
  CHECK(!pyhanabi_state_resample_hand(&f.state, -1, 0));
  CHECK(LastErrorMentions("player -1 is not in [0, 3)"));
  int32_t uids[1];
  CHECK_EQ(pyhanabi_state_legal_move_uids(&f.state, uids, 1), -1);
  CHECK(LastErrorMentions("below MaxMoves"));

  pyhanabi_observation_t observation{};
  CHECK(!NewObservation(&f.state, 5, &observation));
  CHECK(LastErrorMentions("player 5 is not in [0, 3)"));
}

int main() { return hanabi_test::RunAll(); }
