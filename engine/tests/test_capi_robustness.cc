// Tests that the C API rejects bad input with a status instead of crashing:
// malformed or out-of-range JSON in the *FromJson functions, and out-of-range
// moves in MoveIsLegal. Run them under ASan + UBSan (Debug build).

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
    NewGame(&game, 4, params);
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
    while (StateCurPlayer(&state) == -1) StateDealCard(&state);
  }

  // Applies a move built by `make`, which fills a zeroed pyhanabi_move_t.
  void Apply(const std::function<bool(pyhanabi_move_t*)>& make) {
    pyhanabi_move_t move{};
    make(&move);
    StateApplyMove(&state, &move);
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

int main() { return hanabi_test::RunAll(); }
