// Tests for the observation and state buffer functions of the C API: each
// buffer must match the per-card getters, observation buffers must never
// reveal the observer's own cards, and wrong sizes must be rejected without
// writing. Run them under ASan + UBSan too.

#include <cstddef>
#include <cstdint>
#include <nlohmann/json.hpp>
#include <random>
#include <string>
#include <vector>

#include "check.h"
#include "pyhanabi.h"

using json = nlohmann::json;

namespace {

// Number of elements of an array with these (non-negative) dimensions.
size_t Size(int a, int b = 1, int c = 1, int d = 1) { return static_cast<size_t>(a) * b * c * d; }

// Owns one game and its state.
struct Game {
  pyhanabi_game_t game{};
  pyhanabi_state_t state{};
  int players, hand_size, colors, ranks;

  explicit Game(int num_players, int num_colors = 5) {
    const std::string players_str = std::to_string(num_players);
    const std::string colors_str = std::to_string(num_colors);
    const char* params[] = {"players", players_str.c_str(), "colors", colors_str.c_str(), "seed",
                            "11"};
    CHECK(NewGame(&game, 6, params));
    NewState(&game, &state);
    players = NumPlayers(&game);
    hand_size = HandSize(&game);
    colors = NumColors(&game);
    ranks = NumRanks(&game);
    Deal();
  }
  ~Game() {
    DeleteState(&state);
    DeleteGame(&game);
  }
  Game(const Game&) = delete;
  Game& operator=(const Game&) = delete;

  void Deal() {
    while (StateCurPlayer(&state) == -1 && StateDealCard(&state)) {
    }
  }

  // Plays a random legal move; false once the game is over.
  bool Step(std::mt19937* rng) {
    if (StateEndOfGameStatus(&state) != 0) return false;
    void* moves = StateLegalMoves(&state);
    pyhanabi_move_t move{};
    CHECK(GetMove(moves, static_cast<int>((*rng)() % NumMoves(moves)), &move));
    CHECK(StateApplyMove(&state, &move));
    DeleteMove(&move);
    DeleteMoveList(moves);
    Deal();
    return true;
  }
};

// Owns one observation.
struct Obs {
  pyhanabi_observation_t observation{};
  Obs(Game* game, int player) { CHECK(NewObservation(&game->state, player, &observation)); }
  ~Obs() { DeleteObservation(&observation); }
  Obs(const Obs&) = delete;
  Obs& operator=(const Obs&) = delete;
};

// Every buffer of one observation, for comparing two observations.
struct Buffers {
  std::vector<int32_t> hand_sizes;
  std::vector<int8_t> colors, ranks, hinted_colors, hinted_ranks, discard_colors, discard_ranks;
  std::vector<uint8_t> plausible, bits;
  std::vector<int32_t> fireworks, uids, unseen;

  bool operator==(const Buffers& other) const {
    return hand_sizes == other.hand_sizes && colors == other.colors && ranks == other.ranks &&
           hinted_colors == other.hinted_colors && hinted_ranks == other.hinted_ranks &&
           discard_colors == other.discard_colors && discard_ranks == other.discard_ranks &&
           plausible == other.plausible && bits == other.bits && fireworks == other.fireworks &&
           uids == other.uids && unseen == other.unseen;
  }
};

int EncodingLength(pyhanabi_observation_encoder_t* encoder) {
  char* shape = ObservationShape(encoder);
  const int length = std::stoi(shape);  // The canonical encoding is one-dimensional.
  DeleteString(shape);
  return length;
}

Buffers Fill(Game* game, pyhanabi_observation_t* observation) {
  const int p = game->players, h = game->hand_size, c = game->colors, r = game->ranks;
  Buffers b;
  b.hand_sizes.resize(p);
  b.colors.resize(Size(p, h));
  b.ranks.resize(Size(p, h));
  CHECK(pyhanabi_observation_hands(observation, b.hand_sizes.data(), b.colors.data(),
                                   b.ranks.data(), p, h));
  b.hinted_colors.resize(Size(p, h));
  b.hinted_ranks.resize(Size(p, h));
  b.plausible.resize(Size(p, h, c, r));
  CHECK(pyhanabi_observation_card_knowledge(observation, b.hinted_colors.data(),
                                            b.hinted_ranks.data(), b.plausible.data(), p, h, c, r));
  int deck = 0;
  for (int card = 0; card < c * r; ++card) deck += NumCards(&game->game, card / r, card % r);
  b.discard_colors.resize(deck);
  b.discard_ranks.resize(deck);
  const int32_t num_discards = pyhanabi_observation_discard_pile(
      observation, b.discard_colors.data(), b.discard_ranks.data(), deck);
  CHECK(num_discards >= 0);
  b.discard_colors.resize(num_discards);
  b.discard_ranks.resize(num_discards);
  b.fireworks.resize(c);
  CHECK(pyhanabi_observation_fireworks(observation, b.fireworks.data(), c));
  b.uids.resize(MaxMoves(&game->game));
  const int32_t num_uids = pyhanabi_observation_legal_move_uids(
      observation, b.uids.data(), static_cast<int32_t>(b.uids.size()));
  CHECK(num_uids >= 0);
  b.uids.resize(num_uids);
  b.unseen.resize(Size(c, r));
  CHECK(pyhanabi_observation_unseen_card_counts(observation, b.unseen.data(), c * r));
  pyhanabi_observation_encoder_t encoder{};
  CHECK(NewObservationEncoder(&encoder, &game->game, 0));
  b.bits.resize(EncodingLength(&encoder));
  CHECK(pyhanabi_observation_encode(&encoder, observation, b.bits.data(),
                                    static_cast<int32_t>(b.bits.size())));
  DeleteObservationEncoder(&encoder);
  return b;
}

// Checks one observation's buffers against the per-card getters.
void CheckAgainstGetters(Game* game, pyhanabi_observation_t* observation) {
  const int h = game->hand_size, c = game->colors, r = game->ranks;
  const Buffers b = Fill(game, observation);
  for (int offset = 0; offset < game->players; ++offset) {
    const int size = ObsGetHandSize(observation, offset);
    CHECK_EQ(b.hand_sizes[offset], size);
    for (int index = 0; index < h; ++index) {
      const int i = offset * h + index;
      if (index >= size) {
        CHECK(b.colors[i] == -1 && b.ranks[i] == -1);
        CHECK(b.hinted_colors[i] == -1 && b.hinted_ranks[i] == -1);
        for (int card = 0; card < c * r; ++card) CHECK_EQ(b.plausible[i * c * r + card], 0);
        continue;
      }
      pyhanabi_card_t card{};
      CHECK(ObsGetHandCard(observation, offset, index, &card));
      CHECK_EQ(b.colors[i], card.color);
      CHECK_EQ(b.ranks[i], card.rank);
      if (offset == 0) CHECK(card.color == -1 && card.rank == -1);  // Own cards stay hidden.
      pyhanabi_card_knowledge_t knowledge{};
      CHECK(ObsGetHandCardKnowledge(observation, offset, index, &knowledge));
      CHECK_EQ(b.hinted_colors[i], ColorWasHinted(&knowledge) ? KnownColor(&knowledge) : -1);
      CHECK_EQ(b.hinted_ranks[i], RankWasHinted(&knowledge) ? KnownRank(&knowledge) : -1);
      for (int color = 0; color < c; ++color) {
        for (int rank = 0; rank < r; ++rank) {
          const bool expected =
              ColorIsPlausible(&knowledge, color) == 1 && RankIsPlausible(&knowledge, rank) == 1;
          CHECK_EQ(b.plausible[i * c * r + color * r + rank], expected ? 1 : 0);
        }
      }
    }
  }
  CHECK_EQ(static_cast<int>(b.discard_colors.size()), ObsDiscardPileSize(observation));
  for (int index = 0; index < static_cast<int>(b.discard_colors.size()); ++index) {
    pyhanabi_card_t card{};
    CHECK(ObsGetDiscard(observation, index, &card));
    CHECK(b.discard_colors[index] == card.color && b.discard_ranks[index] == card.rank);
  }
  for (int color = 0; color < c; ++color) {
    CHECK_EQ(b.fireworks[color], ObsFireworks(observation, color));
  }
  CHECK_EQ(static_cast<int>(b.uids.size()), ObsNumLegalMoves(observation));
  for (int index = 0; index < static_cast<int>(b.uids.size()); ++index) {
    pyhanabi_move_t move{};
    CHECK(ObsGetLegalMove(observation, index, &move));
    CHECK_EQ(b.uids[index], GetMoveUid(&game->game, &move));
    DeleteMove(&move);
  }
  pyhanabi_observation_encoder_t encoder{};
  CHECK(NewObservationEncoder(&encoder, &game->game, 0));
  char* encoded = EncodeObservation(&encoder, observation);
  std::string expected_bits;
  for (uint8_t bit : b.bits)
    expected_bits += std::string(expected_bits.empty() ? "" : ",") + (bit ? "1" : "0");
  CHECK_EQ(std::string(encoded), expected_bits);
  DeleteString(encoded);
  DeleteObservationEncoder(&encoder);
}

// The deck plus the player's own hand, from the state (which sees everything).
std::vector<int32_t> TrueUnseen(Game* game, int player) {
  std::vector<int32_t> counts(Size(game->colors, game->ranks));
  // Candidate counts without knowledge: the deck plus the card at position 0.
  CHECK(pyhanabi_state_candidate_counts(&game->state, player, 0, false, nullptr, 0, counts.data(),
                                        static_cast<int32_t>(counts.size())));
  for (int index = 1; index < StateGetHandSize(&game->state, player); ++index) {
    pyhanabi_card_t card{};
    CHECK(StateGetHandCard(&game->state, player, index, &card));
    ++counts[card.color * game->ranks + card.rank];
  }
  return counts;
}

}  // namespace

HANABI_TEST(BuffersMatchGettersAlongRandomGames) {
  for (int players = 2; players <= 5; ++players) {
    for (int colors : {5, 2}) {
      Game game(players, colors);
      std::mt19937 rng(players * 10 + colors);
      do {
        for (int player = 0; player < players; ++player) {
          Obs obs(&game, player);
          CheckAgainstGetters(&game, &obs.observation);
        }
      } while (game.Step(&rng));
    }
  }
}

HANABI_TEST(UnseenCountsAreTheDeckAndOwnHand) {
  for (int players = 2; players <= 5; ++players) {
    Game game(players);
    std::mt19937 rng(players);
    for (int step = 0; step < 30 && game.Step(&rng); ++step) {
      for (int player = 0; player < players; ++player) {
        Obs obs(&game, player);
        CHECK(Fill(&game, &obs.observation).unseen == TrueUnseen(&game, player));
      }
    }
  }
}

HANABI_TEST(OwnCardsDoNotChangeAnyBuffer) {
  // Swapping one of the observer's cards with a different card from the deck
  // changes nothing the observer is told.
  Game game(3);
  std::mt19937 rng(5);
  for (int step = 0; step < 8; ++step) game.Step(&rng);
  const int player = StateCurPlayer(&game.state);
  CHECK(player >= 0);
  if (player < 0) return;
  Obs before(&game, player);
  const Buffers expected = Fill(&game, &before.observation);

  char* state_json = StateToJson(&game.state);
  json state = json::parse(state_json);
  DeleteString(state_json);
  json& own = state["hands"][player]["cards"][0];
  json& deck = state["deck"]["card_count"];
  const int old_card = own["color"].get<int>() * game.ranks + own["rank"].get<int>();
  int new_card = 0;
  while (new_card == old_card || deck[new_card].get<int>() == 0) ++new_card;
  own["color"] = new_card / game.ranks;
  own["rank"] = new_card % game.ranks;
  deck[new_card] = deck[new_card].get<int>() - 1;
  deck[old_card] = deck[old_card].get<int>() + 1;
  CHECK(StateFromJson(state.dump().c_str(), &game.state, &game.game));

  pyhanabi_card_t card{};
  CHECK(StateGetHandCard(&game.state, player, 0, &card));
  CHECK_EQ(card.color * game.ranks + card.rank, new_card);
  Obs after(&game, player);
  CHECK(Fill(&game, &after.observation) == expected);
}

HANABI_TEST(WrongSizesAreRejectedWithoutWriting) {
  Game game(3);
  Obs obs(&game, 0);
  pyhanabi_observation_t* o = &obs.observation;
  const int p = game.players, h = game.hand_size, c = game.colors, r = game.ranks;
  std::vector<int32_t> sizes(p + 1, 7);
  std::vector<int8_t> colors(Size(p + 1, h + 1), 7), ranks(Size(p + 1, h + 1), 7);
  std::vector<uint8_t> plausible(Size(p + 1, h + 1, c, r + 1), 7);
  CHECK(!pyhanabi_observation_hands(o, sizes.data(), colors.data(), ranks.data(), p + 1, h));
  CHECK(!pyhanabi_observation_hands(o, sizes.data(), colors.data(), ranks.data(), p, h - 1));
  CHECK(!pyhanabi_observation_card_knowledge(o, colors.data(), ranks.data(), plausible.data(), p, h,
                                             c, r + 1));
  CHECK(!pyhanabi_observation_card_knowledge(o, colors.data(), ranks.data(), plausible.data(), p, h,
                                             c - 1, r));
  std::vector<int32_t> ints(Size(c, r) + 1, 7);
  CHECK(!pyhanabi_observation_fireworks(o, ints.data(), c + 1));
  CHECK(!pyhanabi_observation_unseen_card_counts(o, ints.data(), c * r + 1));
  CHECK_EQ(pyhanabi_observation_legal_move_uids(o, ints.data(), 1), -1);
  CHECK_EQ(pyhanabi_observation_legal_move_uids(o, ints.data(), -1), -1);
  pyhanabi_observation_encoder_t encoder{};
  CHECK(NewObservationEncoder(&encoder, &game.game, 0));
  CHECK(!pyhanabi_observation_encode(&encoder, o, plausible.data(), EncodingLength(&encoder) - 1));
  DeleteObservationEncoder(&encoder);
  for (int32_t v : sizes) CHECK_EQ(v, 7);
  for (int8_t v : colors) CHECK_EQ(v, 7);
  for (int8_t v : ranks) CHECK_EQ(v, 7);
  for (uint8_t v : plausible) CHECK_EQ(v, 7);
  for (int32_t v : ints) CHECK_EQ(v, 7);

  // A full game discards at most the deck; a capacity of 0 fails once one is discarded.
  std::mt19937 rng(1);
  while (ObsDiscardPileSize(o) == 0 && game.Step(&rng)) {
    DeleteObservation(o);
    CHECK(NewObservation(&game.state, 0, o));
  }
  CHECK_EQ(pyhanabi_observation_discard_pile(o, colors.data(), ranks.data(), 0), -1);
}

HANABI_TEST(StateBuffersMatchGettersAlongRandomGames) {
  for (int players = 2; players <= 5; ++players) {
    for (int colors : {5, 2}) {
      Game game(players, colors);
      const int h = game.hand_size;
      const int max_moves = MaxMoves(&game.game);
      std::mt19937 rng(players * 7 + colors);
      do {
        std::vector<int32_t> sizes(players);
        std::vector<int8_t> card_colors(Size(players, h)), card_ranks(Size(players, h));
        CHECK(pyhanabi_state_hands(&game.state, sizes.data(), card_colors.data(), card_ranks.data(),
                                   players, h));
        for (int player = 0; player < players; ++player) {
          CHECK_EQ(sizes[player], StateGetHandSize(&game.state, player));
          for (int index = 0; index < h; ++index) {
            const size_t i = Size(player, h) + index;
            pyhanabi_card_t card{-1, -1};
            if (index < sizes[player]) CHECK(StateGetHandCard(&game.state, player, index, &card));
            CHECK(card_colors[i] == card.color && card_ranks[i] == card.rank);
          }
        }

        std::vector<int32_t> uids(max_moves);
        const int32_t count = pyhanabi_state_legal_move_uids(&game.state, uids.data(), max_moves);
        std::vector<int32_t> expected;
        if (StateEndOfGameStatus(&game.state) == 0) {
          void* moves = StateLegalMoves(&game.state);
          for (int index = 0; index < NumMoves(moves); ++index) {
            pyhanabi_move_t move{};
            CHECK(GetMove(moves, index, &move));
            expected.push_back(GetMoveUid(&game.game, &move));
            DeleteMove(&move);
          }
          DeleteMoveList(moves);
        }
        CHECK(count >= 0);
        CHECK(std::vector<int32_t>(uids.begin(), uids.begin() + count) == expected);
      } while (game.Step(&rng));
    }
  }
}

HANABI_TEST(NoLegalMovesWhileACardMustBeDealt) {
  pyhanabi_game_t game{};
  const char* params[] = {"players", "2"};
  CHECK(NewGame(&game, 2, params));
  pyhanabi_state_t state{};
  NewState(&game, &state);
  CHECK_EQ(StateCurPlayer(&state), -1);
  std::vector<int32_t> uids(MaxMoves(&game), 7);
  CHECK_EQ(pyhanabi_state_legal_move_uids(&state, uids.data(), MaxMoves(&game)), 0);
  DeleteState(&state);
  DeleteGame(&game);
}

HANABI_TEST(StateEncodingMatchesObservationEncoding) {
  for (int players = 2; players <= 5; ++players) {
    Game game(players);
    pyhanabi_observation_encoder_t encoder{};
    CHECK(NewObservationEncoder(&encoder, &game.game, 0));
    const int length = EncodingLength(&encoder);
    std::vector<uint8_t> expected(length), bits(length);
    std::mt19937 rng(players);
    do {
      for (int player = 0; player < players; ++player) {
        Obs obs(&game, player);
        CHECK(pyhanabi_observation_encode(&encoder, &obs.observation, expected.data(), length));
        CHECK(pyhanabi_state_encode_observation(&encoder, &game.state, player, bits.data(),
                                                length));
        CHECK(bits == expected);
      }
    } while (game.Step(&rng));
    DeleteObservationEncoder(&encoder);
  }
}

HANABI_TEST(AssignCopiesAStateAcrossGames) {
  Game source(3);
  Game dest(2, 2);  // Destroyed first: its state then belongs to source's game.
  std::mt19937 rng(3);
  for (int step = 0; step < 2; ++step) CHECK(source.Step(&rng));  // Too few to lose 3 lives.
  pyhanabi_state_assign(&dest.state, &source.state);
  char* source_json = StateToJson(&source.state);
  char* dest_json = StateToJson(&dest.state);
  CHECK_EQ(std::string(dest_json), std::string(source_json));
  DeleteString(dest_json);

  CHECK(dest.Step(&rng));  // The copy is independent of its source.
  char* after_json = StateToJson(&source.state);
  CHECK_EQ(std::string(after_json), std::string(source_json));
  DeleteString(after_json);
  DeleteString(source_json);
}

HANABI_TEST(WrongStateBufferSizesAreRejectedWithoutWriting) {
  Game game(3);
  const int p = game.players, h = game.hand_size;
  std::vector<int32_t> sizes(p + 1, 7);
  std::vector<int8_t> colors(Size(p + 1, h + 1), 7), ranks(Size(p + 1, h + 1), 7);
  CHECK(!pyhanabi_state_hands(&game.state, sizes.data(), colors.data(), ranks.data(), p - 1, h));
  CHECK(!pyhanabi_state_hands(&game.state, sizes.data(), colors.data(), ranks.data(), p, h + 1));
  std::vector<int32_t> uids(MaxMoves(&game.game), 7);
  CHECK_EQ(pyhanabi_state_legal_move_uids(&game.state, uids.data(), MaxMoves(&game.game) - 1), -1);
  pyhanabi_observation_encoder_t encoder{};
  CHECK(NewObservationEncoder(&encoder, &game.game, 0));
  const int length = EncodingLength(&encoder);
  std::vector<uint8_t> bits(length + 1, 7);
  CHECK(!pyhanabi_state_encode_observation(&encoder, &game.state, 0, bits.data(), length - 1));
  CHECK(!pyhanabi_state_encode_observation(&encoder, &game.state, 0, bits.data(), length + 1));
  CHECK(!pyhanabi_state_encode_observation(&encoder, &game.state, -1, bits.data(), length));
  CHECK(!pyhanabi_state_encode_observation(&encoder, &game.state, p, bits.data(), length));
  DeleteObservationEncoder(&encoder);
  for (uint8_t v : bits) CHECK_EQ(v, 7);
  for (int32_t v : sizes) CHECK_EQ(v, 7);
  for (int8_t v : colors) CHECK_EQ(v, 7);
  for (int8_t v : ranks) CHECK_EQ(v, 7);
  for (int32_t v : uids) CHECK_EQ(v, 7);
}

int main() { return hanabi_test::RunAll(); }
