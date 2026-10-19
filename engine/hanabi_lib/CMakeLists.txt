add_library(hanabi STATIC
  canonical_encoders.cc
  hanabi_card.cc
  hanabi_game.cc
  hanabi_hand.cc
  hanabi_history_item.cc
  hanabi_move.cc
  hanabi_observation.cc
  hanabi_state.cc
  json_checks.cc
  util.cc)

# Headers are included both as "hanabi_state.h" (inside the library) and as
# "hanabi_lib/hanabi_state.h" (from the C API).
target_include_directories(hanabi PUBLIC ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/..)
target_link_libraries(hanabi PUBLIC nlohmann_json::nlohmann_json)
target_compile_options(hanabi PRIVATE ${HANABI_WARNINGS})
