# Search on top of the engine: rule-based rollout policies and the tree search.
add_library(search STATIC mcts.cc van_den_bergh.cc)
target_link_libraries(search PUBLIC hanabi)
target_compile_options(search PRIVATE ${HANABI_WARNINGS})
