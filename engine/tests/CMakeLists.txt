add_executable(test_fork_extensions test_fork_extensions.cc)
target_link_libraries(test_fork_extensions PRIVATE hanabi)
target_compile_options(test_fork_extensions PRIVATE ${HANABI_WARNINGS})
add_test(NAME fork_extensions COMMAND test_fork_extensions)

# Replays games recorded from unmodified upstream HLE (see parity/README.md).
add_executable(parity_test parity/parity_test.cc)
target_include_directories(parity_test PRIVATE parity)
target_link_libraries(parity_test PRIVATE hanabi)
target_compile_options(parity_test PRIVATE ${HANABI_WARNINGS})
add_test(NAME parity_vs_upstream
         COMMAND parity_test ${CMAKE_CURRENT_SOURCE_DIR}/parity/golden_upstream_5df6a73.json)
