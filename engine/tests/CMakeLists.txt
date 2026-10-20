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

# Drives the C API with bad input; see the file comment.
add_executable(test_capi_robustness test_capi_robustness.cc)
target_link_libraries(test_capi_robustness PRIVATE pyhanabi nlohmann_json::nlohmann_json)
target_compile_options(test_capi_robustness PRIVATE ${HANABI_WARNINGS})
add_test(NAME capi_robustness COMMAND test_capi_robustness)

add_executable(test_hand_sampler test_hand_sampler.cc)
target_link_libraries(test_hand_sampler PRIVATE hanabi)
target_compile_options(test_hand_sampler PRIVATE ${HANABI_WARNINGS})
add_test(NAME hand_sampler COMMAND test_hand_sampler)

# Observation and state buffer functions of the C API; see the file comment.
add_executable(test_capi_buffers test_capi_buffers.cc)
target_link_libraries(test_capi_buffers PRIVATE pyhanabi nlohmann_json::nlohmann_json)
target_compile_options(test_capi_buffers PRIVATE ${HANABI_WARNINGS})
add_test(NAME capi_buffers COMMAND test_capi_buffers)

# Optimized engine paths against their reference definitions; see the file comment.
add_executable(test_fast_paths test_fast_paths.cc)
target_link_libraries(test_fast_paths PRIVATE hanabi)
target_compile_options(test_fast_paths PRIVATE ${HANABI_WARNINGS})
add_test(NAME fast_paths COMMAND test_fast_paths)
