add_executable(bench_engine bench_engine.cc)
target_link_libraries(bench_engine PRIVATE search)  # Brings in hanabi.
target_compile_options(bench_engine PRIVATE ${HANABI_WARNINGS})
