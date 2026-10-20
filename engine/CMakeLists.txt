add_subdirectory(hanabi_lib)

# C API shared library, loaded from Python via cffi.
add_library(pyhanabi SHARED capi/pyhanabi.cc capi/pyhanabi_observation.cc)
target_include_directories(pyhanabi PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/capi)
target_link_libraries(pyhanabi PRIVATE hanabi)
target_compile_options(pyhanabi PRIVATE ${HANABI_WARNINGS})
# On Linux, bundle the C++ runtime: Python can then load an engine built with
# a newer compiler than the system's (otherwise: "GLIBCXX_... not found").
# --exclude-libs hides the bundled runtime's symbols; exported, its global
# locale facet ids clash with the libstdc++ that torch loads, and torch
# segfaults on import.
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
  target_link_options(pyhanabi PRIVATE -static-libstdc++ -static-libgcc -Wl,--exclude-libs,ALL)
endif()

if(HANABI_BUILD_TESTS)
  add_subdirectory(tests)
endif()

if(HANABI_BUILD_BENCHMARKS)
  add_subdirectory(benchmarks)
endif()
