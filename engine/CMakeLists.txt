add_subdirectory(hanabi_lib)

# C API shared library, loaded from Python via cffi.
add_library(pyhanabi SHARED capi/pyhanabi.cc capi/pyhanabi_observation.cc)
target_include_directories(pyhanabi PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/capi)
target_link_libraries(pyhanabi PRIVATE hanabi)
target_compile_options(pyhanabi PRIVATE ${HANABI_WARNINGS})

if(HANABI_BUILD_TESTS)
  add_subdirectory(tests)
endif()

if(HANABI_BUILD_BENCHMARKS)
  add_subdirectory(benchmarks)
endif()
