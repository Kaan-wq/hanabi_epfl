cmake_minimum_required(VERSION 3.24)
project(hanabi_search LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

option(HANABI_BUILD_TESTS "Build engine tests" ON)
option(HANABI_BUILD_BENCHMARKS "Build engine benchmarks" ON)
option(HANABI_NATIVE_ARCH "Optimize for the build machine's CPU (-march=native)" ON)

# Warnings for the engine. Not -Werror: inherited upstream code still has some.
set(HANABI_WARNINGS -Wall -Wextra -Wpedantic -Wshadow)

add_compile_options(
  "$<$<CONFIG:Debug>:-fsanitize=address,undefined;-fno-omit-frame-pointer>"
  "$<$<AND:$<CONFIG:Release>,$<BOOL:${HANABI_NATIVE_ARCH}>>:-march=native>")
add_link_options("$<$<CONFIG:Debug>:-fsanitize=address,undefined>")

include(FetchContent)
find_package(nlohmann_json 3.11 QUIET)
if(NOT nlohmann_json_FOUND)
  FetchContent_Declare(
    nlohmann_json
    URL https://github.com/nlohmann/json/releases/download/v3.11.3/json.tar.xz
    URL_HASH SHA256=d6c65aca6b1ed68e7a182f4757257b107ae403032760ed6ef121c9d55e81757d)
  FetchContent_MakeAvailable(nlohmann_json)
endif()

if(HANABI_BUILD_TESTS)
  enable_testing()
endif()

add_subdirectory(engine)
