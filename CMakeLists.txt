cmake_minimum_required (VERSION 3.30.5)
project (hanabi_epfl)

set(CMAKE_C_FLAGS "-O2 -std=c++11 -march=native -fPIC")
set(CMAKE_CXX_FLAGS "-O2 -std=c++11 -march=native -fPIC")

find_package(nlohmann_json 3.2.0 REQUIRED)
add_subdirectory (hanabi_lib)

add_library (pyhanabi SHARED pyhanabi.cc)
target_link_libraries (pyhanabi LINK_PUBLIC hanabi)

install(TARGETS pyhanabi LIBRARY DESTINATION hanabi_epfl)
install(FILES __init__.py DESTINATION hanabi_epfl)
install(FILES rl_env.py DESTINATION hanabi_epfl)
install(FILES pyhanabi.py DESTINATION hanabi_epfl)
install(FILES pyhanabi.h DESTINATION hanabi_epfl)