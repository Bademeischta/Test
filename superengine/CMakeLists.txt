cmake_minimum_required(VERSION 3.24)
project(SuperEngine VERSION 0.1 LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -flto -ffast-math")

find_package(Threads REQUIRED)

add_library(engine
    engine/bitboard.cpp
    engine/position.cpp
    engine/movegen.cpp
    engine/nnue_eval.cpp
    engine/search.cpp
    engine/uci.cpp)

target_include_directories(engine PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/engine)
target_link_libraries(engine PUBLIC Threads::Threads)

add_executable(superengine main.cpp)
target_link_libraries(superengine PRIVATE engine)

add_executable(perft_test tests/perft.cpp)
target_link_libraries(perft_test PRIVATE engine)
