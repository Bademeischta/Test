cmake_minimum_required(VERSION 3.24)
project(SuperEngine VERSION 0.1 LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -flto")

include(FetchContent)
FetchContent_Declare(
  Catch2
  GIT_REPOSITORY https://github.com/catchorg/Catch2.git
  GIT_TAG v3.4.0
)
FetchContent_MakeAvailable(Catch2)

find_package(Threads REQUIRED)

add_library(engine
    engine/bitboard.cpp
    engine/movegen.cpp
    engine/position.cpp)

target_include_directories(engine PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/engine)
target_link_libraries(engine PUBLIC Threads::Threads)

add_executable(test_movegen tests/test_movegen.cpp)
target_link_libraries(test_movegen PRIVATE engine Catch2::Catch2WithMain)
add_test(NAME test_movegen COMMAND test_movegen)

add_executable(test_position tests/test_position.cpp)
target_link_libraries(test_position PRIVATE engine Catch2::Catch2WithMain)
add_test(NAME test_position COMMAND test_position)

enable_testing()
