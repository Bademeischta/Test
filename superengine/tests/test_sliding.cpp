#include <catch2/catch_test_macros.hpp>

#include "movegen.h"
#include "position.h"

using namespace movegen;

TEST_CASE("Bishop moves on empty board", "[sliding]") {
    init_attack_tables();
    Position pos("8/8/8/3B4/8/8/8/8 w - - 0 1");
    MoveList ml;
    generate_bishop_moves(pos, ml);
    REQUIRE(ml.size() == 13);
}

TEST_CASE("Rook moves on empty board", "[sliding]") {
    init_attack_tables();
    Position pos("8/8/8/3R4/8/8/8/8 w - - 0 1");
    MoveList ml;
    generate_rook_moves(pos, ml);
    REQUIRE(ml.size() == 14);
}

TEST_CASE("Queen moves on empty board", "[sliding]") {
    init_attack_tables();
    Position pos("8/8/8/3Q4/8/8/8/8 w - - 0 1");
    MoveList ml;
    generate_queen_moves(pos, ml);
    REQUIRE(ml.size() == 27);
}
