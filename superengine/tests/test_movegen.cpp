#include <catch2/catch_test_macros.hpp>
#include "position.h"
#include "movegen.h"

TEST_CASE("Start position pseudo pawn+knight", "[movegen]") {
    Position pos("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
    auto moves = movegen::generate_pawn_knight(pos);
    REQUIRE(moves.size() == 20); // 16 pawn pushes + 4 knight moves
}

TEST_CASE("White pawn double push", "[movegen]") {
    Position pos("8/8/8/8/8/8/PPPPPPPP/8 w - - 0 1");
    auto moves = movegen::generate_pawn_knight(pos);
    int pawn_moves = 0;
    for(auto m: moves) if(m.from>=8 && m.from<16) pawn_moves++;
    REQUIRE(pawn_moves == 16); // 8 single + 8 double
}

TEST_CASE("Knight on d4", "[movegen]") {
    Position pos("8/8/8/3N4/8/8/8/8 w - - 0 1");
    auto moves = movegen::generate_pawn_knight(pos);
    REQUIRE(moves.size() == 8);
}
