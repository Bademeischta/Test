#include <catch2/catch_test_macros.hpp>
#include "position.h"

TEST_CASE("Fen parsing extras", "[position]") {
    Position pos("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
    REQUIRE(pos.side_to_move() == WHITE);
    REQUIRE(pos.castling_rights == 0b1111);
    REQUIRE(pos.en_passant == -1);
    REQUIRE(pos.halfmove_clock == 0);
    REQUIRE(pos.fullmove_number == 1);
    // check piece on e1 is king
    REQUIRE(pos.piece_on(4) == KING);
}
