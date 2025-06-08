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

TEST_CASE("Fen with en passant target", "[position]") {
    Position pos("rnbqkbnr/pppppppp/8/8/8/4P3/PPPP1PPP/RNBQKBNR b KQkq e3 0 1");
    REQUIRE(pos.en_passant == 20);  // e3 square index
    REQUIRE(pos.side_to_move() == BLACK);
    REQUIRE(pos.castling_rights == 0b1111);
    REQUIRE(pos.piece_on(20) == PAWN);  // white pawn on e3
}
