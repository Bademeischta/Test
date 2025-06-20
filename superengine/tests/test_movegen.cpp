#include <catch2/catch_test_macros.hpp>

#include "movegen.h"
#include "position.h"

TEST_CASE("Start position pseudo pawn+knight", "[movegen]") {
    Position pos("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
    auto moves = movegen::generate_pawn_knight(pos);
    REQUIRE(moves.size() == 20);  // 16 pawn pushes + 4 knight moves
}

TEST_CASE("White pawn double push", "[movegen]") {
    Position pos("8/8/8/8/8/8/PPPPPPPP/8 w - - 0 1");
    auto moves = movegen::generate_pawn_knight(pos);
    int pawn_moves = 0;
    for (auto m : moves)
        if (m.from >= 8 && m.from < 16) pawn_moves++;
    REQUIRE(pawn_moves == 16);  // 8 single + 8 double
}

TEST_CASE("Knight on d4", "[movegen]") {
    Position pos("8/8/8/3N4/8/8/8/8 w - - 0 1");
    auto moves = movegen::generate_pawn_knight(pos);
    REQUIRE(moves.size() == 8);
}

TEST_CASE("En passant generation", "[movegen]") {
    Position pos("rnbqkb1r/ppp1pppp/5n2/3pP3/8/8/PPPP1PPP/RNBQKBNR w KQkq d6 0 3");
    auto moves = movegen::generate_pseudo_moves(pos);
    bool found = false;
    for (const auto& m : moves)
        if (m.from == 36 && m.to == 43) found = true;
    REQUIRE(found);
}

TEST_CASE("Start position legal move count", "[movegen]") {
    Position pos("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
    auto moves = movegen::generate_moves(pos);
    REQUIRE(moves.size() == 20);
}

TEST_CASE("Castling move updates board", "[movegen]") {
    Position pos("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1");
    movegen::Move castle{4, 6, 0};
    pos.do_move(castle);
    REQUIRE(pos.piece_on(6) == KING);
    REQUIRE(pos.piece_on(5) == ROOK);
    REQUIRE(pos.piece_on(4) == PIECE_NB);
    REQUIRE(pos.piece_on(7) == PIECE_NB);
    REQUIRE(pos.castling_rights == 12);
    REQUIRE(pos.side_to_move() == BLACK);
}
