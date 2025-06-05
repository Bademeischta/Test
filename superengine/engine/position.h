#pragma once
#include "bitboard.h"

enum Color { WHITE, BLACK };
enum Piece { PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING };

struct Position {
    Bitboard pieces(Piece pc, Color c) const { return 0; }
    Color side_to_move() const { return WHITE; }
    Bitboard occupied() const { return 0; }
    void do_move(const movegen::Move& m) {}
};
