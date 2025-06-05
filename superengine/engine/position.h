#pragma once
#include "bitboard.h"

namespace movegen { struct Move; }

enum Color { WHITE, BLACK };
enum Piece { PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING };

struct Position {
    Bitboard board[2][6]{}; // [color][piece]
    Color stm = WHITE;
    int ep = -1; // square index or -1

    Position();

    Bitboard pieces(Piece pc, Color c) const { return board[c][pc]; }
    Color side_to_move() const { return stm; }
    Bitboard occupied() const {
        return occupied(WHITE) | occupied(BLACK);
    }
    Bitboard occupied(Color c) const {
        Bitboard b=0;
        for(int p=0;p<6;++p) b |= board[c][p];
        return b;
    }
    int ep_square() const { return ep; }

    bool in_check(Color c) const;
    void do_move(const movegen::Move& m);
};
