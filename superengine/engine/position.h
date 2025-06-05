#pragma once
#include "bitboard.h"
#include <array>
#include <string>
#include <vector>

enum Color { WHITE, BLACK };
enum Piece { PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING, PIECE_NB };

namespace movegen { struct Move; }

struct Position {
    std::array<std::array<Bitboard, PIECE_NB>, 2> piece_bb{};
    Color stm = WHITE;

    static Position from_fen(const std::string& fen);
    Bitboard pieces(Piece pc, Color c) const { return piece_bb[c][pc]; }
    Color side_to_move() const { return stm; }
    Bitboard occupied() const {
        return piece_bb[WHITE][PAWN] | piece_bb[WHITE][KNIGHT] | piece_bb[WHITE][BISHOP] |
               piece_bb[WHITE][ROOK] | piece_bb[WHITE][QUEEN] | piece_bb[WHITE][KING] |
               piece_bb[BLACK][PAWN] | piece_bb[BLACK][KNIGHT] | piece_bb[BLACK][BISHOP] |
               piece_bb[BLACK][ROOK] | piece_bb[BLACK][QUEEN] | piece_bb[BLACK][KING];
    }
    void do_move(const movegen::Move& m);
};

extern const Position START_POS;
