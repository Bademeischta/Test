#pragma once
#include <string>

#include "bitboard.h"
namespace movegen {
struct Move;
}

enum Color { WHITE = 0, BLACK = 1 };
enum Piece { PAWN = 0, KNIGHT, BISHOP, ROOK, QUEEN, KING, PIECE_NB };

struct Position {
    Bitboard piece_bb[2][PIECE_NB]{};  // [color][piece]
    Bitboard occupied_bb[2]{};
    Bitboard all_occupied{};
    Color side{};
    uint8_t castling_rights{};  // bit0=K, bit1=Q, bit2=k, bit3=q
    int8_t en_passant{-1};      // -1 if none
    int halfmove_clock{};
    int fullmove_number{1};

    Position() = default;
    explicit Position(const std::string& fen);

    Bitboard pieces(Piece pc, Color c) const { return piece_bb[c][pc]; }
    Bitboard occupied() const { return all_occupied; }
    Color side_to_move() const { return side; }
    Piece piece_on(int sq) const;
    bool in_check(Color c) const;
    void do_move(const movegen::Move& m);
    std::string to_fen() const;
};
