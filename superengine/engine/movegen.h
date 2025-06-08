#pragma once
#include <cstdint>
#include <vector>

#include "bitboard.h"
#include "position.h"

namespace movegen {

struct Move {
    int from;
    int to;
    int promo;
};  // simple struct
using MoveList = std::vector<Move>;

// encode/decode helpers for tests or uci
inline uint32_t encode_move(int from, int to, int promo = 0) {
    return (from & 0x3F) | ((to & 0x3F) << 6) | ((promo & 0x7) << 12);
}

void init_attack_tables();
extern Bitboard KNIGHT_ATTACKS[64];
extern Bitboard KING_ATTACKS[64];

Bitboard rook_attacks(int from_sq, Bitboard occ);
Bitboard bishop_attacks(int from_sq, Bitboard occ);
inline Bitboard queen_attacks(int from_sq, Bitboard occ) {
    return rook_attacks(from_sq, occ) | bishop_attacks(from_sq, occ);
}

void generate_rook_moves(const Position& pos, MoveList& out);
void generate_bishop_moves(const Position& pos, MoveList& out);
void generate_queen_moves(const Position& pos, MoveList& out);

void generate_pawn_moves(const Position& pos, MoveList& out);
void generate_knight_moves(const Position& pos, MoveList& out);
void generate_sliding_moves(const Position& pos, MoveList& out);
void generate_king_moves(const Position& pos, MoveList& out);

// generate all pseudo legal moves without checking king safety
MoveList generate_pseudo_moves(const Position& pos);

// temporary alias for backwards compatibility
inline MoveList generate_pseudo_legal(const Position& pos) { return generate_pseudo_moves(pos); }

// unified move generator returning only legal moves
MoveList generate_moves(const Position& pos);
inline MoveList generate_legal_moves(const Position& pos) { return generate_moves(pos); }

inline MoveList generate_pawn_knight(const Position& pos) {
    MoveList ml;
    ml.reserve(32);
    generate_pawn_moves(pos, ml);
    generate_knight_moves(pos, ml);
    return ml;
}

}  // namespace movegen
