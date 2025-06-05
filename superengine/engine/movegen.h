#pragma once
#include <vector>
#include <cstdint>
#include "bitboard.h"
#include "position.h"

namespace movegen {

struct Move { int from; int to; int promo; }; // simple struct
using MoveList = std::vector<Move>;

// encode/decode helpers for tests or uci
inline uint32_t encode_move(int from, int to, int promo=0) {
    return (from & 0x3F) | ((to & 0x3F) << 6) | ((promo & 0x7) << 12);
}

void init_attack_tables();

void generate_pawn_moves(const Position& pos, MoveList& out);
void generate_knight_moves(const Position& pos, MoveList& out);

inline MoveList generate_pawn_knight(const Position& pos) {
    MoveList ml; ml.reserve(32);
    generate_pawn_moves(pos, ml);
    generate_knight_moves(pos, ml);
    return ml;
}

} // namespace movegen
