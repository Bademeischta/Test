#pragma once
#include <cstdint>
using Bitboard = uint64_t;

namespace bb {
    constexpr Bitboard FILE_A = 0x0101010101010101ULL;
    constexpr Bitboard FILE_H = 0x8080808080808080ULL;
    constexpr Bitboard RANK_1 = 0xFFULL;
    constexpr Bitboard RANK_2 = 0xFFULL << 8;
    constexpr Bitboard RANK_7 = 0xFFULL << 48;
    constexpr Bitboard RANK_8 = 0xFFULL << 56;

    inline int pop_lsb(Bitboard &b) {
        int idx = __builtin_ctzll(b);
        b &= b - 1;
        return idx;
    }

    Bitboard north_one(Bitboard b);
    Bitboard south_one(Bitboard b);
    Bitboard east_one(Bitboard b);
    Bitboard west_one(Bitboard b);

    Bitboard north_east(Bitboard b);
    Bitboard north_west(Bitboard b);
    Bitboard south_east(Bitboard b);
    Bitboard south_west(Bitboard b);
}
