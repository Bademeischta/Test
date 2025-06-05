#pragma once
#include <cstdint>
using Bitboard = uint64_t;

namespace bb {
    constexpr Bitboard FILE_A = 0x0101010101010101ULL;
    constexpr Bitboard RANK_1 = 0xFFULL;
    inline int pop_lsb(Bitboard &b) {
        int idx = __builtin_ctzll(b);
        b &= b - 1;
        return idx;
    }
}
