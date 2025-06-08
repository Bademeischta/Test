#include "bitboard.h"

namespace bb {

Bitboard north_one(Bitboard b) { return b << 8; }
Bitboard south_one(Bitboard b) { return b >> 8; }
Bitboard east_one(Bitboard b) { return (b & ~FILE_H) << 1; }
Bitboard west_one(Bitboard b) { return (b & ~FILE_A) >> 1; }

Bitboard north_east(Bitboard b) { return (b & ~FILE_H) << 9; }
Bitboard north_west(Bitboard b) { return (b & ~FILE_A) << 7; }
Bitboard south_east(Bitboard b) { return (b & ~FILE_H) >> 7; }
Bitboard south_west(Bitboard b) { return (b & ~FILE_A) >> 9; }

}  // namespace bb
