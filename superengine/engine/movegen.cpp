#include "movegen.h"
#include "bitboard.h"

namespace movegen {

static const int knight_offsets[8] = {17, 15, 10, 6, -17, -15, -10, -6};
static const int king_offsets[8] = {1, -1, 8, -8, 9, 7, -9, -7};

MoveList generate_pseudo_legal(const Position& pos) {
    MoveList ml;
    Color us = pos.side_to_move();
    Bitboard occ = pos.occupied();
    Bitboard pawns = pos.pieces(PAWN, us);
    int dir = (us==WHITE)?8:-8;
    while (pawns) {
        int from = bb::pop_lsb(pawns);
        int to = from + dir;
        Bitboard push = 1ULL << to;
        if (!(push & occ))
            ml.push_back({from,to,QUIET});
    }
    Bitboard knights = pos.pieces(KNIGHT, us);
    while (knights) {
        int from = bb::pop_lsb(knights);
        for (int o: knight_offsets) {
            int to = from + o;
            if (to <0 || to >=64) continue;
            ml.push_back({from,to,QUIET});
        }
    }
    Bitboard king = pos.pieces(KING, us);
    while (king) {
        int from = bb::pop_lsb(king);
        for (int o: king_offsets) {
            int to = from + o;
            if (to<0 || to>=64) continue;
            ml.push_back({from,to,QUIET});
        }
    }
    return ml;
}

}
