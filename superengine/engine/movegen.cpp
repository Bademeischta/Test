#include "movegen.h"
#include "bitboard.h"

namespace movegen {

MoveList generate_pseudo_legal(const Position& pos) {
    MoveList ml;
    Bitboard pawns = pos.pieces(PAWN, pos.side_to_move());
    while (pawns) {
        int from = bb::pop_lsb(pawns);
        Bitboard push = (pos.side_to_move() == WHITE)
                        ? 1ULL << (from + 8)
                        : 1ULL << (from - 8);
        if (!(push & pos.occupied()))
            ml.push_back({from, from + (pos.side_to_move()==WHITE?8:-8), 0});
    }
    return ml;
}

}
