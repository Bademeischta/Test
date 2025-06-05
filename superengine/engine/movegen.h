#pragma once
#include <vector>
#include "bitboard.h"
#include "position.h" // placeholder for Position definition

namespace movegen {
enum MoveFlag {
    QUIET = 0,
    CAPTURE = 1 << 0,
    DOUBLE_PUSH = 1 << 1,
    EN_PASSANT = 1 << 2,
    PROMOTION = 1 << 3
};

struct Move {
    int from;
    int to;
    int flags;
    Piece promotion; // used when PROMOTION flag set
};
using MoveList = std::vector<Move>;

MoveList generate_pseudo_legal(const Position& pos);
}
