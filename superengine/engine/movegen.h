#pragma once
#include <vector>
#include "bitboard.h"
#include "position.h"

namespace movegen {

enum MoveType { QUIET, CAPTURE };
struct Move { int from; int to; MoveType type; };
using MoveList = std::vector<Move>;

MoveList generate_pseudo_legal(const Position& pos);

}
