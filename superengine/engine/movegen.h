#pragma once
#include <vector>
#include "bitboard.h"
#include "position.h" // placeholder for Position definition

namespace movegen {
struct Move { int from; int to; int type; };
using MoveList = std::vector<Move>;

MoveList generate_pseudo_legal(const Position& pos);
}
