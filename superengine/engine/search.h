#pragma once
#include "position.h"
#include "movegen.h"

class Search {
public:
    int pv_node(Position& pos, int alpha, int beta, int depth);
};
