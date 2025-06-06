#pragma once
#include "position.h"
#include "movegen.h"

#include <unordered_map>

struct TTEntry { int depth; int score; };

class Search {
public:
    int search(Position& pos, int depth);
private:
    int pv_node(Position& pos, int alpha, int beta, int depth);
    int quiesce(Position& pos, int alpha, int beta);
    std::unordered_map<std::string, TTEntry> tt;
};
