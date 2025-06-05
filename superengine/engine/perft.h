#pragma once
#include "movegen.h"

inline uint64_t perft(Position& pos, int depth){
    if(depth==0) return 1;
    auto moves = movegen::generate_pseudo_legal(pos);
    uint64_t nodes = 0;
    for(const auto& m : moves){
        Position next = pos;
        next.do_move(m);
        nodes += perft(next, depth-1);
    }
    return nodes;
}
