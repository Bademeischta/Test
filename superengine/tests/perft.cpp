#include "../engine/position.h"
#include "../engine/movegen.h"
#include <cassert>
#include <iostream>

static uint64_t perft(const Position& pos, int depth) {
    if (depth == 0) return 1;
    uint64_t nodes = 0;
    auto moves = movegen::generate_pseudo_legal(const_cast<Position&>(pos));
    for (auto& m : moves) {
        Position next = pos;
        next.do_move(m);
        nodes += perft(next, depth-1);
    }
    return nodes;
}

int main() {
    uint64_t n = perft(START_POS, 1);
    assert(n > 0); // sanity
    std::cout << "perft(1)=" << n << "\n";
    return 0;
}
