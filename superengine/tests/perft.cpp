#include "perft.h"
#include <cassert>
#include <iostream>
#include <array>

int main(){
    Position pos; // start position
    std::array<uint64_t,6> results;
    for(int d=1; d<=5; ++d){
        Position p = pos;
        results[d] = perft(p,d);
        std::cout << "depth " << d << ": " << results[d] << std::endl;
    }
    assert(results[1]==20);
    assert(results[2]==400);
    assert(results[3]==8906);
    assert(results[4]==197173);
    assert(results[5]==4868209);
    std::cout << "perft ok" << std::endl;
    return 0;
}
