#include <catch2/catch_test_macros.hpp>
#include "position.h"
#include "movegen.h"

using namespace movegen;

static uint64_t perft(Position& pos, int depth){
    if(depth==0) return 1;
    auto moves = generate_pseudo_legal(pos);
    uint64_t nodes = 0;
    for(const auto& m : moves){
        Position next = pos;
        next.do_move(m);
        nodes += perft(next, depth-1);
    }
    return nodes;
}

TEST_CASE("Perft startpos", "[perft]"){
    init_attack_tables();
    Position pos("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
    REQUIRE(perft(pos,1) == 20);
    REQUIRE(perft(pos,2) == 400);
    REQUIRE(perft(pos,3) == 8902);
}

