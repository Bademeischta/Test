#include <catch2/catch_test_macros.hpp>
#include "position.h"
#include "movegen.h"
#include "search.h"

TEST_CASE("Mate in one", "[search]"){
    movegen::init_attack_tables();
    Position pos("7k/6Q1/7K/8/8/8/8/8 w - - 0 1");
    Search s;
    Limits lim; lim.nodes = 10000; // sufficient
    int score = s.search(pos, lim);
    REQUIRE(score > 30000);
}

