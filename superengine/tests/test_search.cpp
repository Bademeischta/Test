#include <catch2/catch_test_macros.hpp>

#include "movegen.h"
#include "position.h"
#include "search.h"

TEST_CASE("Mate in one", "[search]") {
    movegen::init_attack_tables();
    Position pos("7k/6Q1/7K/8/8/8/8/8 w - - 0 1");
    Search s;
    Limits lim;
    lim.nodes = 10000;  // sufficient
    int score = s.search(pos, lim);
    REQUIRE(score > 30000);
}

TEST_CASE("Multithreaded search matches single", "[search]") {
    movegen::init_attack_tables();
    Position pos("7k/6Q1/7K/8/8/8/8/8 w - - 0 1");
    Search s;
    int single = s.search(pos, 3);
    int multi = s.search(pos, 3, 4);
    REQUIRE(single == multi);
}
