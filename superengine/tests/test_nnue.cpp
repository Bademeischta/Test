#include <catch2/catch_test_macros.hpp>

#include "nnue_eval.h"
#include "position.h"

TEST_CASE("NNUE eval symmetry", "[nnue]") {
    Position pos("8/8/8/8/8/8/8/8 w - - 0 1");
    REQUIRE(nnue::eval(pos) == 0);
}
