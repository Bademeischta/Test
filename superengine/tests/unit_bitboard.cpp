#include "../engine/bitboard.h"
#include <cassert>

int main() {
    Bitboard b = 0b1010;
    int idx = bb::pop_lsb(b);
    assert(idx == 1);
    assert(b == 0b1000);
    return 0;
}
