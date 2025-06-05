#include "nnue_eval.h"
#include "position.h"
#include <fstream>

nnue::Network nnue::net;

static int material(const Position& pos) {
    int score = 0;
    auto add = [&](Piece pc, int val){
        Bitboard w = pos.pieces(pc, WHITE);
        Bitboard b = pos.pieces(pc, BLACK);
        while (w) { bb::pop_lsb(w); score += val; }
        while (b) { bb::pop_lsb(b); score -= val; }
    };
    add(PAWN,100); add(KNIGHT,300); add(BISHOP,300); add(ROOK,500); add(QUEEN,900);
    return score;
}

int nnue::eval(const Position& pos) {
    return material(pos);
}

void nnue::load_network(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if(f)
        f.read(reinterpret_cast<char*>(&net), sizeof(Network));
}
