#include "nnue_eval.h"
#include "position.h"
#include <fstream>

nnue::Network nnue::net;

static int16_t clamp_relu(int32_t x) { return x > 0 ? (x < 32767 ? x : 32767) : 0; }

int nnue::eval(const Position& pos) {
    static const int piece_values[] = {100, 320, 330, 500, 900, 0};
    int score = 0;
    for(int c = 0; c < 2; ++c){
        for(int p = 0; p < PIECE_NB; ++p){
            Bitboard bb = pos.piece_bb[c][p];
            int pc = __builtin_popcountll(bb) * piece_values[p];
            score += (c == WHITE ? pc : -pc);
        }
    }
    return score;
}

void nnue::load_network(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    f.read(reinterpret_cast<char*>(&net), sizeof(Network));
}
