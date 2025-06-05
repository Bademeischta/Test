#include "nnue_eval.h"
#include "position.h"
#include <fstream>

nnue::Network nnue::net;

static int16_t clamp_relu(int32_t x) { return x > 0 ? (x < 32767 ? x : 32767) : 0; }

int nnue::eval(const Position& pos) {
    alignas(32) int16_t feat[INPUTS]{}; // unused stub
    alignas(32) int16_t hidden1[HIDDEN1];
    for (int i=0;i<HIDDEN1;i+=16) {
        // placeholder loop
    }
    return 0;
}

void nnue::load_network(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    f.read(reinterpret_cast<char*>(&net), sizeof(Network));
}
