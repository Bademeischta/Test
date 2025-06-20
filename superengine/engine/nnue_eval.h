#pragma once
#include <array>
#include <string>

#include "position.h"

namespace nnue {

constexpr int INPUTS = 768;
constexpr int HIDDEN1 = 512;
constexpr int HIDDEN2 = 256;
constexpr int OUTPUTS = 1;

struct Network {
    alignas(64) std::array<int16_t, INPUTS * HIDDEN1> w1;
    alignas(64) std::array<int16_t, HIDDEN1 * HIDDEN2> w2;
    alignas(64) std::array<int16_t, HIDDEN2> bias2;
    alignas(64) std::array<int16_t, HIDDEN2> w3;
};

extern Network net;

int eval(const Position& pos);
void load_network(const std::string& path);

}  // namespace nnue
