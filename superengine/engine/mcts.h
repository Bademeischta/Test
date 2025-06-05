#pragma once
#include <atomic>
#include <array>
#include <cmath>
#include "bitboard.h"

struct MCTSNode {
    std::atomic<int> N{0};
    std::atomic<float> W{0};
    Bitboard zobrist{0};
    std::array<MCTSNode*, 64> child{};
};

inline float uct(const MCTSNode* n, int parent_N, float prior_p, float c=1.4f) {
    return (n->N ? n->W / n->N : 0.0f) + c * prior_p * std::sqrt(parent_N) / (1 + n->N);
}
