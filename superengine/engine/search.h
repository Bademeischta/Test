#pragma once
#include "position.h"
#include "movegen.h"

#include <unordered_map>
#include <chrono>
#include <cstddef>

struct TTEntry { int depth; int score; };

struct Limits {
    int time_ms{0};
    std::size_t nodes{0};
};

class Search {
public:
    int search(Position& pos, int depth);
    int search(Position& pos, const Limits& limits);

private:
    int pv_node(Position& pos, int alpha, int beta, int depth);
    int quiesce(Position& pos, int alpha, int beta);
    bool stop() const;
    void store_tt(const std::string& key, TTEntry entry);

    std::unordered_map<std::string, TTEntry> tt;
    Limits limits_{};
    std::size_t nodes_{0};
    std::chrono::steady_clock::time_point start_{};
    static constexpr std::size_t TT_MAX = 100000;

    movegen::Move killer_[2][64]{};
    int history_[64][64]{};
};
