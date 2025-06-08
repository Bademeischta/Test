#include "search.h"

#include <algorithm>

#include "nnue_eval.h"

static const int MVV[6] = {100, 320, 330, 500, 900, 20000};

constexpr int INF = 32000;

int Search::search(Position& pos, int depth) { return pv_node(pos, -INF, INF, depth); }

int Search::search(Position& pos, const Limits& limits) {
    limits_ = limits;
    nodes_ = 0;
    start_ = std::chrono::steady_clock::now();
    int best = 0;
    for (int d = 1;; ++d) {
        int score = pv_node(pos, -INF, INF, d);
        if (!stop()) best = score;
        if (stop()) break;
        if (limits.nodes == 0 && limits.time_ms == 0) break;  // fallback
    }
    return best;
}

int Search::pv_node(Position& pos, int alpha, int beta, int depth) {
    ++nodes_;
    if (stop()) return nnue::eval(pos);

    auto key = pos.to_fen();
    auto it = tt.find(key);
    if (it != tt.end() && it->second.depth >= depth) return it->second.score;

    auto moves = movegen::generate_legal_moves(pos);
    if (moves.empty()) return pos.in_check(pos.side) ? -INF + depth : 0;

    if (depth <= 0) return quiesce(pos, alpha, beta);

    if (depth >= 3 && !pos.in_check(pos.side)) {
        Position null = pos;
        null.side = pos.side == WHITE ? BLACK : WHITE;
        null.en_passant = -1;
        int score = -pv_node(null, -beta, -beta + 1, depth - 3);
        if (score >= beta) {
            store_tt(key, {depth, score});
            return score;
        }
    }

    int eval = nnue::eval(pos);
    if (eval >= beta) return eval;

    moves = movegen::generate_moves(pos);
    std::vector<std::pair<int, movegen::Move>> scored;
    scored.reserve(moves.size());
    for (const auto& m : moves) {
        int score = 0;
        Piece capture = pos.piece_on(m.to);
        Piece piece = pos.piece_on(m.from);
        if (capture != PIECE_NB) score += 10 * MVV[capture] - MVV[piece];
        if (killer_[0][depth].from == m.from && killer_[0][depth].to == m.to) score += 9000;
        if (killer_[1][depth].from == m.from && killer_[1][depth].to == m.to) score += 8000;
        score += history_[m.from][m.to];
        scored.push_back({score, m});
    }
    std::sort(scored.begin(), scored.end(), [](auto& a, auto& b) { return a.first > b.first; });
    for (size_t i = 0; i < scored.size(); ++i) {
        auto m = scored[i].second;
        Position next = pos;
        next.do_move(m);
        int score;
        if (i >= 3 && depth >= 3) {
            score = -pv_node(next, -alpha - 1, -alpha, depth - 2);
            if (score > alpha) score = -pv_node(next, -beta, -alpha, depth - 1);
        } else {
            score = -pv_node(next, -beta, -alpha, depth - 1);
        }
        if (score >= beta) {
            killer_[1][depth] = killer_[0][depth];
            killer_[0][depth] = m;
            history_[m.from][m.to] += depth * depth;
            store_tt(key, {depth, score});
            return score;
        }
        if (score > alpha) {
            alpha = score;
            history_[m.from][m.to] += depth;
        }
    }
    store_tt(key, {depth, alpha});
    return alpha;
}

int Search::quiesce(Position& pos, int alpha, int beta) {
    ++nodes_;
    if (stop()) return nnue::eval(pos);

    int stand_pat = nnue::eval(pos);
    if (stand_pat >= beta) return beta;
    if (stand_pat > alpha) alpha = stand_pat;

    auto moves = movegen::generate_moves(pos);
    for (const auto& m : moves) {
        Piece capture = pos.piece_on(m.to);
        if (capture == PIECE_NB) continue;  // only captures
        Position next = pos;
        next.do_move(m);
        if (next.in_check(pos.side)) continue;
        int score = -quiesce(next, -beta, -alpha);
        if (score >= beta) return score;
        if (score > alpha) alpha = score;
    }
    return alpha;
}

bool Search::stop() const {
    if (limits_.nodes && nodes_ >= limits_.nodes) return true;
    if (limits_.time_ms) {
        auto elapsed =
            std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - start_).count();
        if (elapsed >= limits_.time_ms) return true;
    }
    return false;
}

void Search::store_tt(const std::string& key, TTEntry entry) {
    auto it = tt.find(key);
    if (it == tt.end()) {
        if (tt.size() >= TT_MAX) tt.erase(tt.begin());
        tt.emplace(key, entry);
    } else if (entry.depth >= it->second.depth) {
        it->second = entry;
    }
}
