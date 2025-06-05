#include "search.h"
#include "nnue_eval.h"

constexpr int INF = 32000;

int Search::pv_node(Position& pos, int alpha, int beta, int depth) {
    if (depth == 0)
        return 0; // quiescence placeholder

    int eval = nnue::eval(pos);
    if (eval >= beta) return eval;

    auto moves = movegen::generate_pseudo_legal(pos);
    for (size_t i = 0; i < moves.size(); ++i) {
        Position next = pos;
        next.do_move(moves[i]);
        int score = -pv_node(next, -beta, -alpha, depth - 1);
        if (score >= beta) return score;
        if (score > alpha) alpha = score;
    }
    return alpha;
}
