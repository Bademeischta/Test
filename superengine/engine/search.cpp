#include "search.h"
#include "nnue_eval.h"

constexpr int INF = 32000;

int Search::search(Position& pos, int depth){
    return pv_node(pos, -INF, INF, depth);
}

int Search::pv_node(Position& pos, int alpha, int beta, int depth) {
    auto key = pos.to_fen();
    auto it = tt.find(key);
    if(it != tt.end() && it->second.depth >= depth)
        return it->second.score;

    if (depth <= 0)
        return quiesce(pos, alpha, beta);

    int eval = nnue::eval(pos);
    if (eval >= beta) return eval;

    auto moves = movegen::generate_pseudo_legal(pos);
    for (size_t i = 0; i < moves.size(); ++i) {
        Position next = pos;
        next.do_move(moves[i]);
        int score = -pv_node(next, -beta, -alpha, depth - 1);
        if (score >= beta) {
            tt[key] = {depth, score};
            return score;
        }
        if (score > alpha) alpha = score;
    }
    tt[key] = {depth, alpha};
    return alpha;
}

int Search::quiesce(Position& pos, int alpha, int beta){
    int stand_pat = nnue::eval(pos);
    if(stand_pat >= beta) return beta;
    if(stand_pat > alpha) alpha = stand_pat;

    auto moves = movegen::generate_pseudo_legal(pos);
    for(const auto& m: moves){
        Piece capture = pos.piece_on(m.to);
        if(capture == PIECE_NB) continue; // only captures
        Position next = pos;
        next.do_move(m);
        int score = -quiesce(next, -beta, -alpha);
        if(score >= beta) return score;
        if(score > alpha) alpha = score;
    }
    return alpha;
}
