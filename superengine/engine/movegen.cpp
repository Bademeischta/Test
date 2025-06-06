#include "movegen.h"
#include <cstring>

namespace movegen {

uint64_t KNIGHT_ATTACKS[64];

static void init_knights() {
    for(int sq=0; sq<64; ++sq) {
        int r = sq/8, f=sq%8;
        uint64_t mask=0ULL;
        const int dr[8]={2,2,-2,-2,1,1,-1,-1};
        const int df[8]={1,-1,1,-1,2,-2,2,-2};
        for(int i=0;i<8;++i){
            int nr=r+dr[i], nf=f+df[i];
            if(nr>=0 && nr<8 && nf>=0 && nf<8)
                mask|=1ULL<<(nr*8+nf);
        }
        KNIGHT_ATTACKS[sq]=mask;
    }
}

void init_attack_tables(){
    static bool init=false; if(init) return; init=true; init_knights();
}

void generate_pawn_moves(const Position& pos, MoveList& out){
    Color stm = pos.side;
    Bitboard wp = pos.piece_bb[WHITE][PAWN];
    Bitboard bp = pos.piece_bb[BLACK][PAWN];
    Bitboard occ = pos.all_occupied;
    Bitboard occW = pos.occupied_bb[WHITE];
    Bitboard occB = pos.occupied_bb[BLACK];

    if(stm == WHITE){
    // white single push
    Bitboard single = (wp<<8) & ~occ;
    Bitboard promo_rank = bb::RANK_8;
    Bitboard promo_single = single & promo_rank;
    Bitboard normal_single = single & ~promo_rank;
    while(normal_single){
        int to = bb::pop_lsb(normal_single);
        int from = to-8;
        out.push_back({from,to,0});
    }
    while(promo_single){
        int to = bb::pop_lsb(promo_single);
        int from = to-8;
        for(int p=1;p<=4;++p) out.push_back({from,to,p});
    }
    // white double
    Bitboard rank2 = bb::RANK_2;
    Bitboard single_from_rank2 = ((wp & rank2)<<8) & ~occ;
    Bitboard dbl = (single_from_rank2<<8) & ~occ;
    while(dbl){
        int to = bb::pop_lsb(dbl);
        int from = to-16;
        out.push_back({from,to,0});
    }
    // captures left/right
    Bitboard left = ((wp & ~bb::FILE_A)<<7) & occB;
    Bitboard right = ((wp & ~bb::FILE_H)<<9) & occB;
    Bitboard promo_left = left & promo_rank;
    Bitboard normal_left = left & ~promo_rank;
    while(normal_left){
        int to = bb::pop_lsb(normal_left);
        int from = to-7;
        out.push_back({from,to,0});
    }
    while(promo_left){
        int to = bb::pop_lsb(promo_left);
        int from = to-7;
        for(int p=1;p<=4;++p) out.push_back({from,to,p});
    }
    Bitboard promo_right = right & promo_rank;
    Bitboard normal_right = right & ~promo_rank;
    while(normal_right){
        int to = bb::pop_lsb(normal_right);
        int from = to-9;
        out.push_back({from,to,0});
    }
    while(promo_right){
        int to = bb::pop_lsb(promo_right);
        int from = to-9;
        for(int p=1;p<=4;++p) out.push_back({from,to,p});
    }
    }

    if(stm == BLACK){
    // black moves
    Bitboard singleB = (bp>>8) & ~occ;
    Bitboard promo_rankB = bb::RANK_1;
    Bitboard promo_singleB = singleB & promo_rankB;
    Bitboard normal_singleB = singleB & ~promo_rankB;
    while(normal_singleB){
        int to=bb::pop_lsb(normal_singleB);
        int from=to+8;
        out.push_back({from,to,0});
    }
    while(promo_singleB){
        int to=bb::pop_lsb(promo_singleB);
        int from=to+8;
        for(int p=1;p<=4;++p) out.push_back({from,to,p});
    }
    Bitboard rank7 = bb::RANK_7;
    Bitboard single_from_rank7 = ((bp & rank7)>>8) & ~occ;
    Bitboard dblB = (single_from_rank7>>8) & ~occ;
    while(dblB){
        int to=bb::pop_lsb(dblB);
        int from=to+16;
        out.push_back({from,to,0});
    }
    Bitboard leftB = ((bp & ~bb::FILE_H)>>9) & occW;
    Bitboard rightB = ((bp & ~bb::FILE_A)>>7) & occW;
    Bitboard promo_leftB = leftB & promo_rankB;
    Bitboard normal_leftB = leftB & ~promo_rankB;
    while(normal_leftB){
        int to=bb::pop_lsb(normal_leftB);
        int from=to+9;
        out.push_back({from,to,0});
    }
    while(promo_leftB){
        int to=bb::pop_lsb(promo_leftB);
        int from=to+9;
        for(int p=1;p<=4;++p) out.push_back({from,to,p});
    }
    Bitboard promo_rightB = rightB & promo_rankB;
    Bitboard normal_rightB = rightB & ~promo_rankB;
    while(normal_rightB){
        int to=bb::pop_lsb(normal_rightB);
        int from=to+7;
        out.push_back({from,to,0});
    }
    while(promo_rightB){
        int to=bb::pop_lsb(promo_rightB);
        int from=to+7;
        for(int p=1;p<=4;++p) out.push_back({from,to,p});
    }
    }
}

void generate_knight_moves(const Position& pos, MoveList& out){
    init_attack_tables();
    Color stm = pos.side;
    Bitboard wn = pos.piece_bb[WHITE][KNIGHT];
    Bitboard bn = pos.piece_bb[BLACK][KNIGHT];
    Bitboard occW = pos.occupied_bb[WHITE];
    Bitboard occB = pos.occupied_bb[BLACK];

    if(stm == WHITE){
        while(wn){
            int from = bb::pop_lsb(wn);
            Bitboard targets = KNIGHT_ATTACKS[from] & ~occW;
            while(targets){
                int to = bb::pop_lsb(targets);
                out.push_back({from,to,0});
            }
        }
    } else {
        while(bn){
            int from = bb::pop_lsb(bn);
            Bitboard targets = KNIGHT_ATTACKS[from] & ~occB;
            while(targets){
                int to = bb::pop_lsb(targets);
                out.push_back({from,to,0});
            }
        }
    }
}

void generate_sliding_moves(const Position& pos, MoveList& out){
    Color stm = pos.side;
    Bitboard occW = pos.occupied_bb[WHITE];
    Bitboard occB = pos.occupied_bb[BLACK];
    auto gen_dir = [&](int from, int df, int dr, Bitboard blockers){
        int f = from % 8; int r = from / 8;
        for(int s=1;;++s){
            int nf=f+df*s, nr=r+dr*s;
            if(nf<0||nf>=8||nr<0||nr>=8) break;
            int to = nr*8+nf;
            Bitboard mask = 1ULL<<to;
            if(blockers & mask){
                if((stm==WHITE?occB:occW)&mask) out.push_back({from,to,0});
                break;
            }
            out.push_back({from,to,0});
        }
    };

    Bitboard bishops = pos.piece_bb[stm][BISHOP];
    Bitboard rooks   = pos.piece_bb[stm][ROOK];
    Bitboard queens  = pos.piece_bb[stm][QUEEN];
    Bitboard blockers = pos.all_occupied;

    Bitboard pieces = bishops | queens;
    while(pieces){
        int from = bb::pop_lsb(pieces);
        gen_dir(from,1,1,blockers);
        gen_dir(from,-1,1,blockers);
        gen_dir(from,1,-1,blockers);
        gen_dir(from,-1,-1,blockers);
    }

    pieces = rooks | queens;
    while(pieces){
        int from = bb::pop_lsb(pieces);
        gen_dir(from,1,0,blockers);
        gen_dir(from,-1,0,blockers);
        gen_dir(from,0,1,blockers);
        gen_dir(from,0,-1,blockers);
    }
}

void generate_king_moves(const Position& pos, MoveList& out){
    Color stm = pos.side;
    Bitboard king = pos.piece_bb[stm][KING];
    int from = bb::pop_lsb(king);
    Bitboard occOwn = pos.occupied_bb[stm];
    const int dr[8]={1,1,1,0,0,-1,-1,-1};
    const int df[8]={-1,0,1,-1,1,-1,0,1};
    for(int i=0;i<8;++i){
        int nf=(from%8)+df[i], nr=(from/8)+dr[i];
        if(nf<0||nf>=8||nr<0||nr>=8) continue;
        int to=nr*8+nf;
        if(!(occOwn & (1ULL<<to)))
            out.push_back({from,to,0});
    }
    // castling very simplified
    if(stm==WHITE){
        if((pos.castling_rights & 1) && !(pos.all_occupied & ((1ULL<<5)|(1ULL<<6))))
            out.push_back({4,6,0});
        if((pos.castling_rights & 2) && !(pos.all_occupied & ((1ULL<<1)|(1ULL<<2)|(1ULL<<3))))
            out.push_back({4,2,0});
    }else{
        if((pos.castling_rights & 4) && !(pos.all_occupied & ((1ULL<<61)|(1ULL<<62))))
            out.push_back({60,62,0});
        if((pos.castling_rights & 8) && !(pos.all_occupied & ((1ULL<<57)|(1ULL<<58)|(1ULL<<59))))
            out.push_back({60,58,0});
    }
}

MoveList generate_pseudo_moves(const Position& pos){
    MoveList ml; ml.reserve(64);
    generate_pawn_moves(pos, ml);
    generate_knight_moves(pos, ml);
    generate_sliding_moves(pos, ml);
    generate_king_moves(pos, ml);
    return ml;
}

MoveList generate_moves(const Position& pos){
    MoveList ml;
    auto pseudo = generate_pseudo_moves(pos);
    for(const auto& m: pseudo){
        Position next = pos;
        next.do_move(m);
        if(!next.in_check(pos.side)) ml.push_back(m);
    }
    return ml;
}

} // namespace movegen
