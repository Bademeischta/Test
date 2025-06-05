#include "movegen.h"
#include "bitboard.h"
#include <array>

namespace movegen {

static const std::array<int,8> rook_dirs = {8,-8,1,-1, 0,0,0,0};
static const std::array<int,8> bishop_dirs = {9,7,-7,-9, 0,0,0,0};
static const std::array<int,8> queen_dirs = {8,-8,1,-1,9,7,-7,-9};

static inline bool on_board(int sq) { return sq>=0 && sq<64; }

static inline bool file_a(int sq){ return (sq & 7) == 0; }
static inline bool file_h(int sq){ return (sq & 7) == 7; }

static Bitboard slide(int sq, const std::array<int,8>& dirs, Bitboard occ) {
    Bitboard attacks = 0;
    for(int d: dirs) {
        if(d==0) continue;
        int s = sq;
        while(true){
            int f = s & 7;
            if(d==1 && f==7) break;
            if(d==-1 && f==0) break;
            if((d==9 || d==-7) && f==7) break;
            if((d==7 || d==-9) && f==0) break;
            s += d;
            if(!on_board(s)) break;
            attacks |= 1ULL<<s;
            if(occ & (1ULL<<s)) break;
        }
    }
    return attacks;
}

MoveList generate_pseudo_legal(const Position& pos) {
    MoveList ml;
    Color us = pos.side_to_move();
    Color them = us==WHITE?BLACK:WHITE;
    Bitboard occ = pos.occupied();
    Bitboard own_occ = pos.occupied(us);
    Bitboard enemy_occ = pos.occupied(them);

    // pawns
    Bitboard pawns = pos.pieces(PAWN, us);
    int push_dir = us==WHITE?8:-8;
    Bitboard single = us==WHITE? (pawns<<8) : (pawns>>8);
    single &= ~occ;
    while(single){
        int to = bb::pop_lsb(single);
        int from = to - push_dir;
        if((us==WHITE && to>=56) || (us==BLACK && to<8)) {
            ml.push_back({from,to,QUIET|PROMOTION,QUEEN});
        } else {
            ml.push_back({from,to,QUIET,PAWN});
        }
    }
    Bitboard startRank = us==WHITE? 0x000000000000FF00ULL : 0x00FF000000000000ULL;
    Bitboard unmoved = pawns & startRank;
    Bitboard step = us==WHITE? ((unmoved<<8)&~occ) : ((unmoved>>8)&~occ);
    Bitboard dbl = us==WHITE? ((step<<8)&~occ) : ((step>>8)&~occ);
    while(dbl){
        int to = bb::pop_lsb(dbl);
        int from = to - 2*push_dir;
        ml.push_back({from,to,DOUBLE_PUSH,PAWN});
    }
    // captures
    Bitboard left = us==WHITE? (pawns<<7) : (pawns>>9);
    Bitboard right = us==WHITE? (pawns<<9) : (pawns>>7);
    left &= enemy_occ;
    right &= enemy_occ;
    while(left){
        int to = bb::pop_lsb(left);
        int from = to - (push_dir + (us==WHITE?-1:1));
        int flags = CAPTURE;
        if((us==WHITE && to>=56) || (us==BLACK && to<8)) flags|=PROMOTION;
        ml.push_back({from,to,flags,QUEEN});
    }
    while(right){
        int to = bb::pop_lsb(right);
        int from = to - (push_dir + (us==WHITE?1:-1));
        int flags = CAPTURE;
        if((us==WHITE && to>=56) || (us==BLACK && to<8)) flags|=PROMOTION;
        ml.push_back({from,to,flags,QUEEN});
    }
    // en passant
    if(pos.ep_square()!=-1){
        int ep = pos.ep_square();
        Bitboard ep_bb = 1ULL<<ep;
        Bitboard cap_left = us==WHITE? (pawns<<7) : (pawns>>9);
        Bitboard cap_right = us==WHITE? (pawns<<9) : (pawns>>7);
        if(cap_left & ep_bb){
            int from = ep - (push_dir + (us==WHITE?-1:1));
            ml.push_back({from,ep,EN_PASSANT,PAWN});
        }
        if(cap_right & ep_bb){
            int from = ep - (push_dir + (us==WHITE?1:-1));
            ml.push_back({from,ep,EN_PASSANT,PAWN});
        }
    }

    // knights
    static const int kn_dirs[8] = {17,15,10,6,-17,-15,-10,-6};
    Bitboard knights = pos.pieces(KNIGHT, us);
    while(knights){
        int from = bb::pop_lsb(knights);
        for(int d: kn_dirs){
            int to = from + d;
            int f = from&7;
            if(d==17 && f==7) continue;
            if(d==10 && (f>=6)) continue;
            if(d==6 && f<=1) continue;
            if(d==15 && f==0) continue;
            if(d==-17 && f==0) continue;
            if(d==-10 && f<=1) continue;
            if(d==-6 && f>=6) continue;
            if(d==-15 && f==7) continue;
            if(!on_board(to)) continue;
            Bitboard dst_bb = 1ULL<<to;
            if(dst_bb & own_occ) continue;
            int flags = (dst_bb & enemy_occ) ? CAPTURE : QUIET;
            ml.push_back({from,to,flags,PAWN});
        }
    }

    // bishops
    Bitboard bishops = pos.pieces(BISHOP, us);
    while(bishops){
        int from = bb::pop_lsb(bishops);
        Bitboard att = slide(from, bishop_dirs, occ) & ~own_occ;
        Bitboard tmp = att;
        while(tmp){
            int to = bb::pop_lsb(tmp);
            int flags = ((1ULL<<to) & enemy_occ) ? CAPTURE : QUIET;
            ml.push_back({from,to,flags,PAWN});
        }
    }

    // rooks
    Bitboard rooks = pos.pieces(ROOK, us);
    while(rooks){
        int from = bb::pop_lsb(rooks);
        Bitboard att = slide(from, rook_dirs, occ) & ~own_occ;
        Bitboard tmp = att;
        while(tmp){
            int to = bb::pop_lsb(tmp);
            int flags = ((1ULL<<to) & enemy_occ) ? CAPTURE : QUIET;
            ml.push_back({from,to,flags,PAWN});
        }
    }

    // queens
    Bitboard queens = pos.pieces(QUEEN, us);
    while(queens){
        int from = bb::pop_lsb(queens);
        Bitboard att = slide(from, queen_dirs, occ) & ~own_occ;
        Bitboard tmp = att;
        while(tmp){
            int to = bb::pop_lsb(tmp);
            int flags = ((1ULL<<to) & enemy_occ) ? CAPTURE : QUIET;
            ml.push_back({from,to,flags,PAWN});
        }
    }

    // king
    static const int king_dirs[8] = {8,-8,1,-1,9,7,-7,-9};
    Bitboard king = pos.pieces(KING, us);
    if(king){
        int from = bb::pop_lsb(king);
        for(int d: king_dirs){
            int to = from + d;
            int f = from&7;
            if(d==1 && f==7) continue;
            if(d==-1 && f==0) continue;
            if((d==9||d==-7) && f==7) continue;
            if((d==7||d==-9) && f==0) continue;
            if(!on_board(to)) continue;
            Bitboard dst_bb = 1ULL<<to;
            if(dst_bb & own_occ) continue;
            int flags = (dst_bb & enemy_occ)?CAPTURE:QUIET;
            ml.push_back({from,to,flags,PAWN});
        }
    }

    // legality filtering
    MoveList legal;
    for(const Move& m : ml){
        Position next = pos;
        next.do_move(m);
        if(!next.in_check(us))
            legal.push_back(m);
    }
    return legal;
}

}
