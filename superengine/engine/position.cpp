#include "position.h"
#include "movegen.h"

Position::Position() {
    board[WHITE][PAWN]   = 0x000000000000FF00ULL;
    board[WHITE][KNIGHT] = 0x0000000000000042ULL;
    board[WHITE][BISHOP] = 0x0000000000000024ULL;
    board[WHITE][ROOK]   = 0x0000000000000081ULL;
    board[WHITE][QUEEN]  = 0x0000000000000008ULL;
    board[WHITE][KING]   = 0x0000000000000010ULL;

    board[BLACK][PAWN]   = 0x00FF000000000000ULL;
    board[BLACK][KNIGHT] = 0x4200000000000000ULL;
    board[BLACK][BISHOP] = 0x2400000000000000ULL;
    board[BLACK][ROOK]   = 0x8100000000000000ULL;
    board[BLACK][QUEEN]  = 0x0800000000000000ULL;
    board[BLACK][KING]   = 0x1000000000000000ULL;

    stm = WHITE;
    ep = -1;
}

static inline bool on_board(int sq){ return sq>=0 && sq<64; }
static const int queen_dirs[8] = {8,-8,1,-1,9,7,-7,-9};
static const int knight_dirs[8] = {17,15,10,6,-17,-15,-10,-6};

bool Position::in_check(Color c) const {
    Color them = c==WHITE?BLACK:WHITE;
    Bitboard king_bb = board[c][KING];
    if(!king_bb) return false;
    int sq = __builtin_ctzll(king_bb);

    // pawn attacks
    Bitboard pawns = board[them][PAWN];
    Bitboard attacks = 0;
    if(c==WHITE) {
        attacks = (pawns>>7 & 0x007F7F7F7F7F7F7FULL) | (pawns>>9 & 0x00FEFEFEFEFEFEFEULL);
    } else {
        attacks = (pawns<<7 & 0xFEFEFEFEFEFEFE00ULL) | (pawns<<9 & 0x7F7F7F7F7F7F7F00ULL);
    }
    if(attacks & king_bb) return true;

    // knight
    Bitboard knights = board[them][KNIGHT];
    while(knights){
        int f = bb::pop_lsb(knights);
        for(int d: knight_dirs){
            int to = f + d;
            int file = f&7;
            if(d==17 && file==7) continue;
            if(d==10 && file>=6) continue;
            if(d==6 && file<=1) continue;
            if(d==15 && file==0) continue;
            if(d==-17 && file==0) continue;
            if(d==-10 && file<=1) continue;
            if(d==-6 && file>=6) continue;
            if(d==-15 && file==7) continue;
            if(on_board(to) && to==sq) return true;
        }
    }

    // sliding pieces
    Bitboard occ = occupied();
    for(int d: queen_dirs){
        int s = sq;
        while(true){
            int file = s&7;
            if(d==1 && file==7) break;
            if(d==-1 && file==0) break;
            if((d==9||d==-7) && file==7) break;
            if((d==7||d==-9) && file==0) break;
            s += d;
            if(!on_board(s)) break;
            Bitboard bb = 1ULL<<s;
            if(bb & (board[them][QUEEN] | board[them][ROOK] | board[them][BISHOP])){
                if((d==8||d==-8||d==1||d==-1) && (bb & (board[them][ROOK]|board[them][QUEEN]))) return true;
                if((d==9||d==7||d==-7||d==-9) && (bb & (board[them][BISHOP]|board[them][QUEEN]))) return true;
            }
            if(bb & occ) break;
        }
    }
    // king
    Bitboard kingAtt = 0;
    kingAtt |= (king_bb<<8)|(king_bb>>8);
    Bitboard east = (king_bb & 0xFEFEFEFEFEFEFEFEULL);
    Bitboard west = (king_bb & 0x7F7F7F7F7F7F7F7FULL);
    kingAtt |= (east<<1)|(west>>1)|(east<<9)|(east>>7)|(west<<7)|(west>>9);
    if(kingAtt & board[them][KING]) return true;
    return false;
}

void Position::do_move(const movegen::Move& m) {
    Color us = stm;
    Color them = us==WHITE?BLACK:WHITE;
    Bitboard from_bb = 1ULL<<m.from;
    Bitboard to_bb   = 1ULL<<m.to;

    Piece moved = KING;
    for(int p=0;p<6;++p){
        if(board[us][p] & from_bb){ moved = (Piece)p; break; }
    }

    for(int p=0;p<6;++p){
        if(board[them][p] & to_bb){ board[them][p] &= ~to_bb; }
    }
    if(m.flags & movegen::EN_PASSANT){
        int cap_sq = us==WHITE? m.to-8 : m.to+8;
        Bitboard cap_bb = 1ULL<<cap_sq;
        board[them][PAWN] &= ~cap_bb;
    }

    board[us][moved] &= ~from_bb;
    Piece destPiece = moved;
    if(m.flags & movegen::PROMOTION) destPiece = m.promotion;
    board[us][destPiece] |= to_bb;

    ep = -1;
    if(m.flags & movegen::DOUBLE_PUSH){
        ep = us==WHITE? m.to-8 : m.to+8;
    }

    stm = them;
}
