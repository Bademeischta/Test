#include "position.h"
#include <sstream>
#include <cctype>

Position::Position(const std::string& fen) {
    // initialize bitboards
    for(int c=0;c<2;++c) for(int p=0;p<PIECE_NB;++p) piece_bb[c][p]=0ULL;
    occupied_bb[WHITE]=occupied_bb[BLACK]=0ULL;
    all_occupied=0ULL;
    side=WHITE;

    std::istringstream ss(fen);
    std::string board, stm, castling = "-", ep = "-";
    if (!(ss >> board >> stm))
        return; // invalid FEN, leave empty position
    if (!(ss >> castling)) castling = "-";
    if (!(ss >> ep)) ep = "-";
    if (!(ss >> halfmove_clock)) halfmove_clock = 0;
    if (!(ss >> fullmove_number)) fullmove_number = 1;

    side = (stm=="w"?WHITE:BLACK);
    castling_rights = 0;
    if(castling.find('K') != std::string::npos) castling_rights |= 1;
    if(castling.find('Q') != std::string::npos) castling_rights |= 2;
    if(castling.find('k') != std::string::npos) castling_rights |= 4;
    if(castling.find('q') != std::string::npos) castling_rights |= 8;
    en_passant = -1;
    if(ep != "-" && ep.size()==2){
        int file = ep[0]-'a';
        int rank = ep[1]-'1';
        en_passant = rank*8 + file;
    }

    int sq=56; // start at a8
    for(char ch : board){
        if(ch=='/'){ sq-=16; continue; }
        if(std::isdigit(ch)){ sq+= ch-'0'; continue; }
        Color c = std::isupper(ch)?WHITE:BLACK;
        char l = std::tolower(ch);
        Piece pc;
        switch(l){
            case 'p': pc=PAWN; break;
            case 'n': pc=KNIGHT; break;
            case 'b': pc=BISHOP; break;
            case 'r': pc=ROOK; break;
            case 'q': pc=QUEEN; break;
            case 'k': pc=KING; break;
            default: continue;
        }
        piece_bb[c][pc] |= 1ULL<<sq;
        sq++;
    }
    // compute occupancy
    for(int c=0;c<2;++c){
        Bitboard occ=0ULL;
        for(int p=0;p<PIECE_NB;++p) occ|=piece_bb[c][p];
        occupied_bb[c]=occ;
    }
    all_occupied = occupied_bb[WHITE]|occupied_bb[BLACK];
}

Piece Position::piece_on(int sq) const {
    Bitboard mask = 1ULL<<sq;
    for(int c=0;c<2;++c){
        for(int p=0;p<PIECE_NB;++p){
            if(piece_bb[c][p] & mask)
                return static_cast<Piece>(p);
        }
    }
    return PIECE_NB; // invalid
}
