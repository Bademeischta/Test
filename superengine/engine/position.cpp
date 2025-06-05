#include "position.h"
#include "movegen.h"
#include <sstream>

const Position START_POS = Position::from_fen("startpos");

Position Position::from_fen(const std::string& fen) {
    Position pos;
    if (fen == "startpos")
        return from_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
    std::string board_part = fen;
    size_t space = fen.find(' ');
    if (space != std::string::npos) {
        board_part = fen.substr(0, space);
        pos.stm = (fen.substr(space + 1, 1) == "b") ? BLACK : WHITE;
    }
    int sq = 56; // starting from a8
    for (char c : board_part) {
        if (c == '/') { sq -= 16; continue; }
        if (c >= '1' && c <= '8') { sq += c - '0'; continue; }
        Color col = isupper(c) ? WHITE : BLACK;
        Piece pc;
        switch (tolower(c)) {
            case 'p': pc = PAWN; break;
            case 'n': pc = KNIGHT; break;
            case 'b': pc = BISHOP; break;
            case 'r': pc = ROOK; break;
            case 'q': pc = QUEEN; break;
            default: pc = KING; break;
        }
        pos.piece_bb[col][pc] |= 1ULL << sq;
        sq++;
    }
    return pos;
}

void Position::do_move(const movegen::Move& m) {
    Bitboard from_bb = 1ULL << m.from;
    Bitboard to_bb = 1ULL << m.to;
    for (int c=0;c<2;++c) {
        for (int p=0;p<PIECE_NB;++p) {
            if (piece_bb[c][p] & from_bb) {
                piece_bb[c][p] &= ~from_bb;
                piece_bb[c][p] |= to_bb;
            }
            // capture
            piece_bb[c][p] &= ~to_bb;
        }
    }
    stm = (stm == WHITE) ? BLACK : WHITE;
}
