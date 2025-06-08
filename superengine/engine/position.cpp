#include "position.h"

#include "movegen.h"

namespace movegen {
MoveList generate_pseudo_moves(const Position& pos);
}
#include <cctype>
#include <sstream>

Position::Position(const std::string& fen) {
    // initialize bitboards
    for (int c = 0; c < 2; ++c)
        for (int p = 0; p < PIECE_NB; ++p) piece_bb[c][p] = 0ULL;
    occupied_bb[WHITE] = occupied_bb[BLACK] = 0ULL;
    all_occupied = 0ULL;
    side = WHITE;

    std::istringstream ss(fen);
    std::string board, stm, castling = "-", ep = "-";
    if (!(ss >> board >> stm)) return;  // invalid FEN, leave empty position
    if (!(ss >> castling)) castling = "-";
    if (!(ss >> ep)) ep = "-";
    if (!(ss >> halfmove_clock)) halfmove_clock = 0;
    if (!(ss >> fullmove_number)) fullmove_number = 1;

    side = (stm == "w" ? WHITE : BLACK);
    castling_rights = 0;
    if (castling.find('K') != std::string::npos) castling_rights |= 1;
    if (castling.find('Q') != std::string::npos) castling_rights |= 2;
    if (castling.find('k') != std::string::npos) castling_rights |= 4;
    if (castling.find('q') != std::string::npos) castling_rights |= 8;
    en_passant = -1;
    if (ep != "-" && ep.size() == 2) {
        int file = ep[0] - 'a';
        int rank = ep[1] - '1';
        en_passant = rank * 8 + file;
    }

    int sq = 56;  // start at a8
    for (char ch : board) {
        if (ch == '/') {
            sq -= 16;
            continue;
        }
        if (std::isdigit(ch)) {
            sq += ch - '0';
            continue;
        }
        Color c = std::isupper(ch) ? WHITE : BLACK;
        char l = std::tolower(ch);
        Piece pc;
        switch (l) {
            case 'p':
                pc = PAWN;
                break;
            case 'n':
                pc = KNIGHT;
                break;
            case 'b':
                pc = BISHOP;
                break;
            case 'r':
                pc = ROOK;
                break;
            case 'q':
                pc = QUEEN;
                break;
            case 'k':
                pc = KING;
                break;
            default:
                continue;
        }
        piece_bb[c][pc] |= 1ULL << sq;
        sq++;
    }
    // compute occupancy
    for (int c = 0; c < 2; ++c) {
        Bitboard occ = 0ULL;
        for (int p = 0; p < PIECE_NB; ++p) occ |= piece_bb[c][p];
        occupied_bb[c] = occ;
    }
    all_occupied = occupied_bb[WHITE] | occupied_bb[BLACK];
}

Piece Position::piece_on(int sq) const {
    Bitboard mask = 1ULL << sq;
    for (int c = 0; c < 2; ++c) {
        for (int p = 0; p < PIECE_NB; ++p) {
            if (piece_bb[c][p] & mask) return static_cast<Piece>(p);
        }
    }
    return PIECE_NB;  // invalid
}

bool Position::in_check(Color c) const {
    movegen::init_attack_tables();
    Bitboard king_bb = piece_bb[c][KING];
    if (!king_bb) return false;
    int king_sq = __builtin_ctzll(king_bb);
    Color o = c == WHITE ? BLACK : WHITE;
    if (movegen::KNIGHT_ATTACKS[king_sq] & piece_bb[o][KNIGHT]) return true;
    Bitboard pawn_attacks = c == WHITE
                                ? ((piece_bb[o][PAWN] & ~bb::FILE_H) << 9 | (piece_bb[o][PAWN] & ~bb::FILE_A) << 7)
                                : ((piece_bb[o][PAWN] & ~bb::FILE_A) >> 9 | (piece_bb[o][PAWN] & ~bb::FILE_H) >> 7);
    if (pawn_attacks & king_bb) return true;
    Bitboard rookers = piece_bb[o][ROOK] | piece_bb[o][QUEEN];
    if (movegen::rook_attacks(king_sq, all_occupied) & rookers) return true;
    Bitboard bishops = piece_bb[o][BISHOP] | piece_bb[o][QUEEN];
    if (movegen::bishop_attacks(king_sq, all_occupied) & bishops) return true;
    if (movegen::KING_ATTACKS[king_sq] & piece_bb[o][KING]) return true;
    return false;
}

void Position::do_move(const movegen::Move& m) {
    Piece pc = piece_on(m.from);
    Color c = side;
    Color opp = c == WHITE ? BLACK : WHITE;
    Bitboard from_mask = 1ULL << m.from;
    Bitboard to_mask = 1ULL << m.to;
    Piece captured_piece = piece_on(m.to);
    // update castling rights on capture of rooks
    if (captured_piece == ROOK) {
        if (opp == WHITE) {
            if (m.to == 0) castling_rights &= ~2;  // white queenside rook captured
            if (m.to == 7) castling_rights &= ~1;  // white kingside rook captured
        } else {
            if (m.to == 56) castling_rights &= ~8;  // black queenside rook captured
            if (m.to == 63) castling_rights &= ~4;  // black kingside rook captured
        }
    }
    // update castling rights when moving king or rook
    if (pc == KING) {
        if (c == WHITE)
            castling_rights &= ~3;
        else
            castling_rights &= ~12;
    }
    if (pc == ROOK) {
        if (c == WHITE) {
            if (m.from == 0) castling_rights &= ~2;
            if (m.from == 7) castling_rights &= ~1;
        } else {
            if (m.from == 56) castling_rights &= ~8;
            if (m.from == 63) castling_rights &= ~4;
        }
    }
    // handle en-passant capture
    if (pc == PAWN && m.to == en_passant && en_passant != -1) {
        Bitboard cap_mask = c == WHITE ? (to_mask >> 8) : (to_mask << 8);
        for (int p = 0; p < PIECE_NB; ++p) piece_bb[opp][p] &= ~cap_mask;
    }
    for (int p = 0; p < PIECE_NB; ++p) {
        piece_bb[c][p] &= ~from_mask;
        piece_bb[c][p] &= ~to_mask;
        piece_bb[opp][p] &= ~to_mask;
    }
    piece_bb[c][pc] |= to_mask;
    if (m.promo) {
        piece_bb[c][pc] &= ~to_mask;
        piece_bb[c][m.promo] |= to_mask;
    }
    // handle castling rook move
    if (pc == KING) {
        if (c == WHITE) {
            if (m.from == 4 && m.to == 6) {
                // king side
                Bitboard rfrom = 1ULL << 7, rto = 1ULL << 5;
                for (int p = 0; p < PIECE_NB; ++p) {
                    piece_bb[c][p] &= ~rfrom;
                    piece_bb[c][p] &= ~rto;
                }
                piece_bb[c][ROOK] |= rto;
            } else if (m.from == 4 && m.to == 2) {
                Bitboard rfrom = 1ULL << 0, rto = 1ULL << 3;
                for (int p = 0; p < PIECE_NB; ++p) {
                    piece_bb[c][p] &= ~rfrom;
                    piece_bb[c][p] &= ~rto;
                }
                piece_bb[c][ROOK] |= rto;
            }
        } else {
            if (m.from == 60 && m.to == 62) {
                Bitboard rfrom = 1ULL << 63, rto = 1ULL << 61;
                for (int p = 0; p < PIECE_NB; ++p) {
                    piece_bb[c][p] &= ~rfrom;
                    piece_bb[c][p] &= ~rto;
                }
                piece_bb[c][ROOK] |= rto;
            } else if (m.from == 60 && m.to == 58) {
                Bitboard rfrom = 1ULL << 56, rto = 1ULL << 59;
                for (int p = 0; p < PIECE_NB; ++p) {
                    piece_bb[c][p] &= ~rfrom;
                    piece_bb[c][p] &= ~rto;
                }
                piece_bb[c][ROOK] |= rto;
            }
        }
    }
    en_passant = -1;
    if (pc == PAWN) {
        if (c == WHITE && m.from / 8 == 1 && m.to / 8 == 3)
            en_passant = m.from + 8;
        else if (c == BLACK && m.from / 8 == 6 && m.to / 8 == 4)
            en_passant = m.from - 8;
    }
    // recompute occupancy
    for (int col = 0; col < 2; ++col) {
        Bitboard occ = 0ULL;
        for (int p = 0; p < PIECE_NB; ++p) occ |= piece_bb[col][p];
        occupied_bb[col] = occ;
    }
    all_occupied = occupied_bb[WHITE] | occupied_bb[BLACK];
    side = opp;
}

std::string Position::to_fen() const {
    std::string board = "";
    for (int r = 7; r >= 0; --r) {
        int empty = 0;
        for (int f = 0; f < 8; ++f) {
            int sq = r * 8 + f;
            Piece pc = piece_on(sq);
            if (pc == PIECE_NB) {
                empty++;
                continue;
            }
            if (empty) {
                board += std::to_string(empty);
                empty = 0;
            }
            char c = '?';
            switch (pc) {
                case PAWN:
                    c = 'p';
                    break;
                case KNIGHT:
                    c = 'n';
                    break;
                case BISHOP:
                    c = 'b';
                    break;
                case ROOK:
                    c = 'r';
                    break;
                case QUEEN:
                    c = 'q';
                    break;
                case KING:
                    c = 'k';
                    break;
                default:
                    break;
            }
            if (occupied_bb[WHITE] & (1ULL << sq)) c = toupper(c);
            board += c;
        }
        if (empty) board += std::to_string(empty);
        if (r > 0) board += '/';
    }
    std::string stm = side == WHITE ? "w" : "b";
    std::string cast = "";
    if (castling_rights & 1) cast += 'K';
    if (castling_rights & 2) cast += 'Q';
    if (castling_rights & 4) cast += 'k';
    if (castling_rights & 8) cast += 'q';
    if (cast.empty()) cast = "-";
    std::string ep =
        en_passant == -1 ? "-" : std::string(1, 'a' + (en_passant % 8)) + std::to_string(en_passant / 8 + 1);
    return board + " " + stm + " " + cast + " " + ep + " " + std::to_string(halfmove_clock) + " " +
           std::to_string(fullmove_number);
}
