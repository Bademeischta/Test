#include <iostream>
#include <sstream>
#include "search.h"
#include "movegen.h"

void uci_loop() {
    std::string cmd;
    Search search;
    Position pos;
    while (std::getline(std::cin, cmd)) {
        if (cmd == "uci") {
            std::cout << "id name SuperEngine\nuciok\n";
        } else if (cmd == "isready") {
            std::cout << "readyok\n";
        } else if (cmd.rfind("position",0)==0) {
            auto rest = cmd.substr(8); // skip "position"
            std::istringstream ss(rest);
            std::string token;
            ss >> token;
            if(token == "startpos"){
                pos = Position("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
                if(ss >> token && token == "moves"){
                    while(ss >> token){
                        movegen::Move m;
                        m.from = (token[0]-'a') + (token[1]-'1')*8;
                        m.to   = (token[2]-'a') + (token[3]-'1')*8;
                        m.promo = 0;
                        if(token.size()==5){
                            switch(token[4]){case 'q':m.promo=QUEEN;break;case 'r':m.promo=ROOK;break;case 'b':m.promo=BISHOP;break;case 'n':m.promo=KNIGHT;break;}
                        }
                        pos.do_move(m);
                    }
                }
            } else if(token == "fen") {
                std::string fen;
                std::getline(ss, fen);
                fen.erase(0, fen.find_first_not_of(' '));
                pos = Position(fen);
            }
        } else if (cmd.rfind("go",0)==0) {
            auto moves = movegen::generate_moves(pos);
            movegen::Move best{};
            int bestScore = -32000;
            for(const auto& m : moves){
                Position next = pos;
                next.do_move(m);
                int score = -search.search(next, 3);
                if(score > bestScore){
                    bestScore = score;
                    best = m;
                }
            }
            char promoChar = 0;
            if(best.promo){
                promoChar = " nbrq"[best.promo];
            }
            std::string uci;
            uci += char('a' + best.from % 8);
            uci += char('1' + best.from / 8);
            uci += char('a' + best.to % 8);
            uci += char('1' + best.to / 8);
            if(promoChar) uci += promoChar;
            std::cout << "bestmove " << uci << "\n";
        } else if (cmd == "quit") {
            break;
        }
    }
}
