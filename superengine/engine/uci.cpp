#include <iostream>
#include "search.h"

void uci_loop() {
    std::string cmd;
    Search search;
    Position pos = START_POS;
    while (std::getline(std::cin, cmd)) {
        if (cmd == "uci") {
            std::cout << "id name SuperEngine\nuciok\n";
        } else if (cmd == "isready") {
            std::cout << "readyok\n";
        } else if (cmd.rfind("position",0)==0) {
            auto fen = cmd.substr(9);
            if (fen == " startpos" || fen.empty())
                pos = START_POS;
            else
                pos = Position::from_fen(fen);
        } else if (cmd.rfind("go",0)==0) {
            int score = search.pv_node(pos, -32000, 32000, 3);
            std::cout << "bestmove 0000\n";
        } else if (cmd == "quit") {
            break;
        }
    }
}
