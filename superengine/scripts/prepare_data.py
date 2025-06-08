#!/usr/bin/env python3

import sys

import chess
import chess.pgn
import numpy as np


def is_quiet(board):
    return not board.is_check() and (
        not board.is_capture(board.peek()) if board.move_stack else True
    )


X, y = [], []
for pgn in sys.argv[1:]:
    with open(pgn) as fh:
        while True:
            game = chess.pgn.read_game(fh)
            if game is None:
                break
            board = game.board()
            for move in game.mainline_moves():
                board.push(move)
                if is_quiet(board):
                    X.append(board.fen())
                    y.append(game.headers.get("StockfishEval", "0"))
np.save("fen.npy", X)
np.save("score.npy", y)
