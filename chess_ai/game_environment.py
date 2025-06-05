import chess
import numpy as np


class GameEnvironment:
    """Wrapper around python-chess for board management and encoding."""

    NUM_CHANNELS = 18

    def __init__(self):
        self.board = chess.Board()

    def reset(self):
        self.board.reset()
        return self.get_state()

    def get_state(self):
        return self.encode_board(self.board)

    def legal_moves(self):
        return list(self.board.legal_moves)

    def step(self, move):
        self.board.push(move)
        done = self.board.is_game_over()
        reward = 0.0
        if done:
            result = self.board.result()
            if result == '1-0':
                reward = 1.0
            elif result == '0-1':
                reward = -1.0
            else:
                reward = 0.0
        return self.get_state(), reward, done

    def undo(self):
        self.board.pop()

    def is_quiet_move(self, move: chess.Move) -> bool:
        """Return True if ``move`` is non-capturing, non-checking and current
        player is not already in check."""
        return (
            not self.board.is_capture(move)
            and not self.board.gives_check(move)
            and not self.board.is_check()
        )

    @classmethod
    def encode_board(cls, board: chess.Board):
        """Encode board into an 8x8x18 tensor of binary features."""
        planes = np.zeros((cls.NUM_CHANNELS, 8, 8), dtype=np.float32)
        piece_map = board.piece_map()
        for square, piece in piece_map.items():
            row = square // 8
            col = square % 8
            offset = 0 if piece.color == chess.WHITE else 6
            piece_idx = {
                chess.PAWN: 0,
                chess.ROOK: 1,
                chess.KNIGHT: 2,
                chess.BISHOP: 3,
                chess.QUEEN: 4,
                chess.KING: 5,
            }[piece.piece_type]
            planes[offset + piece_idx][row][col] = 1
        planes[12][:] = int(board.turn)
        planes[13][:] = int(board.has_kingside_castling_rights(chess.WHITE))
        planes[14][:] = int(board.has_queenside_castling_rights(chess.WHITE))
        planes[15][:] = int(board.has_kingside_castling_rights(chess.BLACK))
        planes[16][:] = int(board.has_queenside_castling_rights(chess.BLACK))
        if board.ep_square is not None:
            row = board.ep_square // 8
            col = board.ep_square % 8
            planes[17][row][col] = 1
        return planes
