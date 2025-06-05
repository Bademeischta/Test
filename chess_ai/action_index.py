import chess

PROMOTIONS = [None, chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT]
ACTION_SIZE = 64 * 64 * len(PROMOTIONS)


def move_to_index(move: chess.Move) -> int:
    promo_idx = PROMOTIONS.index(move.promotion)
    return (
        ((move.from_square * 64) + move.to_square) * len(PROMOTIONS)
        + promo_idx
    )


def index_to_move(index: int) -> chess.Move:
    promo_idx = index % len(PROMOTIONS)
    idx = index // len(PROMOTIONS)
    from_sq = idx // 64
    to_sq = idx % 64
    promotion = PROMOTIONS[promo_idx]
    return chess.Move(from_sq, to_sq, promotion=promotion)
