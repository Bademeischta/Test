import chess
from functools import lru_cache

PROMOTIONS = [None, chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT]
ACTION_SIZE = 64 * 64 * len(PROMOTIONS)


@lru_cache(maxsize=None)
def move_to_index(move: chess.Move) -> int:
    """Convert a chess move to a unique action index.

    The result is cached to speed up repeated conversions during search.
    """
    promo_idx = PROMOTIONS.index(move.promotion)
    return ((move.from_square * 64) + move.to_square) * len(PROMOTIONS) + promo_idx


@lru_cache(maxsize=None)
def index_to_move(index: int) -> chess.Move:
    """Inverse of :func:`move_to_index` with caching."""
    promo_idx = index % len(PROMOTIONS)
    idx = index // len(PROMOTIONS)
    from_sq = idx // 64
    to_sq = idx % 64
    promotion = PROMOTIONS[promo_idx]
    return chess.Move(from_sq, to_sq, promotion=promotion)
