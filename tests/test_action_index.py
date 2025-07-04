import chess

from chess_ai.action_index import move_to_index, index_to_move, PROMOTIONS


def test_move_to_index_promotions():
    move = chess.Move.from_uci("a7a8q")
    base_idx = move_to_index(move)
    seen = {base_idx}
    for promo in PROMOTIONS[1:]:
        mv = chess.Move(0, 63, promotion=promo)
        idx = move_to_index(mv)
        assert index_to_move(idx) == mv
        assert idx not in seen
        seen.add(idx)

