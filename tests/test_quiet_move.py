import chess

from chess_ai.game_environment import GameEnvironment


def test_is_quiet_move():
    env = GameEnvironment()

    # Quiet opening move
    move = chess.Move.from_uci("e2e4")
    assert env.is_quiet_move(move)
    env.step(move)

    # Capture is not quiet
    env.step(chess.Move.from_uci("d7d5"))
    capture = chess.Move.from_uci("e4d5")
    assert not env.is_quiet_move(capture)

    # Check move (Qxf7+) is not quiet
    env.reset()
    env.step(chess.Move.from_uci("e2e4"))
    env.step(chess.Move.from_uci("e7e5"))
    env.step(chess.Move.from_uci("d1h5"))
    env.step(chess.Move.from_uci("b8c6"))
    check_move = chess.Move.from_uci("h5f7")
    assert not env.is_quiet_move(check_move)

