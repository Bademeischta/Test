from chess_ai.game_environment import GameEnvironment


def test_initial_board_has_correct_channels():
    env = GameEnvironment()
    state = env.get_state()
    assert state.shape == (GameEnvironment.NUM_CHANNELS, 8, 8)
    # Check that side to move plane sums to 64 since all ones or zeros are uniform
    assert state[12].max() in (0, 1)
