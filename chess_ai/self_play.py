import numpy as np
import torch

from .game_environment import GameEnvironment
from .mcts import MCTS
from .network import PolicyValueNet
from .replay_buffer import ReplayBuffer
from .network_manager import NetworkManager
from .config import Config


class SelfPlayWorker:
    """Plays games against itself and stores training data."""

    def __init__(self, manager: NetworkManager, buffer: ReplayBuffer):
        self.manager = manager
        self.buffer = buffer
        self.games_since_reload = 0
        self._load_model()

    def _load_model(self) -> None:
        path = self.manager.latest_checkpoint()
        self.net = PolicyValueNet(GameEnvironment.NUM_CHANNELS, 4672)
        if path:
            checkpoint = torch.load(path, map_location=Config.DEVICE)
            self.net.load_state_dict(checkpoint["model_state"])
        self.net.to(Config.DEVICE)
        self.net.eval()
        self.mcts = MCTS(self.net)

    def play_game(self) -> None:
        env = GameEnvironment()
        state = env.reset()
        trajectory = []
        current_player = 1
        move_count = 0
        while True:
            visit_counts = self.mcts.run(state)
            counts = np.array(list(visit_counts.values()), dtype=np.float32)
            if move_count < 30:
                probs = counts / np.sum(counts)
                move_idx = np.random.choice(list(visit_counts.keys()), p=probs)
            else:
                move_idx = max(visit_counts, key=visit_counts.get)
            move = list(env.legal_moves())[move_idx]
            trajectory.append((state, counts, current_player))
            state, reward, done = env.step(move)
            if done:
                for s, pi, player in trajectory:
                    z = reward if player == current_player else -reward
                    self.buffer.add(s, pi, z)
                break
            current_player *= -1
            move_count += 1
        self.games_since_reload += 1
        if self.games_since_reload >= Config.RELOAD_INTERVAL:
            self._load_model()
            self.games_since_reload = 0
