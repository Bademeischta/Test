import os
import random
import subprocess
import uuid

import ray

ray.init(address="auto")


@ray.remote
def selfplay_game(net_path):
    seed = random.getrandbits(32)
    game_id = uuid.uuid4()
    cmd = [
        "./superengine",
        "selfplay",
        "--seed",
        str(seed),
        "--net",
        net_path,
        "--games",
        "1",
    ]
    p = subprocess.run(cmd, capture_output=True, text=True)
    os.makedirs("games", exist_ok=True)
    with open(f"games/{game_id}.pgn", "w") as f:
        f.write(p.stdout)
    return 1


futures = [selfplay_game.remote("nets/tiny_v1.nnue") for _ in range(10000)]
ray.get(futures)
