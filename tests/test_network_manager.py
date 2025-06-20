import torch
from chess_ai.network_manager import NetworkManager
from chess_ai.policy_value_net import PolicyValueNet
from chess_ai.action_index import ACTION_SIZE
from chess_ai.game_environment import GameEnvironment


def test_load_handles_orig_mod_prefix(tmp_path):
    manager = NetworkManager(checkpoint_dir=tmp_path)
    net = PolicyValueNet(GameEnvironment.NUM_CHANNELS, ACTION_SIZE, num_blocks=1, filters=8)
    optim = torch.optim.SGD(net.parameters(), lr=0.1)
    path = manager.save(net, optim, "test")
    data = torch.load(path)
    # simulate old checkpoint with _orig_mod prefix
    data["model_state"] = {f"_orig_mod.{k}": v for k, v in data["model_state"].items()}
    torch.save(data, path)

    loaded = manager.load(path, net, optim)
    assert isinstance(loaded, dict)
    expected_keys = set(k[len("_orig_mod."):] if k.startswith("_orig_mod.") else k for k in data["model_state"]) 
    assert set(net.state_dict().keys()) == expected_keys
