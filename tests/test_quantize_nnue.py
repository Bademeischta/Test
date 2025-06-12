import os
import sys

import torch

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "superengine", "scripts"))
from quantize_nnue import quantize_state_dict


def _dummy_state_dict(use_old_names=False):
    state = {}
    if use_old_names:
        state["l1.weight"] = torch.ones(256, 64)
        state["l1.bias"] = torch.zeros(256)
        state["l2.weight"] = torch.ones(1, 256)
        state["l2.bias"] = torch.zeros(1)
        state["out.weight"] = torch.ones(10, 128)
        state["out.bias"] = torch.zeros(10)
    else:
        state["fc_value1.weight"] = torch.ones(256, 64)
        state["fc_value1.bias"] = torch.zeros(256)
        state["fc_value2.weight"] = torch.ones(1, 256)
        state["fc_value2.bias"] = torch.zeros(1)
        state["fc_policy.weight"] = torch.ones(10, 128)
        state["fc_policy.bias"] = torch.zeros(10)
    return state


def test_quantize_new_names():
    sd = _dummy_state_dict()
    data = quantize_state_dict(sd)
    expected_size = (256 * 64 + 256 + 1 * 256 + 1 + 10 * 128 + 10) * 2
    assert len(data) == expected_size


def test_quantize_old_names():
    sd = _dummy_state_dict(use_old_names=True)
    data = quantize_state_dict(sd)
    expected_size = (256 * 64 + 256 + 1 * 256 + 1 + 10 * 128 + 10) * 2
    assert len(data) == expected_size
