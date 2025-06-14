import os
from glob import glob

import torch

from .config import Config


def _unwrap(model):
    """Return the underlying PyTorch module for wrapped/compiled models."""
    if hasattr(model, "_orig_mod"):
        return model._orig_mod
    if hasattr(model, "_original_module"):
        return model._original_module
    return model


class NetworkManager:
    def __init__(self, checkpoint_dir: str = Config.CHECKPOINT_DIR):
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(self.checkpoint_dir, exist_ok=True)

    def latest_checkpoint(self):
        files = glob(os.path.join(self.checkpoint_dir, "*.pt"))
        if not files:
            return None
        return max(files, key=os.path.getmtime)

    def save(self, model, optimizer, name):
        path = os.path.join(self.checkpoint_dir, f"{name}.pt")
        base_model = _unwrap(model)
        torch.save(
            {
                "model_state": base_model.state_dict(),
                "optim_state": optimizer.state_dict(),
            },
            path,
        )
        return path

    def load(self, path, model, optimizer=None):
        """Load a checkpoint into ``model`` and optionally ``optimizer``."""
        checkpoint = torch.load(path, map_location=Config.DEVICE)
        base_model = _unwrap(model)
        base_model.load_state_dict(checkpoint["model_state"])
        if optimizer and "optim_state" in checkpoint:
            optimizer.load_state_dict(checkpoint["optim_state"])
        return checkpoint
