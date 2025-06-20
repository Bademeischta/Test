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


def _fix_legacy_state_dict(state_dict: dict, prefix: str = "_orig_mod.") -> dict:
    """Remove wrapper prefixes from old checkpoints."""
    if any(k.startswith(prefix) for k in state_dict):
        return {k[len(prefix) :] if k.startswith(prefix) else k: v for k, v in state_dict.items()}
    return state_dict


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
        try:
            # PyTorch 2.6+: explizit weights_only=False, damit auch Optimizer-States etc. geladen werden
            checkpoint = torch.load(
                path,
                map_location=Config.DEVICE,
                weights_only=False
            )
        except TypeError:
            # Fallback für ältere PyTorch-Versionen ohne weights_only-Parameter
            checkpoint = torch.load(
                path,
                map_location=Config.DEVICE
            )
        base_model = _unwrap(model)
        state_dict = _fix_legacy_state_dict(checkpoint.get("model_state", {}))
        base_model.load_state_dict(state_dict)
        if optimizer and "optim_state" in checkpoint:
            optimizer.load_state_dict(checkpoint["optim_state"])
        return checkpoint
