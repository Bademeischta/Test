import os
from glob import glob

import torch


class NetworkManager:
    def __init__(self, checkpoint_dir="checkpoints"):
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(self.checkpoint_dir, exist_ok=True)

    def latest_checkpoint(self):
        files = glob(os.path.join(self.checkpoint_dir, "*.pt"))
        if not files:
            return None
        return max(files, key=os.path.getmtime)

    def save(self, model, optimizer, name):
        path = os.path.join(self.checkpoint_dir, f"{name}.pt")
        torch.save({"model_state": model.state_dict(), "optim_state": optimizer.state_dict()}, path)
        return path
