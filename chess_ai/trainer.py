"""Training utilities for the chess network."""

from typing import Iterable

import torch
from torch.utils.data import DataLoader, Dataset

from .config import Config


class Trainer:
    def __init__(self, network, buffer, optimizer, batch_size: int = Config.BATCH_SIZE, epochs: int = 1):
        self.network = network.to(Config.DEVICE)
        self.buffer = buffer
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.epochs = epochs
        if Config.LR_DECAY_TYPE == "cosine":
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=Config.LR_WARMUP_STEPS
            )
        else:
            self.scheduler = None

    def train(self):
        """Train the network for one epoch using data from the replay buffer."""

        if len(self.buffer) < self.batch_size:
            return

        class BufferDataset(Dataset):
            def __len__(self_inner):
                return len(self.buffer)

            def __getitem__(self_inner, idx):
                state = self.buffer.file["states"][idx]
                pi = self.buffer.file["pis"][idx]
                z = self.buffer.file["zs"][idx]
                return (
                    torch.tensor(state, dtype=torch.float32),
                    torch.tensor(pi, dtype=torch.float32),
                    torch.tensor(z, dtype=torch.float32),
                )

        loader = DataLoader(BufferDataset(), batch_size=self.batch_size, shuffle=True)
        scaler = torch.cuda.amp.GradScaler()
        for _ in range(self.epochs):
            for s, p_target, v_target in loader:
                s = s.to(Config.DEVICE)
                p_target = p_target.to(Config.DEVICE)
                v_target = v_target.to(Config.DEVICE)
                self.optimizer.zero_grad()
                with torch.cuda.amp.autocast():
                    log_p, v = self.network(s)
                    loss_policy = -(p_target * log_p).sum(dim=1).mean()
                    loss_value = torch.mean((v.view(-1) - v_target) ** 2)
                    loss = loss_policy + loss_value
                scaler.scale(loss).backward()
                scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), 1.0)
                scaler.step(self.optimizer)
                scaler.update()
            if self.scheduler:
                self.scheduler.step()
