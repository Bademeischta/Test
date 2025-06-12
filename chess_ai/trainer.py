"""Training utilities for the chess network."""

import torch
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import CosineAnnealingLR
import wandb
from tqdm.auto import tqdm

from .config import Config


class Trainer:
    def __init__(
        self,
        network,
        buffer,
        optimizer,
        batch_size: int = Config.BATCH_SIZE,
        epochs: int = 1,
        log_dir: str = Config.LOG_DIR,
        use_wandb: bool = False,
    ):
        self.network = network.to(Config.DEVICE)
        self.buffer = buffer
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.epochs = epochs
        self.writer = SummaryWriter(log_dir)
        self.use_wandb = use_wandb
        if self.use_wandb:
            wandb.init(project=Config.WANDB_PROJECT)
            wandb.watch(self.network)

    def train(self):
        """Train the network using data from the replay buffer."""

        if len(self.buffer) < self.batch_size:
            return
        states, policies, values = self.buffer.sample(self.batch_size)
        states = torch.tensor(states, dtype=torch.float32)
        policies = torch.tensor(policies, dtype=torch.float32)
        values = torch.tensor(values, dtype=torch.float32)
        dataset = TensorDataset(states, policies, values)
        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
        )
        scheduler = CosineAnnealingLR(self.optimizer, T_max=len(loader) * self.epochs)
        for epoch in range(self.epochs):
            epoch_loss = 0.0
            prog_bar = tqdm(loader, desc=f"Epoch {epoch + 1}/{self.epochs}", unit="batch")
            for batch_idx, (s, p_target, v_target) in enumerate(prog_bar):
                s = s.to(Config.DEVICE, non_blocking=True)
                p_target = p_target.to(Config.DEVICE, non_blocking=True)
                v_target = v_target.to(Config.DEVICE, non_blocking=True)
                log_p, v = self.network(s)
                loss_policy = -(p_target * log_p).sum(dim=1).mean()
                loss_value = torch.mean((v.view(-1) - v_target) ** 2)
                loss = loss_policy + loss_value
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), 5.0)
                self.optimizer.step()
                scheduler.step()
                epoch_loss += loss.item()
                global_step = epoch * len(loader) + batch_idx
                self.writer.add_scalar("Loss/train", loss.item(), global_step)
                prog_bar.set_postfix(loss=loss.item())
            avg_loss = epoch_loss / len(loader)
            print(f"Epoch {epoch + 1}/{self.epochs} - loss {avg_loss:.4f}")
            self.writer.add_scalar("loss", avg_loss, epoch)
            if self.use_wandb:
                wandb.log({"loss": avg_loss})
