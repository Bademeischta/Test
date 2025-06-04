from typing import Iterable

import torch
from torch.utils.data import DataLoader, TensorDataset


class Trainer:
    def __init__(self, network, buffer, optimizer, batch_size=32, epochs=1):
        self.network = network
        self.buffer = buffer
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.epochs = epochs

    def train(self):
        if len(self.buffer) < self.batch_size:
            return
        states, policies, values = self.buffer.sample(self.batch_size)
        states = torch.tensor(states, dtype=torch.float32)
        policies = torch.tensor(policies, dtype=torch.float32)
        values = torch.tensor(values, dtype=torch.float32)
        dataset = TensorDataset(states, policies, values)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        for _ in range(self.epochs):
            for s, p_target, v_target in loader:
                log_p, v = self.network(s)
                loss_policy = -(p_target * log_p).sum(dim=1).mean()
                loss_value = torch.mean((v.view(-1) - v_target)**2)
                loss = loss_policy + loss_value
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
