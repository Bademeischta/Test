import lightning as L
import numpy as np
import torch
from torch import nn


class TinyNNUE(nn.Module):
    def __init__(self, inputs=768, h1=512, h2=256):
        super().__init__()
        self.l1 = nn.Linear(inputs, h1, bias=False)
        self.l2 = nn.Linear(h1, h2, bias=True)
        self.out = nn.Linear(h2, 1, bias=True)

    def forward(self, x):
        x = torch.relu(self.l1(x))
        x = torch.relu(self.l2(x))
        return self.out(x)


class Module(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.net = TinyNNUE()
        self.loss = nn.MSELoss()

    def training_step(self, batch, _):
        x, y = batch
        return self.loss(self.net(x), y)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), 1e-3)


data = torch.utils.data.TensorDataset(
    torch.from_numpy(np.load("fen_vec.npy")),
    torch.from_numpy(np.load("score.npy")).unsqueeze(1),
)
loader = torch.utils.data.DataLoader(
    data,
    batch_size=8192,
    shuffle=True,
    num_workers=4,
)
trainer = L.Trainer(
    max_epochs=4,
    precision="bf16-mixed",
    devices=8,
    accelerator="gpu",
)
trainer.fit(Module(), loader)
torch.save(trainer.model.net.state_dict(), "../nets/tiny_v1.nnue")
