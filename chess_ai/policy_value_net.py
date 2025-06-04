import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, filters: int):
        super().__init__()
        self.conv1 = nn.Conv2d(filters, filters, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(filters)
        self.conv2 = nn.Conv2d(filters, filters, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(filters)

    def forward(self, x):
        identity = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity
        out = F.relu(out)
        return out


class PolicyValueNet(nn.Module):
    def __init__(self, in_channels: int, action_size: int, num_blocks: int = 19, filters: int = 256):
        super().__init__()
        self.conv0 = nn.Conv2d(in_channels, filters, kernel_size=3, padding=1)
        self.bn0 = nn.BatchNorm2d(filters)
        self.res_blocks = nn.ModuleList([ResidualBlock(filters) for _ in range(num_blocks)])
        # policy head
        self.conv_policy = nn.Conv2d(filters, 2, kernel_size=1)
        self.bn_policy = nn.BatchNorm2d(2)
        self.fc_policy = nn.Linear(2 * 8 * 8, action_size)
        # value head
        self.conv_value = nn.Conv2d(filters, 1, kernel_size=1)
        self.bn_value = nn.BatchNorm2d(1)
        self.fc_value1 = nn.Linear(1 * 8 * 8, 256)
        self.fc_value2 = nn.Linear(256, 1)

    def forward(self, x):
        x = F.relu(self.bn0(self.conv0(x)))
        for block in self.res_blocks:
            x = block(x)
        p = F.relu(self.bn_policy(self.conv_policy(x)))
        p = p.view(p.size(0), -1)
        p = F.log_softmax(self.fc_policy(p), dim=1)
        v = F.relu(self.bn_value(self.conv_value(x)))
        v = v.view(v.size(0), -1)
        v = F.relu(self.fc_value1(v))
        v = torch.tanh(self.fc_value2(v))
        return p, v
