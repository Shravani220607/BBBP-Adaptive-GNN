import torch.nn as nn


class Adapter(nn.Module):
    def __init__(self, input_dim=128, bottleneck_dim=32):
        super().__init__()

        self.down = nn.Linear(input_dim, bottleneck_dim)
        self.activation = nn.ReLU()
        self.up = nn.Linear(bottleneck_dim, input_dim)

    def forward(self, x):
        return x + self.up(self.activation(self.down(x)))