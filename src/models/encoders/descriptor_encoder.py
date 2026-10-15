import torch.nn as nn


class DescriptorEncoder(nn.Module):
    def __init__(self, input_dim=7, hidden_dim=64, output_dim=128):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return self.encoder(x)