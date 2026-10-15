import torch.nn as nn


class BBBClassifier(nn.Module):
    def __init__(self, input_dim=128):
        super().__init__()

        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.classifier(x)