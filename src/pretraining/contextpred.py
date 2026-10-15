
import torch
import torch.nn as nn


NUM_ATOM_TYPES = 119


def mask_atoms(x, mask_rate=0.15):
    if x.size(0) == 0:
        raise ValueError("Cannot mask atoms in an empty graph.")

    mask = torch.rand(x.size(0), device=x.device) < mask_rate

    if not mask.any():
        index = torch.randint(x.size(0), (1,), device=x.device)
        mask[index] = True

    targets = x[mask, 0].long().clamp(0, NUM_ATOM_TYPES - 1)
    masked_x = x.clone()
    masked_x[mask] = 0

    return masked_x, mask, targets


class MaskedAtomHead(nn.Module):
    def __init__(self, hidden_dim=128):
        super().__init__()
        self.linear = nn.Linear(hidden_dim, NUM_ATOM_TYPES)

    def forward(self, embeddings):
        return self.linear(embeddings)
