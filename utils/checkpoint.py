
import os
import torch


def save_checkpoint(state, path):
    os.makedirs(os.path.dirname(str(path)) or ".", exist_ok=True)
    torch.save(state, path)


def load_checkpoint(path, device="cpu"):
    return torch.load(path, map_location=device, weights_only=False)
