
import torch.nn.functional as F

from pretraining.contextpred import mask_atoms


def pretrain_epoch(encoder, head, loader, optimizer, device, mask_rate=0.15):
    encoder.train()
    head.train()

    total_loss = 0.0
    correct = 0
    total_masked = 0

    for batch in loader:
        batch = batch.to(device)
        masked_x, mask, targets = mask_atoms(batch.x, mask_rate)

        embeddings = encoder.node_embeddings(masked_x, batch.edge_index)
        predictions = head(embeddings[mask])
        loss = F.cross_entropy(predictions, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        correct += (predictions.argmax(dim=1) == targets).sum().item()
        total_masked += targets.numel()

    if len(loader) == 0:
        raise ValueError("The pretraining dataset is empty.")

    return total_loss / len(loader), correct / max(total_masked, 1)
