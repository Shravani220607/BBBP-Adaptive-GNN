
import torch
from torch_geometric.loader import DataLoader

from datasets.chembl_dataset import ChEMBLDataset
from models.encoders.gin_encoder import GINEncoder
from pretraining.contextpred import MaskedAtomHead
from pretraining.pretrain_trainer import pretrain_epoch
from utils.paths import CHEMBL_CLEAN, PRETRAINED_CKPT, load_config, require
from utils.seed import set_seed


def pretrain():
    require(CHEMBL_CLEAN, hint="python main.py preprocess-chembl")

    config = load_config("pretrain")
    set_seed(config["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = ChEMBLDataset(CHEMBL_CLEAN, config["num_molecules"])

    if len(dataset) == 0:
        raise ValueError("No valid ChEMBL molecular graphs were loaded.")

    loader = DataLoader(
        dataset,
        batch_size=config["batch_size"],
        shuffle=True,
    )

    encoder = GINEncoder(input_dim=4).to(device)
    head = MaskedAtomHead().to(device)

    optimizer = torch.optim.Adam(
        list(encoder.parameters()) + list(head.parameters()),
        lr=config["lr"],
    )

    PRETRAINED_CKPT.parent.mkdir(parents=True, exist_ok=True)
    best_loss = float("inf")

    for epoch in range(config["epochs"]):
        loss, accuracy = pretrain_epoch(
            encoder,
            head,
            loader,
            optimizer,
            device,
            config["mask_rate"],
        )

        print(
            f"Epoch {epoch + 1}/{config['epochs']}: "
            f"loss={loss:.4f}, masked-atom accuracy={accuracy:.3f}"
        )

        if loss < best_loss:
            best_loss = loss
            torch.save(encoder.state_dict(), PRETRAINED_CKPT)

    print("Saved pretrained encoder:", PRETRAINED_CKPT)
