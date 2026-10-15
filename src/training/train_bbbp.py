
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from torch.utils.data import DataLoader, Subset

from datasets.bbbp_dataset import (
    BBBDataset,
    compute_descriptor_stats,
    compute_geometry_stats,
    load_split_indices,
)
from datasets.collate import collate_fn
from models.bbbp_model import BBBModel
from training.losses import get_loss
from training.trainer import Trainer
from evaluation.evaluate import evaluate
from evaluation.metrics import find_optimal_precision_threshold
from utils.checkpoint import save_checkpoint, load_checkpoint
from utils.logger import get_logger
from utils.paths import (
    MOL_GRAPHS, LINE_GRAPHS, DIST_GRAPHS, DESCRIPTORS,
    LABELS, GEOM_FEATURES, PRETRAINED_CKPT, FINETUNED_CKPT,
    RESULTS_DIR, load_config,
)
from utils.seed import set_seed
from utils.visualization import plot_training_curve


def load_pretrained_encoder(model, device):
    if not PRETRAINED_CKPT.exists() or PRETRAINED_CKPT.stat().st_size == 0:
        print("No pretrained weights found; training from scratch.")
        return

    model.graph_encoder.load_state_dict(
        torch.load(PRETRAINED_CKPT, map_location=device, weights_only=True)
    )
    print("Loaded pretrained graph encoder.")


def get_probabilities(model, loader, device):
    model.eval()
    probabilities = []
    labels = []

    with torch.no_grad():
        for batch in loader:
            distances = batch["distance_graphs"]

            if isinstance(distances, (list, tuple)):
                distances = [graph.to(device) for graph in distances]
            else:
                distances = distances.to(device)

            geometry = batch["geometry"]
            if geometry is not None:
                geometry = geometry.to(device)

            output = model(
                graph_data=batch["graph"].to(device),
                line_data=batch["line_graph"].to(device),
                distance_data=distances,
                descriptors=batch["descriptor"].to(device),
                geometry_data=geometry,
            )

            probabilities.extend(torch.sigmoid(output).view(-1).cpu().tolist())
            labels.extend(batch["label"].view(-1).cpu().tolist())

    return probabilities, labels


def train(
    split=None,
    epochs=None,
    use_pretrained=None,
    seed=None,
    ablation_mode="full",
    save_ckpt_path=None,
    model_type=None,
    edge_budget_ratio=None,
    min_recall=None,
    preserve_bonds=True,
):
    config = load_config("finetune")

    split = split or config.get("split", "scaffold")
    epochs = epochs or config.get("epochs", 50)
    seed = config.get("seed", 42) if seed is None else seed
    use_pretrained = config.get("use_pretrained", True) if use_pretrained is None else use_pretrained
    model_type = model_type or config.get("model_type", "adaptive")
    edge_budget_ratio = config.get("edge_budget_ratio", 0.5) if edge_budget_ratio is None else edge_budget_ratio
    min_recall = config.get("min_recall", 0.70) if min_recall is None else min_recall

    use_adaptive = model_type == "adaptive" or ablation_mode.startswith("adaptive")
    checkpoint_path = Path(save_ckpt_path) if save_ckpt_path else FINETUNED_CKPT

    logger = get_logger()
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info(
        f"device={device}, split={split}, epochs={epochs}, seed={seed}, "
        f"model={model_type}, edge_budget={edge_budget_ratio}"
    )

    train_idx, val_idx, test_idx = load_split_indices(split)

    desc_mean, desc_std = compute_descriptor_stats(DESCRIPTORS, train_idx)
    geom_mean, geom_std = compute_geometry_stats(GEOM_FEATURES, train_idx)

    dataset = BBBDataset(
        MOL_GRAPHS,
        LINE_GRAPHS,
        DIST_GRAPHS,
        DESCRIPTORS,
        LABELS,
        geometry_path=GEOM_FEATURES if GEOM_FEATURES.exists() else None,
        desc_mean=desc_mean,
        desc_std=desc_std,
        geom_mean=geom_mean,
        geom_std=geom_std,
    )

    batch_size = config.get("batch_size", 32)

    train_loader = DataLoader(
        Subset(dataset, train_idx),
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        Subset(dataset, val_idx),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )
    test_loader = DataLoader(
        Subset(dataset, test_idx),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )

    model = BBBModel(
        ablation_mode=ablation_mode,
        use_adaptive_spatial=use_adaptive,
        edge_budget_ratio=edge_budget_ratio,
        preserve_bonds=preserve_bonds,
    ).to(device)

    if use_pretrained:
        load_pretrained_encoder(model, device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.get("lr", 0.001))
    trainer = Trainer(model, optimizer, get_loss(), device)

    history = {"train_loss": [], "val_auc": []}
    best_auc = -1.0

    for epoch in range(epochs):
        train_loss = trainer.train_epoch(train_loader)
        val_metrics = evaluate(model, val_loader, device, threshold=0.5)
        val_auc = float(np.nan_to_num(val_metrics["roc_auc"], nan=0.0))

        history["train_loss"].append(train_loss)
        history["val_auc"].append(val_auc)

        logger.info(
            f"Epoch {epoch + 1}/{epochs}: loss={train_loss:.4f}, "
            f"val_auc={val_auc:.4f}, "
            f"precision={val_metrics['precision']:.4f}, "
            f"recall={val_metrics['recall']:.4f}"
        )

        if val_auc > best_auc:
            best_auc = val_auc

            save_checkpoint({
                "model_state": model.state_dict(),
                "desc_mean": desc_mean.tolist(),
                "desc_std": desc_std.tolist(),
                "geom_mean": geom_mean.tolist() if geom_mean is not None else None,
                "geom_std": geom_std.tolist() if geom_std is not None else None,
                "split": split,
                "epoch": epoch + 1,
                "val_auc": val_auc,
                "ablation_mode": ablation_mode,
                "model_type": model_type,
                "edge_budget_ratio": edge_budget_ratio,
                "preserve_bonds": preserve_bonds,
            }, checkpoint_path)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint was not saved: {checkpoint_path}")

    checkpoint = load_checkpoint(checkpoint_path, device)
    model.load_state_dict(checkpoint["model_state"])

    val_probs, val_labels = get_probabilities(model, val_loader, device)
    threshold_info = find_optimal_precision_threshold(
        val_labels,
        val_probs,
        min_recall=min_recall,
    )

    threshold = threshold_info["threshold"]
    test_metrics = evaluate(model, test_loader, device, threshold=threshold)

    test_metrics.update({
        "val_selected_threshold": float(threshold),
        "min_recall_required": float(min_recall),
        "min_recall_met": threshold_info["min_recall_met"],
        "model_type": model_type,
        "edge_budget_ratio": float(edge_budget_ratio),
    })

    logger.info(
        f"Test results: precision={test_metrics['precision']:.4f}, "
        f"recall={test_metrics['recall']:.4f}, "
        f"ROC-AUC={test_metrics['roc_auc']:.4f}"
    )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    prefix = f"test_metrics_{split}_{model_type}"

    if ablation_mode != "full":
        prefix += f"_{ablation_mode}"

    pd.DataFrame([{
        key: value
        for key, value in test_metrics.items()
        if not isinstance(value, (dict, list))
    }]).to_csv(RESULTS_DIR / f"{prefix}.csv", index=False)

    plot_training_curve(history, RESULTS_DIR / f"training_curve_{split}.png")

    return test_metrics
