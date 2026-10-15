
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem.Draw import SimilarityMaps


def save_atom_importance_image(smiles, weights, path):
    molecule = Chem.MolFromSmiles(smiles)

    if molecule is None:
        return

    try:
        figure = SimilarityMaps.GetSimilarityMapFromWeights(
            molecule,
            [float(weight) for weight in weights],
            size=(400, 400),
        )
        figure.savefig(path, bbox_inches="tight", dpi=150)
        plt.close(figure)
    except Exception:
        important_atoms = [
            idx for idx, weight in enumerate(weights)
            if weight >= 0.5
        ]
        Draw.MolToFile(
            molecule,
            str(path),
            size=(400, 400),
            highlightAtoms=important_atoms,
        )


def visualize_view_attention(model, dataloader, device, save_dir):
    model.eval()
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    attention_values = []
    view_names = None

    with torch.no_grad():
        for batch in dataloader:
            distance_graphs = batch.get(
                "distance_graphs",
                batch["distance_graph"],
            )

            if isinstance(distance_graphs, (list, tuple)):
                distance_graphs = [
                    graph.to(device) for graph in distance_graphs
                ]
            else:
                distance_graphs = distance_graphs.to(device)

            model(
                graph_data=batch["graph"].to(device),
                line_data=batch["line_graph"].to(device),
                distance_data=distance_graphs,
                descriptors=batch["descriptor"].to(device),
                geometry_data=(
                    batch["geometry"].to(device)
                    if batch.get("geometry") is not None else None
                ),
            )

            weights = model.fusion.get_last_weights()

            if weights is not None:
                attention_values.append(weights)

                if view_names is None:
                    view_names = model.fusion.last_view_names

    if not attention_values:
        print("No view attention weights were collected.")
        return None

    weights = np.concatenate(attention_values, axis=0)
    mean_weights = weights.mean(axis=0)
    std_weights = weights.std(axis=0)

    if view_names is None or len(view_names) != len(mean_weights):
        view_names = [
            f"View {idx + 1}" for idx in range(len(mean_weights))
        ]

    results = pd.DataFrame({
        "View": view_names,
        "Mean_Attention": mean_weights,
        "Std_Attention": std_weights,
    })

    results.to_csv(save_dir / "view_attention.csv", index=False)

    plt.figure(figsize=(7, 4.5))
    plt.bar(
        results["View"],
        results["Mean_Attention"],
        yerr=results["Std_Attention"],
        capsize=5,
    )
    plt.ylabel("Mean attention weight")
    plt.title("Multi-View Attention")
    plt.tight_layout()
    plt.savefig(save_dir / "view_attention.png", dpi=200)
    plt.close()

    print(f"View attention results saved to {save_dir}")
    return results


def visualize_cutoff_attention(model, dataloader, device, save_dir):
    model.eval()
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    if not hasattr(model, "cutoff_attention"):
        print("This model has no cutoff attention layer.")
        return None

    if model.cutoff_attention is None:
        print("This model has no cutoff attention layer.")
        return None

    all_weights = []

    with torch.no_grad():
        for batch in dataloader:
            distance_graphs = batch.get(
                "distance_graphs",
                batch["distance_graph"],
            )

            if not isinstance(distance_graphs, (list, tuple)):
                continue

            distance_graphs = [
                graph.to(device) for graph in distance_graphs
            ]

            model(
                graph_data=batch["graph"].to(device),
                line_data=batch["line_graph"].to(device),
                distance_data=distance_graphs,
                descriptors=batch["descriptor"].to(device),
                geometry_data=(
                    batch["geometry"].to(device)
                    if batch.get("geometry") is not None else None
                ),
            )

            weights = model.cutoff_attention.get_last_weights()

            if weights is not None:
                all_weights.append(weights)

    if not all_weights:
        print("No cutoff attention weights were collected.")
        return None

    weights = np.concatenate(all_weights, axis=0)
    mean_weights = weights.mean(axis=0)
    std_weights = weights.std(axis=0)

    cutoffs = ["2 Å", "3 Å", "4 Å", "5 Å"][:len(mean_weights)]

    results = pd.DataFrame({
        "Cutoff": cutoffs,
        "Mean_Attention": mean_weights,
        "Std_Attention": std_weights,
    })

    results.to_csv(save_dir / "cutoff_attention.csv", index=False)

    plt.figure(figsize=(6, 4))
    plt.bar(
        results["Cutoff"],
        results["Mean_Attention"],
        yerr=results["Std_Attention"],
        capsize=5,
    )
    plt.xlabel("Distance cutoff")
    plt.ylabel("Mean attention weight")
    plt.title("Distance Cutoff Attention")
    plt.tight_layout()
    plt.savefig(save_dir / "cutoff_attention.png", dpi=200)
    plt.close()

    print(f"Cutoff attention results saved to {save_dir}")
    return results


def run_attention_explanations():
    from torch.utils.data import DataLoader, Subset

    from datasets.bbbp_dataset import BBBDataset, load_split_indices
    from datasets.collate import collate_fn
    from models.bbbp_model import BBBModel
    from utils.checkpoint import load_checkpoint
    from utils.paths import (
        MOL_GRAPHS,
        LINE_GRAPHS,
        DIST_GRAPHS,
        DESCRIPTORS,
        LABELS,
        GEOM_FEATURES,
        FINETUNED_CKPT,
        RESULTS_DIR,
        require,
    )

    require(FINETUNED_CKPT, hint="python main.py train")

    device = torch.device("cpu")
    checkpoint = load_checkpoint(FINETUNED_CKPT, device)

    model_type = checkpoint.get("model_type", "adaptive")
    ablation_mode = checkpoint.get("ablation_mode", "full")

    model = BBBModel(
        ablation_mode=ablation_mode,
        use_adaptive_spatial=(
            model_type == "adaptive"
            or ablation_mode.startswith("adaptive")
        ),
        edge_budget_ratio=checkpoint.get("edge_budget_ratio", 0.5),
        preserve_bonds=checkpoint.get("preserve_bonds", True),
    ).to(device)

    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    geometry_path = GEOM_FEATURES if GEOM_FEATURES.exists() else None

    geom_mean = checkpoint.get("geom_mean")
    geom_std = checkpoint.get("geom_std")

    dataset = BBBDataset(
        MOL_GRAPHS,
        LINE_GRAPHS,
        DIST_GRAPHS,
        DESCRIPTORS,
        LABELS,
        geometry_path=geometry_path,
        desc_mean=np.array(checkpoint["desc_mean"], dtype=np.float32),
        desc_std=np.array(checkpoint["desc_std"], dtype=np.float32),
        geom_mean=(
            np.array(geom_mean, dtype=np.float32)
            if geom_mean is not None else None
        ),
        geom_std=(
            np.array(geom_std, dtype=np.float32)
            if geom_std is not None else None
        ),
    )

    _, _, test_indices = load_split_indices(
        checkpoint.get("split", "scaffold")
    )

    test_dataset = Subset(dataset, test_indices)
    loader = DataLoader(
        test_dataset,
        batch_size=32,
        shuffle=False,
        collate_fn=collate_fn,
    )

    output_dir = RESULTS_DIR / "explainability"

    visualize_view_attention(model, loader, device, output_dir)
    visualize_cutoff_attention(model, loader, device, output_dir)
