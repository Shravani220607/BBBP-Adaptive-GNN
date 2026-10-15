
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from datasets.bbbp_dataset import BBBDataset, load_split_indices
from datasets.collate import collate_fn
from explainability.visualize_explanations import save_atom_importance_image
from models.bbbp_model import BBBModel
from utils.checkpoint import load_checkpoint
from utils.paths import (
    MOL_GRAPHS,
    LINE_GRAPHS,
    DIST_GRAPHS,
    DESCRIPTORS,
    LABELS,
    CLEANED_B3DB,
    FINETUNED_CKPT,
    RESULTS_DIR,
    load_config,
    require,
)


def explain_atoms(
    model,
    batch,
    epochs=200,
    lr=0.05,
    size_coef=0.01,
    entropy_coef=0.01,
):
    graph = batch["graph"]

    with torch.no_grad():
        output = model(
            graph_data=graph,
            line_data=batch["line_graph"],
            distance_data=batch["distance_graph"],
            descriptors=batch["descriptor"],
            geometry_data=batch.get("geometry"),
        )
        target = (torch.sigmoid(output) > 0.5).float().view(-1)

    mask_logits = torch.nn.Parameter(
        torch.ones(graph.x.size(0), 1, device=graph.x.device)
    )
    optimizer = torch.optim.Adam([mask_logits], lr=lr)

    for _ in range(epochs):
        mask = torch.sigmoid(mask_logits)

        masked_graph = graph.clone()
        masked_graph.x = graph.x * mask

        output = model(
            graph_data=masked_graph,
            line_data=batch["line_graph"],
            distance_data=batch["distance_graph"],
            descriptors=batch["descriptor"],
            geometry_data=batch.get("geometry"),
        )

        prediction_loss = F.binary_cross_entropy_with_logits(
            output.view(-1), target
        )
        size_loss = mask.mean()
        entropy_loss = -(
            mask * torch.log(mask + 1e-8)
            + (1 - mask) * torch.log(1 - mask + 1e-8)
        ).mean()

        loss = (
            prediction_loss
            + size_coef * size_loss
            + entropy_coef * entropy_loss
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return torch.sigmoid(mask_logits).detach().view(-1).cpu().numpy()


def run_gnn_explainer():
    config = load_config("explainability")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    require(FINETUNED_CKPT, hint="python main.py train")
    checkpoint = load_checkpoint(FINETUNED_CKPT, device)

    model = BBBModel(
        ablation_mode=checkpoint.get("ablation_mode", "full"),
        use_adaptive_spatial=(
            checkpoint.get("model_type", "adaptive") == "adaptive"
            or checkpoint.get("ablation_mode", "").startswith("adaptive")
        ),
        edge_budget_ratio=checkpoint.get("edge_budget_ratio", 0.5),
        preserve_bonds=checkpoint.get("preserve_bonds", True),
    ).to(device)

    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    for parameter in model.parameters():
        parameter.requires_grad_(False)

    dataset = BBBDataset(
        MOL_GRAPHS,
        LINE_GRAPHS,
        DIST_GRAPHS,
        DESCRIPTORS,
        LABELS,
        geometry_path=None,
        desc_mean=np.array(checkpoint["desc_mean"], dtype=np.float32),
        desc_std=np.array(checkpoint["desc_std"], dtype=np.float32),
    )

    smiles = pd.read_csv(CLEANED_B3DB)["smiles"].tolist()
    assert len(smiles) == len(dataset), (
        "Dataset sizes differ. Run `python main.py verify`."
    )

    _, _, test_indices = load_split_indices(
        checkpoint.get("split", "scaffold")
    )

    output_dir = RESULTS_DIR / "gnn_explainer"
    output_dir.mkdir(parents=True, exist_ok=True)

    count = config.get("num_molecules", 20)

    for idx in test_indices[:count]:
        batch = collate_fn([dataset[idx]])
        batch = {
            key: value.to(device) if value is not None else None
            for key, value in batch.items()
        }

        weights = explain_atoms(
            model,
            batch,
            epochs=config.get("explainer_epochs", 200),
            lr=config.get("explainer_lr", 0.05),
            size_coef=config.get("size_coef", 0.01),
            entropy_coef=config.get("entropy_coef", 0.01),
        )

        assert dataset.graphs[idx].num_nodes == len(weights)

        save_atom_importance_image(
            smiles[idx],
            weights,
            output_dir / f"mol_{idx}.png",
        )

        important_atoms = np.argsort(-weights)[:3].tolist()
        print(f"Molecule {idx}: top atoms {important_atoms}")

    print(f"Explanation images saved to {output_dir}")
