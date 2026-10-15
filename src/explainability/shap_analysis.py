
import numpy as np
import pandas as pd
import torch

from datasets.bbbp_dataset import BBBDataset, load_split_indices
from datasets.collate import collate_fn
from models.bbbp_model import BBBModel
from preprocessing.compute_descriptors import DESCRIPTOR_COLUMNS
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
    load_config,
    require,
)


def run_shap():
    import shap
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    config = load_config("explainability")
    device = torch.device("cpu")

    require(FINETUNED_CKPT, hint="python main.py train")
    checkpoint = load_checkpoint(FINETUNED_CKPT, device)

    desc_mean = np.array(checkpoint["desc_mean"], dtype=np.float32)
    desc_std = np.array(checkpoint["desc_std"], dtype=np.float32)

    geom_mean = checkpoint.get("geom_mean")
    geom_std = checkpoint.get("geom_std")

    if geom_mean is not None:
        geom_mean = np.array(geom_mean, dtype=np.float32)
    if geom_std is not None:
        geom_std = np.array(geom_std, dtype=np.float32)

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

    train_idx, _, test_idx = load_split_indices(
        checkpoint.get("split", "scaffold")
    )

    rng = np.random.RandomState(0)
    background_size = min(
        len(train_idx),
        config.get("shap_background", 20),
    )
    background_idx = rng.choice(
        train_idx,
        size=background_size,
        replace=False,
    )
    background = dataset.descriptors[background_idx]

    num_molecules = min(
        len(test_idx),
        config.get("num_molecules", 20),
    )
    sample_count = config.get("shap_nsamples", 100)

    all_values = []

    print(f"Explaining {num_molecules} test molecules")

    for idx in test_idx[:num_molecules]:
        batch = collate_fn([dataset[idx]])

        with torch.no_grad():
            graph_embedding = (
                model.graph_encoder(batch["graph"])
                if model.use_2d else None
            )
            line_embedding = (
                model.line_encoder(batch["line_graph"])
                if model.use_line else None
            )

            distance_graphs = batch.get(
                "distance_graphs",
                batch["distance_graph"],
            )

            if isinstance(distance_graphs, (list, tuple)):
                scale_embeddings = [
                    model.distance_encoder(graph)
                    for graph in distance_graphs
                ]
                distance_embedding = model.cutoff_attention(
                    scale_embeddings
                )
            else:
                distance_embedding = model.distance_encoder(
                    distance_graphs
                )

            geometry_embedding = None

            if model.use_geom and batch.get("geometry") is not None:
                geometry_embedding = model.geometry_encoder(
                    batch["geometry"]
                )

        def predict(descriptors):
            descriptor_tensor = torch.tensor(
                descriptors,
                dtype=torch.float32,
            )
            count = descriptor_tensor.shape[0]

            with torch.no_grad():
                descriptor_embedding = model.descriptor_encoder(
                    descriptor_tensor
                )

                graph_part = (
                    graph_embedding.expand(count, -1)
                    if graph_embedding is not None else None
                )
                line_part = (
                    line_embedding.expand(count, -1)
                    if line_embedding is not None else None
                )
                distance_part = (
                    distance_embedding.expand(count, -1)
                    if distance_embedding is not None else None
                )
                geometry_part = (
                    geometry_embedding.expand(count, -1)
                    if geometry_embedding is not None else None
                )

                combined = model.fusion(
                    h2d=graph_part,
                    hline=line_part,
                    h3d=distance_part,
                    hdesc=descriptor_embedding,
                    hgeom=geometry_part,
                )

                logits = model.classifier(model.adapter(combined))
                return torch.sigmoid(logits).view(-1).numpy()

        explainer = shap.KernelExplainer(predict, background)
        values = explainer.shap_values(
            dataset.descriptors[idx:idx + 1],
            nsamples=sample_count,
        )

        if isinstance(values, list):
            values = values[0]

        all_values.append(
            np.asarray(values).reshape(-1, len(DESCRIPTOR_COLUMNS))[0]
        )

    importance = np.abs(np.asarray(all_values)).mean(axis=0)

    results = pd.DataFrame({
        "descriptor": DESCRIPTOR_COLUMNS,
        "mean_abs_shap": importance,
    }).sort_values("mean_abs_shap", ascending=False)

    output_dir = RESULTS_DIR / "shap"
    output_dir.mkdir(parents=True, exist_ok=True)

    results.to_csv(
        output_dir / "descriptor_importance.csv",
        index=False,
    )

    plt.figure(figsize=(7, 4.5))
    plt.barh(
        results["descriptor"][::-1],
        results["mean_abs_shap"][::-1],
    )
    plt.xlabel("Mean absolute SHAP value")
    plt.title("Descriptor Importance")
    plt.tight_layout()
    plt.savefig(output_dir / "descriptor_importance.png", dpi=200)
    plt.close()

    print(results.to_string(index=False))
    print(f"SHAP results saved to {output_dir}")
