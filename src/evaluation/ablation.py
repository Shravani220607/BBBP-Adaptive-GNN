
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from training.train_bbbp import train
from utils.paths import CKPT_DIR, RESULTS_DIR, load_config


def run_ablation_studies(split="scaffold", epochs=None, specific_experiment=None):
    config = load_config("ablation")
    split = split or config.get("split", "scaffold")
    epochs = epochs or config.get("epochs", 20)
    experiments = config.get("experiments", [])

    if specific_experiment:
        experiments = [
            exp for exp in experiments
            if exp["name"] == specific_experiment
        ]

        if not experiments:
            raise ValueError(f"Unknown experiment: {specific_experiment}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = CKPT_DIR / "ablation"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    results = []

    for exp in experiments:
        name = exp["name"]
        label = exp.get("label", name)
        model_type = exp.get("model_type", "adaptive")
        mode = exp.get("ablation_mode", "adaptive")
        budget = exp.get("edge_budget_ratio", 0.5)
        preserve_bonds = exp.get("preserve_bonds", True)

        print(f"Running experiment: {label}")

        metrics = train(
            split=split,
            epochs=epochs,
            use_pretrained=False,
            ablation_mode=mode,
            save_ckpt_path=checkpoint_dir / f"{name}_best.pt",
            model_type=model_type,
            edge_budget_ratio=budget,
            preserve_bonds=preserve_bonds,
            min_recall=config.get("min_recall", 0.70),
        )

        results.append({
            "Experiment": label,
            "Model Type": model_type,
            "Edge Budget": budget,
            "Preserve Bonds": preserve_bonds,
            "Threshold": metrics.get("threshold", 0.5),
            "Precision": metrics.get("precision", 0.0),
            "Recall": metrics.get("recall", 0.0),
            "F1": metrics.get("f1", 0.0),
            "PR-AUC": metrics.get("pr_auc", 0.0),
            "ROC-AUC": metrics.get("roc_auc", float("nan")),
            "Specificity": metrics.get("specificity", 0.0),
            "Predicted Positives": metrics.get("predicted_positives", 0),
            "False Positives": metrics.get("false_positives", 0),
            "Inference Latency (ms)": metrics.get("inference_latency_ms", 0.0),
            "Spatial Sparsity": metrics.get("spatial_sparsity_ratio", 0.0),
            "Avg Retained Edges": metrics.get("avg_retained_edges", 0.0),
            "Accuracy": metrics.get("accuracy", 0.0),
        })

    results_df = pd.DataFrame(results)
    csv_path = RESULTS_DIR / "ablation_results.csv"
    results_df.to_csv(csv_path, index=False)

    print(results_df.to_string(index=False))
    print(f"Results saved to: {csv_path}")

    if not results_df.empty:
        columns = ["ROC-AUC", "PR-AUC", "Accuracy", "F1"]
        ax = results_df.set_index("Experiment")[columns].plot(
            kind="bar",
            figsize=(10, 5),
        )
        ax.set_ylabel("Score")
        ax.set_ylim(0, 1.05)
        ax.tick_params(axis="x", labelrotation=20)
        plt.tight_layout()

        plot_path = RESULTS_DIR / "ablation_comparison.png"
        plt.savefig(plot_path, dpi=200)
        plt.close()

        print(f"Plot saved to: {plot_path}")

    return results_df
