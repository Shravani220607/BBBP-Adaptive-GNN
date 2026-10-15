"""Multi-seed evaluation runner for BBBP Multi-View GNN.
Evaluates model robustness and statistical significance across 5 random seeds.
Reports:
- ROC-AUC mean +- std
- PR-AUC mean +- std
- Accuracy mean +- std
- F1 mean +- std
Exports both detailed and summary CSV files.
"""

from pathlib import Path
import numpy as np
import pandas as pd

from training.train_bbbp import train
from utils.paths import CKPT_DIR, RESULTS_DIR, load_config

DEFAULT_SEEDS = [42, 101, 202, 303, 404]


def run_multi_seed_evaluation(seeds=None, split="scaffold", epochs=None):
    if seeds is None:
        seeds = DEFAULT_SEEDS

    cfg = load_config("finetune")
    split = split or cfg.get("split", "scaffold")
    epochs = epochs or cfg.get("epochs", 30)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    seeds_ckpt_dir = CKPT_DIR / "multi_seed"
    seeds_ckpt_dir.mkdir(parents=True, exist_ok=True)

    print("\n=======================================================")
    print("STARTING MULTI-SEED EVALUATION (5 RANDOM SEEDS)")
    print(f"Seeds: {seeds} | Split: {split} | Epochs per seed: {epochs}")
    print("=======================================================\n")

    seed_results = []

    for seed in seeds:
        print(f"\n---> Training with Seed = {seed}")
        ckpt_path = seeds_ckpt_dir / f"seed_{seed}_model.pt"

        test_metrics = train(
            split=split,
            epochs=epochs,
            seed=seed,
            ablation_mode="full",
            save_ckpt_path=ckpt_path,
        )

        res = {
            "Seed": seed,
            "ROC-AUC": float(test_metrics.get("roc_auc", 0.0)),
            "PR-AUC": float(test_metrics.get("pr_auc", 0.0)),
            "Accuracy": float(test_metrics.get("accuracy", 0.0)),
            "F1": float(test_metrics.get("f1", 0.0)),
        }
        seed_results.append(res)
        print(f"Seed {seed} Result: ROC-AUC={res['ROC-AUC']:.4f}, PR-AUC={res['PR-AUC']:.4f}, Acc={res['Accuracy']:.4f}, F1={res['F1']:.4f}")

    df_seeds = pd.DataFrame(seed_results)
    raw_csv = RESULTS_DIR / "multi_seed_results.csv"
    df_seeds.to_csv(raw_csv, index=False)

    # Compute statistical mean and standard deviation
    metrics_list = ["ROC-AUC", "PR-AUC", "Accuracy", "F1"]
    summary_rows = []

    print("\n=======================================================")
    print("MULTI-SEED STATISTICAL SUMMARY (Mean +- Std):")
    print("=======================================================")

    for m in metrics_list:
        vals = df_seeds[m].values
        mean_val = float(np.mean(vals))
        std_val = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
        fmt = f"{mean_val:.4f} ± {std_val:.4f}"
        print(f"  {m:12s}: {fmt}")
        summary_rows.append({
            "Metric": m,
            "Mean": round(mean_val, 4),
            "Std": round(std_val, 4),
            "Formatted": fmt,
        })

    print("=======================================================")

    df_summary = pd.DataFrame(summary_rows)
    summary_csv = RESULTS_DIR / "multi_seed_summary.csv"
    df_summary.to_csv(summary_csv, index=False)

    print(f"Saved seed run details to: {raw_csv}")
    print(f"Saved summary statistics to: {summary_csv}\n")

    return df_seeds, df_summary
