
import argparse
import subprocess
import sys
from pathlib import Path


# Project folders
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))


def main():
    parser = argparse.ArgumentParser(
        description="BBBP Blood-Brain Barrier Prediction"
    )

    commands = parser.add_subparsers(dest="command", required=True)

    p = commands.add_parser("predict")
    p.add_argument("--smiles", required=True)
    p.add_argument(
        "--model",
        choices=["adaptive", "baseline"],
        default="adaptive",
    )
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--checkpoint", type=str, default=None)

    # Preprocess the dataset
    p = commands.add_parser("preprocess")
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument(
        "--stage",
        choices=[
            "all", "clean", "conformers", "descriptors",
            "graphs", "splits", "verify"
        ],
        default="all",
    )

    # Check data and model shapes
    p = commands.add_parser("verify")
    p.add_argument(
        "--what",
        choices=["data", "shapes", "all"],
        default="all",
    )

    # ChEMBL pretraining
    commands.add_parser("preprocess-chembl")
    commands.add_parser("pretrain")

    # Train a model
    p = commands.add_parser("train")
    p.add_argument("--split", choices=["scaffold", "random"], default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--no-pretrained", action="store_true")
    p.add_argument(
        "--model-type",
        choices=["adaptive", "baseline"],
        default=None,
    )
    p.add_argument("--edge-budget", type=float, default=None)
    p.add_argument("--min-recall", type=float, default=None)
    p.add_argument("--no-preserve-bonds", action="store_true")
    p.add_argument(
        "--ablation",
        choices=[
            "full", "2d_only", "2d_3d", "2d_3d_line",
            "no_cutoff_attn", "no_geometry"
        ],
        default="full",
    )

    # Evaluate a trained model
    p = commands.add_parser("evaluate")
    p.add_argument("--split", choices=["scaffold", "random"], default=None)
    p.add_argument("--checkpoint", type=str, default=None)
    p.add_argument("--min-recall", type=float, default=0.70)
    p.add_argument("--threshold", type=float, default=None)

    # Compare baseline and adaptive models
    p = commands.add_parser("compare")
    p.add_argument("--split", choices=["scaffold", "random"], default="scaffold")
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--edge-budget", type=float, default=0.5)
    p.add_argument("--min-recall", type=float, default=0.70)

    # Explain model predictions
    p = commands.add_parser("explain")
    p.add_argument(
        "--which",
        choices=["shap", "attention", "gnn", "all"],
        default="all",
    )

    # Run ablation experiments
    p = commands.add_parser("ablation")
    p.add_argument("--split", choices=["scaffold", "random"], default="scaffold")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--experiment", type=str, default=None)

    # Test performance across multiple random seeds
    p = commands.add_parser("multiseed")
    p.add_argument("--split", choices=["scaffold", "random"], default="scaffold")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--seeds", type=int, nargs="+", default=None)

    # Launch the interactive demo
    p = commands.add_parser("demo")
    p.add_argument("--port", type=int, default=8501)

    args = parser.parse_args()

    # Preprocessing
    if args.command == "preprocess":
        from preprocessing.clean_b3db import clean_b3db
        from preprocessing.generate_conformers import generate_conformers
        from preprocessing.compute_descriptors import compute_descriptors
        from graphs.graph_utils import build_all_graphs
        from preprocessing.scaffold_split import make_splits
        from verify import verify_data

        stages = {
            "clean": clean_b3db,
            "conformers": lambda: generate_conformers(
                num_workers=args.workers,
                max_molecules=args.limit,
            ),
            "descriptors": compute_descriptors,
            "graphs": build_all_graphs,
            "splits": make_splits,
            "verify": verify_data,
        }

        for name, function in stages.items():
            if args.stage == "all" or args.stage == name:
                print(f"\n--- Running {name} ---")
                function()

    # Verification
    elif args.command == "verify":
        from verify import verify_data, shape_trace

        if args.what in ("shapes", "all"):
            shape_trace()

        if args.what in ("data", "all"):
            verify_data()

    # ChEMBL pretraining
    elif args.command == "preprocess-chembl":
        from preprocessing.clean_chembl import clean_chembl
        clean_chembl()

    elif args.command == "pretrain":
        from pretraining.train_pretrain import pretrain
        pretrain()

    
    # Model training
    elif args.command == "train":
        from training.train_bbbp import train

        train(
            split=args.split,
            epochs=args.epochs,
            seed=args.seed,
            ablation_mode=args.ablation,
            use_pretrained=False if args.no_pretrained else None,
            model_type=args.model_type,
            edge_budget_ratio=args.edge_budget,
            min_recall=args.min_recall,
            preserve_bonds=not args.no_preserve_bonds,
        )

    # Single-molecule prediction
    elif args.command == "predict":
        from src.predict import predict

        predict(
            smiles=args.smiles,
            model_type=args.model,
            threshold=args.threshold,
            checkpoint_path=args.checkpoint,
        )

    # Model evaluation
    elif args.command == "evaluate":
        from evaluation.evaluate import evaluate_checkpoint

        evaluate_checkpoint(
            split=args.split,
            checkpoint_path=args.checkpoint,
            min_recall=args.min_recall,
            threshold=args.threshold,
        )


    # --------------------------------------------------
    # Baseline vs adaptive comparison
    # --------------------------------------------------
    elif args.command == "compare":
        import pandas as pd
        from training.train_bbbp import train
        from utils.paths import CKPT_DIR, RESULTS_DIR

        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        CKPT_DIR.mkdir(parents=True, exist_ok=True)

        print("\nBaseline vs Adaptive Model")
        print(f"Split: {args.split}")
        print(f"Epochs: {args.epochs}")
        print(f"Edge budget: {args.edge_budget}")

        # Train the baseline model
        print("\nTraining baseline model...")
        baseline_path = CKPT_DIR / "baseline_best.pt"

        baseline = train(
            split=args.split,
            epochs=args.epochs,
            use_pretrained=False,
            model_type="baseline",
            ablation_mode="full",
            min_recall=args.min_recall,
            save_ckpt_path=baseline_path,
        )

        # Train the adaptive model
        print("\nTraining adaptive model...")
        adaptive_path = CKPT_DIR / "adaptive_best.pt"

        adaptive = train(
            split=args.split,
            epochs=args.epochs,
            use_pretrained=False,
            model_type="adaptive",
            ablation_mode="adaptive",
            edge_budget_ratio=args.edge_budget,
            preserve_bonds=True,
            min_recall=args.min_recall,
            save_ckpt_path=adaptive_path,
        )

        # Collect the results from both runs
        results = [
            {
                "Model": "Baseline",
                "Threshold": baseline.get("threshold"),
                "Precision": baseline.get("precision"),
                "Recall": baseline.get("recall"),
                "F1": baseline.get("f1"),
                "PR-AUC": baseline.get("pr_auc"),
                "ROC-AUC": baseline.get("roc_auc"),
                "Specificity": baseline.get("specificity"),
                "Predicted Positives": baseline.get("predicted_positives"),
                "False Positives": baseline.get("false_positives"),
                "Inference Latency (ms)": baseline.get("inference_latency_ms"),
                "Peak Memory (MB)": baseline.get("peak_memory_mb"),
                "Spatial Sparsity": 0.0,
                "Avg Retained Edges": baseline.get(
                    "avg_retained_edges", "N/A"
                ),
            },
            {
                "Model": "Adaptive",
                "Threshold": adaptive.get("threshold"),
                "Precision": adaptive.get("precision"),
                "Recall": adaptive.get("recall"),
                "F1": adaptive.get("f1"),
                "PR-AUC": adaptive.get("pr_auc"),
                "ROC-AUC": adaptive.get("roc_auc"),
                "Specificity": adaptive.get("specificity"),
                "Predicted Positives": adaptive.get("predicted_positives"),
                "False Positives": adaptive.get("false_positives"),
                "Inference Latency (ms)": adaptive.get("inference_latency_ms"),
                "Peak Memory (MB)": adaptive.get("peak_memory_mb"),
                "Spatial Sparsity": adaptive.get("spatial_sparsity_ratio"),
                "Avg Retained Edges": adaptive.get("avg_retained_edges"),
            },
        ]

        results_df = pd.DataFrame(results)
        output_file = RESULTS_DIR / "model_comparison.csv"
        results_df.to_csv(output_file, index=False)

        print("\nComparison results:")
        print(results_df.to_string(index=False))
        print(f"\nResults saved to: {output_file}")

    # Explainability
    elif args.command == "explain":
        if args.which in ("shap", "all"):
            from explainability.shap_analysis import run_shap
            run_shap()

        if args.which in ("attention", "all"):
            from explainability.visualize_explanations import (
                run_attention_explanations
            )
            run_attention_explanations()

        if args.which == "gnn":
            from explainability.gnn_explainer import run_gnn_explainer
            run_gnn_explainer()

    # Ablation experiments
    elif args.command == "ablation":
        from evaluation.ablation import run_ablation_studies

        run_ablation_studies(
            split=args.split,
            epochs=args.epochs,
            specific_experiment=args.experiment,
        )


    # Multiple-seed experiments
    elif args.command == "multiseed":
        from evaluation.multi_seed import run_multi_seed_evaluation

        run_multi_seed_evaluation(
            seeds=args.seeds,
            split=args.split,
            epochs=args.epochs,
        )


    # Interactive demo
    elif args.command == "demo":
        app_file = ROOT / "streamlit_app.py"

        print(f"Starting demo on port {args.port}...")
        subprocess.run([
            sys.executable, "-m", "streamlit", "run",
            str(app_file), "--server.port", str(args.port)
        ], check=False)


if __name__ == "__main__":
    main()
