import time
import tracemalloc
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from evaluation.metrics import (
    compute_metrics,
    compute_comprehensive_metrics,
    find_optimal_precision_threshold,
)


def evaluate(model, dataloader, device, threshold=0.5):
    """Evaluates model performance measuring latency, peak memory, edge counts, and precision-first metrics."""
    model.eval()
    y_true, y_prob = [], []
    total_candidates, total_retained, edge_stat_batches = 0, 0, 0
    total_selection_time_ms = 0.0

    tracemalloc.start()
    t_start = time.perf_counter()

    with torch.no_grad():
        for batch in dataloader:
            graph = batch["graph"].to(device)
            line_graph = batch["line_graph"].to(device)

            dist_item = batch.get("distance_graphs", batch.get("distance_graph"))
            if isinstance(dist_item, (list, tuple)):
                distance_graph = [g.to(device) for g in dist_item]
            else:
                distance_graph = dist_item.to(device)

            descriptors = batch["descriptor"].to(device)
            geometry = (
                batch["geometry"].to(device)
                if batch.get("geometry") is not None
                else None
            )
            labels = batch["label"].to(device)

            outputs = model(
                graph_data=graph,
                line_data=line_graph,
                distance_data=distance_graph,
                descriptors=descriptors,
                geometry_data=geometry,
            )

            probs = torch.sigmoid(outputs).view(-1)
            y_true.extend(labels.view(-1).cpu().tolist())
            y_prob.extend(probs.cpu().tolist())

            edge_stats = getattr(model, "last_edge_stats", {})
            if edge_stats and "candidates" in edge_stats:
                total_candidates += edge_stats.get("candidates", 0)
                total_retained += edge_stats.get("retained", 0)
                total_selection_time_ms += edge_stats.get("time_ms", 0.0)
                edge_stat_batches += 1

    t_end = time.perf_counter()
    _, peak_mem_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    n_samples = max(1, len(y_true))
    total_eval_time_ms = (t_end - t_start) * 1000.0
    latency_per_sample_ms = total_eval_time_ms / n_samples

    timing_stats = {
        "inference_latency_ms": float(latency_per_sample_ms),
        "total_inference_time_ms": float(total_eval_time_ms),
        "peak_memory_mb": float(peak_mem_bytes / (1024 * 1024)),
    }

    edge_summary = {}
    if edge_stat_batches > 0:
        edge_summary = {
            "avg_candidate_edges": float(total_candidates / edge_stat_batches),
            "avg_retained_edges": float(total_retained / edge_stat_batches),
            "spatial_sparsity_ratio": float(1.0 - (total_retained / max(1, total_candidates))),
            "edge_scoring_latency_ms": float(total_selection_time_ms / edge_stat_batches),
        }

    return compute_comprehensive_metrics(
        y_true, y_prob, threshold=threshold, timing_stats=timing_stats, edge_stats=edge_summary
    )


def evaluate_checkpoint(split=None, checkpoint_path=None, min_recall=0.70, threshold=None):
    """Evaluates checkpoint with validation threshold tuning and test evaluation."""
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
        require,
    )

    ckpt_file = checkpoint_path or FINETUNED_CKPT
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    require(ckpt_file, hint="python main.py train")
    ckpt = load_checkpoint(ckpt_file, device)

    split = split or ckpt.get("split", "scaffold")
    ablation_mode = ckpt.get("ablation_mode", "full")
    model_type = ckpt.get("model_type", "adaptive")
    use_adaptive = (model_type == "adaptive" or ablation_mode.startswith("adaptive"))

    model = BBBModel(
        ablation_mode=ablation_mode,
        use_adaptive_spatial=use_adaptive,
        edge_budget_ratio=ckpt.get("edge_budget_ratio", 0.5),
        preserve_bonds=ckpt.get("preserve_bonds", True),
    ).to(device)
    model.load_state_dict(ckpt["model_state"])

    geom_mean = np.array(ckpt["geom_mean"], dtype=np.float32) if ckpt.get("geom_mean") is not None else None
    geom_std = np.array(ckpt["geom_std"], dtype=np.float32) if ckpt.get("geom_std") is not None else None

    dataset = BBBDataset(
        MOL_GRAPHS,
        LINE_GRAPHS,
        DIST_GRAPHS,
        DESCRIPTORS,
        LABELS,
        geometry_path=GEOM_FEATURES if GEOM_FEATURES.exists() else None,
        desc_mean=np.array(ckpt["desc_mean"], dtype=np.float32),
        desc_std=np.array(ckpt["desc_std"], dtype=np.float32),
        geom_mean=geom_mean,
        geom_std=geom_std,
    )
    _, val_idx, test_idx = load_split_indices(split)

    # 1. Validation-based threshold tuning (No leakage onto test set)
    val_loader = DataLoader(Subset(dataset, val_idx), batch_size=64, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(Subset(dataset, test_idx), batch_size=64, shuffle=False, collate_fn=collate_fn)

    tuned_threshold = threshold
    if tuned_threshold is None:
        val_eval = evaluate(model, val_loader, device, threshold=0.5)
        # Extract validation probabilities
        val_probs, val_trues = [], []
        model.eval()
        with torch.no_grad():
            for b in val_loader:
                out = model(
                    b["graph"].to(device),
                    b["line_graph"].to(device),
                    [g.to(device) for g in b["distance_graphs"]] if isinstance(b["distance_graphs"], list) else b["distance_graphs"].to(device),
                    b["descriptor"].to(device),
                    b["geometry"].to(device) if b.get("geometry") is not None else None,
                )
                val_probs.extend(torch.sigmoid(out).view(-1).cpu().tolist())
                val_trues.extend(b["label"].view(-1).cpu().tolist())
        thresh_info = find_optimal_precision_threshold(val_trues, val_probs, min_recall=min_recall)
        tuned_threshold = thresh_info["threshold"]
        print(f"Validation threshold selection (target min_recall={min_recall}):")
        print(f"  Selected threshold : {tuned_threshold:.4f}")
        print(f"  Val precision      : {thresh_info['val_precision']:.4f}")
        print(f"  Val recall         : {thresh_info['val_recall']:.4f}")
        print(f"  Min recall achieved: {thresh_info['min_recall_met']}")

    # 2. Final evaluation on held-out test split
    metrics = evaluate(model, test_loader, device, threshold=tuned_threshold)
    print(f"\n==========================================")
    print(f"Test Set Evaluation ({split} split, model={model_type}):")
    print(f"==========================================")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {k:25s}: {v:.4f}")
        else:
            print(f"  {k:25s}: {v}")
    print(f"==========================================\n")
    return metrics
