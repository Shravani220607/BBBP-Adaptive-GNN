
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
)


def find_optimal_precision_threshold(y_true, y_prob, min_recall=0.70):
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)

    if len(y_true) == 0:
        raise ValueError("Cannot select a threshold from an empty dataset.")

    best_result = None

    for threshold in np.linspace(0.01, 0.99, 99):
        predictions = (y_prob >= threshold).astype(int)
        precision = precision_score(y_true, predictions, zero_division=0)
        recall = recall_score(y_true, predictions, zero_division=0)

        if recall >= min_recall:
            result = (precision, recall, float(threshold))

            if best_result is None or result[:2] > best_result[:2]:
                best_result = result

    recall_met = best_result is not None

    if not recall_met:
        for threshold in np.linspace(0.01, 0.99, 99):
            predictions = (y_prob >= threshold).astype(int)
            precision = precision_score(y_true, predictions, zero_division=0)
            recall = recall_score(y_true, predictions, zero_division=0)
            result = (recall, precision, float(threshold))

            if best_result is None or result > (
                best_result[1], best_result[0], best_result[2]
            ):
                best_result = (precision, recall, float(threshold))

    precision, recall, threshold = best_result

    return {
        "threshold": threshold,
        "val_precision": float(precision),
        "val_recall": float(recall),
        "min_recall_required": float(min_recall),
        "min_recall_met": recall_met,
    }


def compute_metrics(y_true, y_pred, y_prob):
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)

    if len(y_true) == 0:
        raise ValueError("Cannot calculate metrics for an empty dataset.")

    try:
        roc_auc = roc_auc_score(y_true, y_prob)
    except ValueError:
        roc_auc = float("nan")

    try:
        pr_auc = average_precision_score(y_true, y_prob)
    except ValueError:
        pr_auc = float("nan")

    tn, fp, fn, tp = confusion_matrix(
        y_true, y_pred, labels=[0, 1]
    ).ravel()

    specificity = tn / (tn + fp) if tn + fp > 0 else 0.0

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "specificity": specificity,
    }


def compute_comprehensive_metrics(
    y_true,
    y_prob,
    threshold=0.5,
    timing_stats=None,
    edge_stats=None,
):
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    predictions = (y_prob >= threshold).astype(int)

    metrics = compute_metrics(y_true, predictions, y_prob)

    tn, fp, fn, tp = confusion_matrix(
        y_true, predictions, labels=[0, 1]
    ).ravel()

    results = {
        "threshold": float(threshold),
        **metrics,
        "confusion_matrix": {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
        },
        "predicted_positives": int(tp + fp),
        "false_positives": int(fp),
        "true_positives": int(tp),
        "true_negatives": int(tn),
        "false_negatives": int(fn),
    }

    if timing_stats:
        results.update(timing_stats)

    if edge_stats:
        results.update(edge_stats)

    return results
