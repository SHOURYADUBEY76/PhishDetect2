"""Evaluation metrics calculation and threshold optimization.
Reports Precision, Recall, F1, ROC-AUC, False Positive Rate (FPR), False Negative Rate (FNR).
"""

import time
import os
import numpy as np
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)


def find_optimal_threshold(y_true, y_probs, metric="f1", grid_size=100) -> float:
    """Find the decision threshold that maximizes validation F1 score.
    
    Tuned on validation sets only to prevent optimistic test bias.
    """
    thresholds = np.linspace(0.05, 0.95, grid_size)
    best_thresh = 0.5
    best_score = -1.0

    for t in thresholds:
        preds = (y_probs >= t).astype(int)
        if metric == "f1":
            score = f1_score(y_true, preds, zero_division=0)
        else:
            score = f1_score(y_true, preds, zero_division=0)

        if score > best_score:
            best_score = score
            best_thresh = float(t)

    return best_thresh


def calculate_metrics(y_true, y_probs, threshold=0.5, latency_ms=0.0, model_size_mb=0.0) -> dict:
    """Compute comprehensive evaluation metrics.
    
    Includes precision, recall, F1, ROC-AUC, FPR, FNR, latency, and model size.
    """
    y_true = np.array(y_true, dtype=int)
    y_probs = np.array(y_probs, dtype=float)
    y_pred = (y_probs >= threshold).astype(int)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0

    # Handle single class edge cases for ROC-AUC
    try:
        roc_auc = float(roc_auc_score(y_true, y_probs))
    except Exception:
        roc_auc = 0.5

    return {
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": roc_auc,
        "fpr": float(fpr),
        "fnr": float(fnr),
        "threshold": float(threshold),
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
        "latency_ms": float(latency_ms),
        "model_size_mb": float(model_size_mb),
    }


def measure_inference_latency(model, X_sample, n_iterations=100) -> float:
    """Measure single-sample inference latency in milliseconds."""
    if len(X_sample) == 0:
        return 0.0
    # Warmup
    _ = model.predict_proba(X_sample[:min(5, len(X_sample))])
    
    start_time = time.perf_counter()
    for _ in range(n_iterations):
        _ = model.predict_proba(X_sample[:1])
    elapsed = time.perf_counter() - start_time
    return (elapsed / n_iterations) * 1000.0


def compute_model_size_mb(filepath: str) -> float:
    """Get model file size in Megabytes."""
    if os.path.exists(filepath):
        return os.path.getsize(filepath) / (1024 * 1024)
    return 0.0
