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


def measure_inference_latency(model, X_sample, n_iterations=100, warmup_iterations=20) -> float:
    """Measure single-sample model inference latency in milliseconds (mean across N runs)."""
    if len(X_sample) == 0:
        return 0.0
    # Warmup CPU/cache
    for _ in range(warmup_iterations):
        _ = model.predict_proba(X_sample[:1])
    
    timings = []
    for _ in range(n_iterations):
        t0 = time.perf_counter()
        _ = model.predict_proba(X_sample[:1])
        timings.append((time.perf_counter() - t0) * 1000.0)
    return float(np.mean(timings))


def measure_detailed_latencies(model, X_sample, n_iterations=100, warmup_iterations=20) -> dict:
    """Measure single-sample inference latency (mean, median, P95, P99) in milliseconds."""
    if len(X_sample) == 0:
        return {"latency_mean_ms": 0.0, "latency_median_ms": 0.0, "latency_p95_ms": 0.0, "latency_p99_ms": 0.0}
    # Warmup
    for _ in range(warmup_iterations):
        _ = model.predict_proba(X_sample[:1])
    
    timings = []
    for _ in range(n_iterations):
        t0 = time.perf_counter()
        _ = model.predict_proba(X_sample[:1])
        timings.append((time.perf_counter() - t0) * 1000.0)
    return {
        "latency_mean_ms": float(np.mean(timings)),
        "latency_median_ms": float(np.median(timings)),
        "latency_p95_ms": float(np.percentile(timings, 95)),
        "latency_p99_ms": float(np.percentile(timings, 99))
    }


def measure_pipeline_latency(model, sample_url: str = "https://sbi-verification.xyz/login", n_iterations=100) -> dict:
    """Measure full end-to-end pipeline latency: URL string -> feature extraction -> model inference."""
    from src.features.extractor import extract_features_dict, get_feature_names
    cols = get_feature_names(include_regional=True)
    
    # Warmup
    for _ in range(10):
        fd = extract_features_dict(sample_url, include_regional=True)
        vec = np.array([[fd[c] for c in cols]])
        _ = model.predict_proba(vec)
        
    feat_times = []
    infer_times = []
    total_times = []
    for _ in range(n_iterations):
        t0 = time.perf_counter()
        fd = extract_features_dict(sample_url, include_regional=True)
        vec = np.array([[fd[c] for c in cols]])
        t1 = time.perf_counter()
        _ = model.predict_proba(vec)
        t2 = time.perf_counter()
        
        feat_times.append((t1 - t0) * 1000.0)
        infer_times.append((t2 - t1) * 1000.0)
        total_times.append((t2 - t0) * 1000.0)
        
    return {
        "feature_extraction_ms": float(np.mean(feat_times)),
        "model_inference_ms": float(np.mean(infer_times)),
        "end_to_end_latency_ms": float(np.mean(total_times)),
        "end_to_end_p95_ms": float(np.percentile(total_times, 95))
    }


def compute_model_size_mb(filepath: str) -> float:
    """Get model file size in Megabytes."""
    if os.path.exists(filepath):
        return os.path.getsize(filepath) / (1024 * 1024)
    return 0.0
