"""Statistical Testing and Clustered Resampling Module.
Implements paired cluster-bootstrap hypothesis testing over registered domain clusters
to evaluate metric differences without violating independent and identically distributed (i.i.d.) assumptions.
"""

from typing import Callable, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score, precision_score, recall_score


def paired_cluster_bootstrap(
    y_true: np.ndarray,
    y_pred_a: np.ndarray,
    y_pred_b: np.ndarray,
    clusters: np.ndarray,
    metric_func: Callable[[np.ndarray, np.ndarray], float] = f1_score,
    n_bootstraps: int = 1000,
    alpha: float = 0.05,
    random_state: int = 42
) -> Dict[str, float]:
    """Perform paired cluster-bootstrap hypothesis test over registered domains.
    
    Args:
        y_true: Ground-truth binary labels (N,).
        y_pred_a: Predictions or probabilities from Model A (N,).
        y_pred_b: Predictions or probabilities from Model B (N,).
        clusters: Registered domain or cluster IDs for each sample (N,).
        metric_func: Metric function f(y_true, y_pred) -> float.
        n_bootstraps: Number of bootstrap cluster resamples.
        alpha: Significance level (default 0.05 for 95% CI).
        random_state: Seed for reproducibility.
        
    Returns:
        Dict with delta_mean, ci_lower, ci_upper, p_value, is_significant.
    """
    rng = np.random.default_rng(random_state)
    
    y_true = np.asarray(y_true)
    y_pred_a = np.asarray(y_pred_a)
    y_pred_b = np.asarray(y_pred_b)
    clusters = np.asarray(clusters)

    # Point estimates on observed data
    val_a = metric_func(y_true, y_pred_a)
    val_b = metric_func(y_true, y_pred_b)
    observed_delta = float(val_a - val_b)

    # Group sample indices by cluster
    unique_clusters = np.unique(clusters)
    n_clusters = len(unique_clusters)
    cluster_to_indices = {c: np.where(clusters == c)[0] for c in unique_clusters}

    boot_deltas = np.empty(n_bootstraps, dtype=np.float64)

    for i in range(n_bootstraps):
        # Resample clusters with replacement
        sampled_clusters = rng.choice(unique_clusters, size=n_clusters, replace=True)
        # Collect sample indices corresponding to resampled clusters
        boot_indices = np.concatenate([cluster_to_indices[c] for c in sampled_clusters])

        y_boot_true = y_true[boot_indices]
        # Skip resample if only one class is present (e.g. for AUC)
        if len(np.unique(y_boot_true)) < 2:
            boot_deltas[i] = observed_delta
            continue

        boot_a = metric_func(y_boot_true, y_pred_a[boot_indices])
        boot_b = metric_func(y_boot_true, y_pred_b[boot_indices])
        boot_deltas[i] = boot_a - boot_b

    # Percentile confidence interval
    lower_pct = 100.0 * (alpha / 2.0)
    upper_pct = 100.0 * (1.0 - alpha / 2.0)
    ci_lower = float(np.percentile(boot_deltas, lower_pct))
    ci_upper = float(np.percentile(boot_deltas, upper_pct))

    # Two-sided p-value: fraction of centered bootstraps as extreme as observed delta
    # Center bootstrap distribution under null hypothesis (mean delta = 0)
    centered_deltas = boot_deltas - np.mean(boot_deltas)
    p_value = float(np.mean(np.abs(centered_deltas) >= np.abs(observed_delta)))

    return {
        "observed_metric_a": float(val_a),
        "observed_metric_b": float(val_b),
        "observed_delta": observed_delta,
        "bootstrap_mean_delta": float(np.mean(boot_deltas)),
        "bootstrap_std": float(np.std(boot_deltas)),
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "p_value": p_value,
        "is_significant": bool(ci_lower > 0.0 or ci_upper < 0.0),
    }


def cluster_bootstrap_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    clusters: np.ndarray,
    metric_func: Callable[[np.ndarray, np.ndarray], float] = f1_score,
    n_bootstraps: int = 1000,
    alpha: float = 0.05,
    random_state: int = 42
) -> Dict[str, float]:
    """Compute domain-clustered confidence interval for a single model's metric."""
    rng = np.random.default_rng(random_state)
    
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    clusters = np.asarray(clusters)

    point_estimate = float(metric_func(y_true, y_pred))

    unique_clusters = np.unique(clusters)
    n_clusters = len(unique_clusters)
    cluster_to_indices = {c: np.where(clusters == c)[0] for c in unique_clusters}

    boot_vals = np.empty(n_bootstraps, dtype=np.float64)

    for i in range(n_bootstraps):
        sampled_clusters = rng.choice(unique_clusters, size=n_clusters, replace=True)
        boot_indices = np.concatenate([cluster_to_indices[c] for c in sampled_clusters])

        y_boot_true = y_true[boot_indices]
        if len(np.unique(y_boot_true)) < 2:
            boot_vals[i] = point_estimate
            continue

        boot_vals[i] = metric_func(y_boot_true, y_pred[boot_indices])

    lower_pct = 100.0 * (alpha / 2.0)
    upper_pct = 100.0 * (1.0 - alpha / 2.0)
    ci_lower = float(np.percentile(boot_vals, lower_pct))
    ci_upper = float(np.percentile(boot_vals, upper_pct))

    return {
        "point_estimate": point_estimate,
        "bootstrap_mean": float(np.mean(boot_vals)),
        "bootstrap_std": float(np.std(boot_vals)),
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
    }
