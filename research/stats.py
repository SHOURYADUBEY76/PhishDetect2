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
    
    Estimand Note:
        Resampling registered-domain clusters and retaining all URLs within sampled clusters
        estimates URL-weighted performance under cluster-correlated sampling variance (accounting
        for intra-domain correlation). It is NOT equal-domain-weighted performance.
    
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
        Dict with observed_delta, bootstrap_mean_delta, ci_lower, ci_upper, interval_width,
        p_value, is_significant, n_invalid_replicates, n_valid_replicates.
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
    cluster_to_indices = {c: np.where(clusters == c)[0] for c in unique_clusters}

    # Stratified cluster pools by majority class
    clusters_0 = [c for c in unique_clusters if np.mean(y_true[cluster_to_indices[c]]) < 0.5]
    clusters_1 = [c for c in unique_clusters if np.mean(y_true[cluster_to_indices[c]]) >= 0.5]
    has_two_classes = (len(clusters_0) > 0 and len(clusters_1) > 0)

    boot_deltas = []
    n_invalid = 0

    for _ in range(n_bootstraps):
        if has_two_classes:
            sampled_0 = rng.choice(clusters_0, size=len(clusters_0), replace=True)
            sampled_1 = rng.choice(clusters_1, size=len(clusters_1), replace=True)
            sampled_clusters = np.concatenate([sampled_0, sampled_1])
        else:
            sampled_clusters = rng.choice(unique_clusters, size=len(unique_clusters), replace=True)

        boot_indices = np.concatenate([cluster_to_indices[c] for c in sampled_clusters])
        y_boot_true = y_true[boot_indices]

        if len(np.unique(y_boot_true)) < 2:
            n_invalid += 1
            continue

        try:
            boot_a = float(metric_func(y_boot_true, y_pred_a[boot_indices]))
            boot_b = float(metric_func(y_boot_true, y_pred_b[boot_indices]))
            if np.isnan(boot_a) or np.isnan(boot_b):
                n_invalid += 1
            else:
                boot_deltas.append(boot_a - boot_b)
        except Exception:
            n_invalid += 1

    boot_deltas = np.array(boot_deltas)
    if len(boot_deltas) == 0:
        return {
            "observed_metric_a": float(val_a),
            "observed_metric_b": float(val_b),
            "observed_delta": observed_delta,
            "bootstrap_mean_delta": float("nan"),
            "bootstrap_std": float("nan"),
            "ci_lower": float("nan"),
            "ci_upper": float("nan"),
            "interval_width": float("nan"),
            "p_value": float("nan"),
            "is_significant": False,
            "n_invalid_replicates": n_invalid,
            "n_valid_replicates": 0,
        }

    # Percentile confidence interval
    lower_pct = 100.0 * (alpha / 2.0)
    upper_pct = 100.0 * (1.0 - alpha / 2.0)
    ci_lower = float(np.percentile(boot_deltas, lower_pct))
    ci_upper = float(np.percentile(boot_deltas, upper_pct))
    interval_width = float(ci_upper - ci_lower)

    # Two-sided empirical p-value against null hypothesis mean = 0
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
        "interval_width": interval_width,
        "p_value": p_value,
        "is_significant": bool(ci_lower > 0.0 or ci_upper < 0.0),
        "n_invalid_replicates": n_invalid,
        "n_valid_replicates": len(boot_deltas),
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
    """Compute domain-clustered confidence interval for a single model's metric.
    
    Estimand Note:
        Resampling registered-domain clusters and retaining all URLs within sampled clusters
        estimates URL-weighted performance under cluster-correlated sampling variance.
        It is NOT equal-domain-weighted performance.
    """
    rng = np.random.default_rng(random_state)
    
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    clusters = np.asarray(clusters)

    point_estimate = float(metric_func(y_true, y_pred))

    unique_clusters = np.unique(clusters)
    cluster_to_indices = {c: np.where(clusters == c)[0] for c in unique_clusters}

    clusters_0 = [c for c in unique_clusters if np.mean(y_true[cluster_to_indices[c]]) < 0.5]
    clusters_1 = [c for c in unique_clusters if np.mean(y_true[cluster_to_indices[c]]) >= 0.5]
    has_two_classes = (len(clusters_0) > 0 and len(clusters_1) > 0)

    boot_vals = []
    n_invalid = 0

    for _ in range(n_bootstraps):
        if has_two_classes:
            sampled_0 = rng.choice(clusters_0, size=len(clusters_0), replace=True)
            sampled_1 = rng.choice(clusters_1, size=len(clusters_1), replace=True)
            sampled_clusters = np.concatenate([sampled_0, sampled_1])
        else:
            sampled_clusters = rng.choice(unique_clusters, size=len(unique_clusters), replace=True)

        boot_indices = np.concatenate([cluster_to_indices[c] for c in sampled_clusters])
        y_boot_true = y_true[boot_indices]

        if len(np.unique(y_boot_true)) < 2:
            n_invalid += 1
            continue

        try:
            val = float(metric_func(y_boot_true, y_pred[boot_indices]))
            if np.isnan(val):
                n_invalid += 1
            else:
                boot_vals.append(val)
        except Exception:
            n_invalid += 1

    boot_vals = np.array(boot_vals)
    if len(boot_vals) == 0:
        return {
            "point_estimate": point_estimate,
            "bootstrap_mean": float("nan"),
            "bootstrap_std": float("nan"),
            "ci_lower": float("nan"),
            "ci_upper": float("nan"),
            "interval_width": float("nan"),
            "n_invalid_replicates": n_invalid,
            "n_valid_replicates": 0,
        }

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
        "interval_width": float(ci_upper - ci_lower),
        "n_invalid_replicates": n_invalid,
        "n_valid_replicates": len(boot_vals),
    }
