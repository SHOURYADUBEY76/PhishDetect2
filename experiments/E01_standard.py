"""Experiment E1: Standard Baseline Evaluation.
Trains classifiers on conventional global phishing data (PhreshPhish) and evaluates on held-out global test set.
Uses conventional features (lexical + structural, without regional features).
Evaluates across 5 seeds and includes source-bias audit (path/query removal).
Outputs results/E01.csv.
"""

import os
import sys
import pandas as pd
import numpy as np

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models.baseline import PhishingClassifier
from src.utils.data_loader import load_processed_split, split_train_val_grouped, get_feature_matrix


SEEDS = [42, 43, 44, 45, 46]
MODEL_TYPES = ["logistic_regression", "random_forest", "lightgbm"]


def run_experiment_e1():
    os.makedirs("results", exist_ok=True)
    os.makedirs("backend", exist_ok=True)
    print("\n=======================================================", flush=True)
    print("   EXPERIMENT E1: STANDARD BENCHMARK GENERALIZATION", flush=True)
    print("=======================================================", flush=True)

    df_train_full = load_processed_split("global_train")
    df_test = load_processed_split("global_test")
    print(f"Loaded Global Train ({len(df_train_full)} rows) and Global Test ({len(df_test)} rows)", flush=True)

    # 1. Main Standard Baseline (Conventional Features: Lexical + Structural)
    results_records = []
    best_lgbm_model = None
    best_lgbm_f1 = -1.0

    for model_name in MODEL_TYPES:
        print(f"\n--- Evaluating Model: {model_name.upper()} over 5 seeds ---", flush=True)
        seed_metrics = []

        for seed in SEEDS:
            # Grouped train/val split for threshold calibration
            df_train, df_val = split_train_val_grouped(df_train_full, val_domain_ratio=0.15, random_state=seed)

            # Feature matrices (lexical + structural only, include_regional=False)
            X_train, y_train, _ = get_feature_matrix(df_train, f"gtrain_seed{seed}", include_regional=False, include_path_query=True)
            X_val, y_val, _ = get_feature_matrix(df_val, f"gval_seed{seed}", include_regional=False, include_path_query=True)
            X_test, y_test, _ = get_feature_matrix(df_test, "gtest", include_regional=False, include_path_query=True)

            clf = PhishingClassifier(model_type=model_name, random_state=seed)
            clf.fit(X_train, y_train, X_val=X_val, y_val=y_val)

            metrics = clf.evaluate(X_test, y_test)
            seed_metrics.append(metrics)

            if model_name == "lightgbm" and metrics["f1"] > best_lgbm_f1:
                best_lgbm_f1 = metrics["f1"]
                best_lgbm_model = clf

        # Aggregate metrics across seeds
        avg_res = {
            "experiment": "E01_standard",
            "model": model_name,
            "feature_set": "lexical_structural",
            "path_query": True,
            "precision_mean": float(np.mean([m["precision"] for m in seed_metrics])),
            "precision_std": float(np.std([m["precision"] for m in seed_metrics])),
            "recall_mean": float(np.mean([m["recall"] for m in seed_metrics])),
            "recall_std": float(np.std([m["recall"] for m in seed_metrics])),
            "f1_mean": float(np.mean([m["f1"] for m in seed_metrics])),
            "f1_std": float(np.std([m["f1"] for m in seed_metrics])),
            "roc_auc_mean": float(np.mean([m["roc_auc"] for m in seed_metrics])),
            "roc_auc_std": float(np.std([m["roc_auc"] for m in seed_metrics])),
            "fpr_mean": float(np.mean([m["fpr"] for m in seed_metrics])),
            "fpr_std": float(np.std([m["fpr"] for m in seed_metrics])),
            "fnr_mean": float(np.mean([m["fnr"] for m in seed_metrics])),
            "fnr_std": float(np.std([m["fnr"] for m in seed_metrics])),
            "latency_ms": float(np.mean([m["latency_ms"] for m in seed_metrics])),
            "model_size_mb": float(np.mean([m["model_size_mb"] for m in seed_metrics])),
        }
        results_records.append(avg_res)
        print(f"  Result [{model_name}]: F1={avg_res['f1_mean']:.4f}±{avg_res['f1_std']:.4f} | ROC-AUC={avg_res['roc_auc_mean']:.4f}±{avg_res['roc_auc_std']:.4f} | Latency={avg_res['latency_ms']:.3f}ms", flush=True)

    # 2. Source Bias Audit: evaluate LightGBM with path and query features masked/removed
    print("\n--- Running Source Bias Audit (Path & Query Removed) ---", flush=True)
    bias_metrics = []
    for seed in SEEDS:
        df_train, df_val = split_train_val_grouped(df_train_full, val_domain_ratio=0.15, random_state=seed)
        X_train, y_train, _ = get_feature_matrix(df_train, f"gtrain_seed{seed}_nopq", include_regional=False, include_path_query=False)
        X_val, y_val, _ = get_feature_matrix(df_val, f"gval_seed{seed}_nopq", include_regional=False, include_path_query=False)
        X_test, y_test, _ = get_feature_matrix(df_test, "gtest_nopq", include_regional=False, include_path_query=False)

        clf = PhishingClassifier(model_type="lightgbm", random_state=seed)
        clf.fit(X_train, y_train, X_val=X_val, y_val=y_val)
        bias_metrics.append(clf.evaluate(X_test, y_test))

    bias_res = {
        "experiment": "E01_source_bias_audit",
        "model": "lightgbm",
        "feature_set": "lexical_structural_nopq",
        "path_query": False,
        "precision_mean": float(np.mean([m["precision"] for m in bias_metrics])),
        "precision_std": float(np.std([m["precision"] for m in bias_metrics])),
        "recall_mean": float(np.mean([m["recall"] for m in bias_metrics])),
        "recall_std": float(np.std([m["recall"] for m in bias_metrics])),
        "f1_mean": float(np.mean([m["f1"] for m in bias_metrics])),
        "f1_std": float(np.std([m["f1"] for m in bias_metrics])),
        "roc_auc_mean": float(np.mean([m["roc_auc"] for m in bias_metrics])),
        "roc_auc_std": float(np.std([m["roc_auc"] for m in bias_metrics])),
        "fpr_mean": float(np.mean([m["fpr"] for m in bias_metrics])),
        "fpr_std": float(np.std([m["fpr"] for m in bias_metrics])),
        "fnr_mean": float(np.mean([m["fnr"] for m in bias_metrics])),
        "fnr_std": float(np.std([m["fnr"] for m in bias_metrics])),
        "latency_ms": float(np.mean([m["latency_ms"] for m in bias_metrics])),
        "model_size_mb": float(np.mean([m["model_size_mb"] for m in bias_metrics])),
    }
    results_records.append(bias_res)
    print(f"  Source Bias Audit [LGBM No Path/Query]: F1={bias_res['f1_mean']:.4f}±{bias_res['f1_std']:.4f} | ROC-AUC={bias_res['roc_auc_mean']:.4f}±{bias_res['roc_auc_std']:.4f}", flush=True)

    # Save output table
    res_df = pd.DataFrame(results_records)
    res_df.to_csv("results/E01.csv", index=False)
    print(f"\nSaved Experiment E1 results to results/E01.csv", flush=True)

    # Save baseline model for initial backend use
    if best_lgbm_model is not None:
        best_lgbm_model.save("backend/model_baseline.joblib")
        print("Saved best LightGBM baseline model to backend/model_baseline.joblib", flush=True)

    return res_df


if __name__ == "__main__":
    run_experiment_e1()
