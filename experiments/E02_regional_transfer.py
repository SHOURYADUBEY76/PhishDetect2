"""Experiment E2: Regional Transfer Gap Evaluation.
Quantifies the performance degradation when models trained on conventional global benchmarks
are tested out-of-distribution on Indian-targeted phishing traffic.
Uses conventional features (lexical + structural only).
Averaged over 5 seeds. Outputs results/E02.csv.
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


def run_experiment_e2():
    os.makedirs("results", exist_ok=True)
    print("\n=======================================================", flush=True)
    print("   EXPERIMENT E2: REGIONAL TRANSFER GAP EVALUATION", flush=True)
    print("   (Global Train -> Indian-Targeted Test)", flush=True)
    print("=======================================================", flush=True)

    df_train_full = load_processed_split("global_train")
    df_indian_test = load_processed_split("indian_test")
    print(f"Loaded Global Train ({len(df_train_full)} rows) and Indian Test ({len(df_indian_test)} rows)", flush=True)

    # Indian test feature matrix (conventional features: lexical + structural only)
    X_itest, y_itest, _ = get_feature_matrix(df_indian_test, "indian_test", include_regional=False, include_path_query=True)

    results_records = []

    for model_name in MODEL_TYPES:
        print(f"\n--- Evaluating Model: {model_name.upper()} on Indian Test Set ---", flush=True)
        seed_metrics = []

        for seed in SEEDS:
            # Grouped train/val split from global train for calibrated threshold
            df_train, df_val = split_train_val_grouped(df_train_full, val_domain_ratio=0.15, random_state=seed)

            X_train, y_train, _ = get_feature_matrix(df_train, f"gtrain_seed{seed}", include_regional=False, include_path_query=True)
            X_val, y_val, _ = get_feature_matrix(df_val, f"gval_seed{seed}", include_regional=False, include_path_query=True)

            clf = PhishingClassifier(model_type=model_name, random_state=seed)
            clf.fit(X_train, y_train, X_val=X_val, y_val=y_val)

            metrics = clf.evaluate(X_itest, y_itest)
            seed_metrics.append(metrics)

        avg_res = {
            "experiment": "E02_regional_transfer",
            "model": model_name,
            "feature_set": "lexical_structural",
            "test_target": "indian_test",
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
        print(f"  Result [{model_name}]: F1={avg_res['f1_mean']:.4f}±{avg_res['f1_std']:.4f} | Recall={avg_res['recall_mean']:.4f}±{avg_res['recall_std']:.4f} | FNR={avg_res['fnr_mean']:.4f}±{avg_res['fnr_std']:.4f}", flush=True)

    # Save output table
    res_df = pd.DataFrame(results_records)
    res_df.to_csv("results/E02.csv", index=False)
    print(f"\nSaved Experiment E2 results to results/E02.csv", flush=True)
    return res_df


if __name__ == "__main__":
    run_experiment_e2()
