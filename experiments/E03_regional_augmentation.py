"""Experiment E3: Regional Augmentation & Strict Feature Ablation.
Evaluates the contribution of region-aware feature engineering when training on
augmented data (Global Train + Indian Train) and testing on Indian Test Set.
Conducts 3-way feature ablation:
  1. Lexical Only
  2. Lexical + Structural
  3. Full (Lexical + Structural + Regional)
Averaged across 5 seeds. Outputs results/E03.csv and saves backend/model_region_aware.joblib.
"""

import os
import sys
import pandas as pd
import numpy as np

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models.baseline import PhishingClassifier
from src.utils.data_loader import load_processed_split, split_train_val_grouped, get_feature_matrix
from src.features.extractor import LEXICAL_FEATURE_NAMES, STRUCTURAL_FEATURE_NAMES, REGIONAL_FEATURE_NAMES


SEEDS = [42, 43, 44, 45, 46]


def run_experiment_e3():
    os.makedirs("results", exist_ok=True)
    print("\n=======================================================", flush=True)
    print("   EXPERIMENT E3: REGIONAL AUGMENTATION & ABLATION", flush=True)
    print("=======================================================", flush=True)

    df_gtrain = load_processed_split("global_train")
    df_itrain = load_processed_split("indian_train")
    df_itest = load_processed_split("indian_test")

    # Combined training set: Global Train + Indian Train
    df_train_full = pd.concat([df_gtrain, df_itrain], ignore_index=True)
    print(f"Combined Training Set: {len(df_train_full)} samples (Global: {len(df_gtrain)}, Indian: {len(df_itrain)})", flush=True)
    print(f"Evaluation Target: Indian Test Set ({len(df_itest)} samples)", flush=True)

    # Feature configurations for ablation
    ablation_configs = [
        {
            "name": "lexical_only",
            "include_regional": False,
            "feature_subset": LEXICAL_FEATURE_NAMES,
            "desc": "Lexical Features Only"
        },
        {
            "name": "lexical_structural",
            "include_regional": False,
            "feature_subset": LEXICAL_FEATURE_NAMES + STRUCTURAL_FEATURE_NAMES,
            "desc": "Lexical + Structural Baseline"
        },
        {
            "name": "full_region_augmented",
            "include_regional": True,
            "feature_subset": LEXICAL_FEATURE_NAMES + STRUCTURAL_FEATURE_NAMES + REGIONAL_FEATURE_NAMES,
            "desc": "Full Region-Aware Augmented"
        }
    ]

    results_records = []
    best_overall_model = None
    best_overall_f1 = -1.0

    for cfg in ablation_configs:
        print(f"\n--- Ablation Regime: {cfg['desc']} ({len(cfg['feature_subset'])} features) ---", flush=True)
        seed_metrics = []

        for seed in SEEDS:
            df_train, df_val = split_train_val_grouped(df_train_full, val_domain_ratio=0.15, random_state=seed)

            # Feature matrices
            X_tr, y_tr, fnames_tr = get_feature_matrix(
                df_train, f"aug_train_seed{seed}",
                include_regional=cfg["include_regional"], include_path_query=True
            )
            X_vl, y_vl, fnames_vl = get_feature_matrix(
                df_val, f"aug_val_seed{seed}",
                include_regional=cfg["include_regional"], include_path_query=True
            )
            X_te, y_te, fnames_te = get_feature_matrix(
                df_itest, "indian_test_full",
                include_regional=cfg["include_regional"], include_path_query=True
            )

            # Filter columns to match feature_subset
            col_indices = [fnames_tr.index(f) for f in cfg["feature_subset"] if f in fnames_tr]
            X_tr_sub = X_tr[:, col_indices]
            X_vl_sub = X_vl[:, col_indices]
            X_te_sub = X_te[:, col_indices]

            clf = PhishingClassifier(model_type="lightgbm", random_state=seed)
            clf.fit(X_tr_sub, y_tr, X_val=X_vl_sub, y_val=y_vl)

            metrics = clf.evaluate(X_te_sub, y_te)
            seed_metrics.append(metrics)

            if cfg["name"] == "full_region_augmented" and metrics["f1"] > best_overall_f1:
                best_overall_f1 = metrics["f1"]
                best_overall_model = clf

        avg_res = {
            "experiment": "E03_ablation",
            "ablation_regime": cfg["name"],
            "feature_count": len(cfg["feature_subset"]),
            "model": "lightgbm",
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
        print(f"  Result [{cfg['name']}]: F1={avg_res['f1_mean']:.4f}±{avg_res['f1_std']:.4f} | Recall={avg_res['recall_mean']:.4f}±{avg_res['recall_std']:.4f} | FNR={avg_res['fnr_mean']:.4f}±{avg_res['fnr_std']:.4f}", flush=True)

    # Save output table
    res_df = pd.DataFrame(results_records)
    res_df.to_csv("results/E03.csv", index=False)
    print(f"\nSaved Experiment E3 ablation results to results/E03.csv", flush=True)

    # Save best region-aware model for deployment in backend
    if best_overall_model is not None:
        best_overall_model.save("backend/model_region_aware.joblib")
        print("Saved best region-aware model to backend/model_region_aware.joblib", flush=True)

    return res_df


if __name__ == "__main__":
    run_experiment_e3()
