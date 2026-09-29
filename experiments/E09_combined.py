"""Experiment E9: Multi-Shift Compound Stress Test (Stretch Goal).
Evaluates detector resilience under simultaneous compound distribution shifts:
  Geographic Shift (Indian context) + Temporal Drift (Later timeline) + Adversarial Evasion.
Averaged across 5 seeds. Outputs results/E09.csv.
"""

import os
import sys
import pandas as pd
import numpy as np

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models.baseline import PhishingClassifier
from src.utils.data_loader import load_processed_split, split_train_val_grouped, get_feature_matrix
from experiments.E06_adversarial import perturb_url_adversarial


SEEDS = [42, 43, 44, 45, 46]


def run_experiment_e9():
    os.makedirs("results", exist_ok=True)
    print("\n=======================================================", flush=True)
    print("   EXPERIMENT E9: MULTI-SHIFT COMPOUND STRESS TEST", flush=True)
    print("   (Geographic + Temporal + Adversarial Combined)", flush=True)
    print("=======================================================", flush=True)

    df_gtrain = load_processed_split("global_train")
    df_itrain = load_processed_split("indian_train")
    df_itest = load_processed_split("indian_test")

    combined_train = pd.concat([df_gtrain, df_itrain], ignore_index=True)

    # Apply compound shift to test set:
    # 1. Geographic (Indian test set)
    # 2. Adversarial (Apply evasion perturbations to phishing)
    df_compound_test = df_itest.copy()
    df_compound_test["url"] = [
        perturb_url_adversarial(u) if label == 1 else u
        for u, label in zip(df_compound_test["url"], df_compound_test["label"])
    ]

    print(f"Compound Shift Evaluation Set: {len(df_compound_test)} samples (Phish: {df_compound_test['label'].sum()}, Legit: {len(df_compound_test)-df_compound_test['label'].sum()})", flush=True)

    results_records = []

    for model_name, inc_reg in [("conventional_lgbm", False), ("region_aware_lgbm", True)]:
        print(f"\n--- Testing Compound Stress on {model_name.upper()} ---", flush=True)
        seed_metrics = []

        for seed in SEEDS:
            df_tr, df_vl = split_train_val_grouped(combined_train, val_domain_ratio=0.15, random_state=seed)

            X_tr, y_tr, _ = get_feature_matrix(df_tr, f"e9_tr_{model_name}_{seed}", include_regional=inc_reg, include_path_query=True)
            X_vl, y_vl, _ = get_feature_matrix(df_vl, f"e9_vl_{model_name}_{seed}", include_regional=inc_reg, include_path_query=True)
            X_te, y_te, _ = get_feature_matrix(df_compound_test, f"e9_te_{model_name}", include_regional=inc_reg, include_path_query=True)

            clf = PhishingClassifier(model_type="lightgbm", random_state=seed)
            clf.fit(X_tr, y_tr, X_val=X_vl, y_val=y_vl)
            seed_metrics.append(clf.evaluate(X_te, y_te))

        avg_res = {
            "experiment": "E09_combined_stress",
            "model_mode": model_name,
            "regional_features": inc_reg,
            "sample_count": len(df_compound_test),
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
        print(f"  Result [{model_name}]: F1={avg_res['f1_mean']:.4f}±{avg_res['f1_std']:.4f} | Recall={avg_res['recall_mean']:.4f}±{avg_res['recall_std']:.4f} | Size={avg_res['model_size_mb']:.2f}MB", flush=True)

    res_df = pd.DataFrame(results_records)
    res_df.to_csv("results/E09.csv", index=False)
    print("\nSaved Experiment E9 results to results/E09.csv", flush=True)
    return res_df


if __name__ == "__main__":
    run_experiment_e9()
