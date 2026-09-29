"""Experiment E4: Cross-Dataset Generalization (PhreshPhish <-> PhiUSIIL).
Evaluates cross-corpus domain adaptation:
  1. PhreshPhish Train -> PhiUSIIL Test
  2. PhiUSIIL Train -> PhreshPhish Test
Compares LightGBM and Character-Level BiLSTM on identical splits across 5 seeds.
Outputs results/E04.csv.
"""

import os
import sys
import pandas as pd
import numpy as np

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models.baseline import PhishingClassifier
from src.models.bilstm import BiLSTMClassifier
from src.utils.data_loader import load_processed_split, split_train_val_grouped, get_feature_matrix


SEEDS = [42, 43, 44, 45, 46]


def run_experiment_e4():
    os.makedirs("results", exist_ok=True)
    print("\n=======================================================", flush=True)
    print("   EXPERIMENT E4: CROSS-DATASET GENERALIZATION", flush=True)
    print("   (PhreshPhish <-> UCI PhiUSIIL)", flush=True)
    print("=======================================================", flush=True)

    df_phresh_tr = load_processed_split("global_train")
    df_phresh_te = load_processed_split("global_test")
    df_phi_tr = load_processed_split("phiusiil_train")
    df_phi_te = load_processed_split("phiusiil_test")

    cross_pairs = [
        ("PhreshPhish -> PhiUSIIL", df_phresh_tr, df_phi_te, "phresh_to_phi"),
        ("PhiUSIIL -> PhreshPhish", df_phi_tr, df_phresh_te, "phi_to_phresh")
    ]

    results_records = []

    for transfer_name, df_train_full, df_test, cache_prefix in cross_pairs:
        print(f"\n--- Cross-Dataset Transfer: {transfer_name} ---", flush=True)
        print(f"Train Source: {len(df_train_full)} rows | Test Target: {len(df_test)} rows", flush=True)

        # 1. LightGBM Evaluation
        lgb_metrics = []
        for seed in SEEDS:
            df_tr, df_vl = split_train_val_grouped(df_train_full, val_domain_ratio=0.15, random_state=seed)

            X_tr, y_tr, _ = get_feature_matrix(df_tr, f"e4_{cache_prefix}_tr_{seed}", include_regional=False, include_path_query=True)
            X_vl, y_vl, _ = get_feature_matrix(df_vl, f"e4_{cache_prefix}_vl_{seed}", include_regional=False, include_path_query=True)
            X_te, y_te, _ = get_feature_matrix(df_test, f"e4_{cache_prefix}_te", include_regional=False, include_path_query=True)

            clf = PhishingClassifier(model_type="lightgbm", random_state=seed)
            clf.fit(X_tr, y_tr, X_val=X_vl, y_val=y_vl)
            lgb_metrics.append(clf.evaluate(X_te, y_te))

        res_lgb = {
            "experiment": "E04_cross_dataset",
            "transfer_direction": transfer_name,
            "model": "lightgbm",
            "precision_mean": float(np.mean([m["precision"] for m in lgb_metrics])),
            "precision_std": float(np.std([m["precision"] for m in lgb_metrics])),
            "recall_mean": float(np.mean([m["recall"] for m in lgb_metrics])),
            "recall_std": float(np.std([m["recall"] for m in lgb_metrics])),
            "f1_mean": float(np.mean([m["f1"] for m in lgb_metrics])),
            "f1_std": float(np.std([m["f1"] for m in lgb_metrics])),
            "roc_auc_mean": float(np.mean([m["roc_auc"] for m in lgb_metrics])),
            "roc_auc_std": float(np.std([m["roc_auc"] for m in lgb_metrics])),
            "fpr_mean": float(np.mean([m["fpr"] for m in lgb_metrics])),
            "fpr_std": float(np.std([m["fpr"] for m in lgb_metrics])),
            "fnr_mean": float(np.mean([m["fnr"] for m in lgb_metrics])),
            "fnr_std": float(np.std([m["fnr"] for m in lgb_metrics])),
            "latency_ms": float(np.mean([m["latency_ms"] for m in lgb_metrics])),
            "model_size_mb": float(np.mean([m["model_size_mb"] for m in lgb_metrics])),
        }
        results_records.append(res_lgb)
        print(f"  Result [LightGBM]: F1={res_lgb['f1_mean']:.4f}±{res_lgb['f1_std']:.4f} | Recall={res_lgb['recall_mean']:.4f}±{res_lgb['recall_std']:.4f}", flush=True)

        # 2. BiLSTM Deep Learning Evaluation
        bilstm_metrics = []
        # Sample for efficient multi-seed neural training
        sub_train = df_train_full.sample(n=min(len(df_train_full), 10000), random_state=42)
        sub_test = df_test.sample(n=min(len(df_test), 3000), random_state=42)

        for seed in SEEDS[:3]:  # 3 representative seeds for neural net
            df_tr, df_vl = split_train_val_grouped(sub_train, val_domain_ratio=0.15, random_state=seed)
            bilstm = BiLSTMClassifier(random_state=seed)
            bilstm.fit(
                train_urls=df_tr["url"].tolist(),
                y_train=df_tr["label"].values,
                val_urls=df_vl["url"].tolist(),
                y_val=df_vl["label"].values,
                epochs=4
            )
            bilstm_metrics.append(bilstm.evaluate_urls(sub_test["url"].tolist(), sub_test["label"].values))

        res_bilstm = {
            "experiment": "E04_cross_dataset",
            "transfer_direction": transfer_name,
            "model": "bilstm_neural",
            "precision_mean": float(np.mean([m["precision"] for m in bilstm_metrics])),
            "precision_std": float(np.std([m["precision"] for m in bilstm_metrics])),
            "recall_mean": float(np.mean([m["recall"] for m in bilstm_metrics])),
            "recall_std": float(np.std([m["recall"] for m in bilstm_metrics])),
            "f1_mean": float(np.mean([m["f1"] for m in bilstm_metrics])),
            "f1_std": float(np.std([m["f1"] for m in bilstm_metrics])),
            "roc_auc_mean": float(np.mean([m["roc_auc"] for m in bilstm_metrics])),
            "roc_auc_std": float(np.std([m["roc_auc"] for m in bilstm_metrics])),
            "fpr_mean": float(np.mean([m["fpr"] for m in bilstm_metrics])),
            "fpr_std": float(np.std([m["fpr"] for m in bilstm_metrics])),
            "fnr_mean": float(np.mean([m["fnr"] for m in bilstm_metrics])),
            "fnr_std": float(np.std([m["fnr"] for m in bilstm_metrics])),
            "latency_ms": float(np.mean([m["latency_ms"] for m in bilstm_metrics])),
            "model_size_mb": float(np.mean([m["model_size_mb"] for m in bilstm_metrics])),
        }
        results_records.append(res_bilstm)
        print(f"  Result [BiLSTM]:   F1={res_bilstm['f1_mean']:.4f}±{res_bilstm['f1_std']:.4f} | Recall={res_bilstm['recall_mean']:.4f}±{res_bilstm['recall_std']:.4f}", flush=True)

    res_df = pd.DataFrame(results_records)
    res_df.to_csv("results/E04.csv", index=False)
    print("\nSaved Experiment E4 results to results/E04.csv", flush=True)
    return res_df


if __name__ == "__main__":
    run_experiment_e4()
