"""Experiment E5: Temporal Drift Evaluation.
Evaluates model degradation when trained on earlier historical URLs and tested on later emerging URLs.
Uses genuine timestamps from PhreshPhish dataset.
Enforces zero registered domain leakage between temporal partitions.
Outputs results/E05.csv.
"""

import os
import sys
import pandas as pd
import numpy as np

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models.baseline import PhishingClassifier
from src.utils.data_loader import split_train_val_grouped, get_feature_matrix
from src.utils.domain import extract_registered_domain


SEEDS = [42, 43, 44, 45, 46]


def run_experiment_e5():
    os.makedirs("results", exist_ok=True)
    print("\n=======================================================", flush=True)
    print("   EXPERIMENT E5: TEMPORAL CONCEPT DRIFT EVALUATION", flush=True)
    print("   (Earlier Time Windows -> Later Time Windows)", flush=True)
    print("=======================================================", flush=True)

    df_raw = pd.read_csv("data/raw/phreshphish_raw.csv")
    df_raw = df_raw.dropna(subset=["date", "url"]).copy()
    df_raw["registered_domain"] = df_raw["url"].apply(extract_registered_domain)
    df_raw = df_raw.drop_duplicates(subset=["url"]).sort_values("date").reset_index(drop=True)

    min_date = df_raw["date"].min()
    max_date = df_raw["date"].max()
    print(f"Dataset date range: {min_date} to {max_date} ({len(df_raw)} total samples)", flush=True)

    # Chronological partition by unique registered domains based on their earliest appearance
    domain_first_seen = df_raw.groupby("registered_domain")["date"].min().sort_values()
    split_idx = int(0.7 * len(domain_first_seen))

    early_domains = set(domain_first_seen.index[:split_idx])
    late_domains = set(domain_first_seen.index[split_idx:])

    df_early = df_raw[df_raw["registered_domain"].isin(early_domains)].copy()
    df_late = df_raw[df_raw["registered_domain"].isin(late_domains)].copy()

    # Balance temporal train and test sets
    def balance_split(df):
        phish = df[df["label"] == 1]
        legit = df[df["label"] == 0]
        n = min(len(phish), len(legit))
        return pd.concat([
            phish.sample(n=n, random_state=42),
            legit.sample(n=n, random_state=42)
        ], ignore_index=True).sample(frac=1.0, random_state=42).reset_index(drop=True)

    df_early_bal = balance_split(df_early)
    df_late_bal = balance_split(df_late)

    print(f"Temporal Train (Earlier): {len(df_early_bal)} samples ({df_early['date'].min()} to {df_early['date'].max()})", flush=True)
    print(f"Temporal Test  (Later)  : {len(df_late_bal)} samples ({df_late['date'].min()} to {df_late['date'].max()})", flush=True)
    print(f"Domain Overlap: {len(early_domains.intersection(late_domains))} (Strict Zero Leakage: True)", flush=True)

    results_records = []

    for model_name in ["random_forest", "lightgbm"]:
        print(f"\n--- Evaluating Temporal Generalization with {model_name.upper()} ---", flush=True)
        seed_metrics = []

        for seed in SEEDS:
            df_tr, df_vl = split_train_val_grouped(df_early_bal, val_domain_ratio=0.15, random_state=seed)

            X_tr, y_tr, _ = get_feature_matrix(df_tr, f"e5_tr_{seed}", include_regional=False, include_path_query=True)
            X_vl, y_vl, _ = get_feature_matrix(df_vl, f"e5_vl_{seed}", include_regional=False, include_path_query=True)
            X_te, y_te, _ = get_feature_matrix(df_late_bal, "e5_te", include_regional=False, include_path_query=True)

            clf = PhishingClassifier(model_type=model_name, random_state=seed)
            clf.fit(X_tr, y_tr, X_val=X_vl, y_val=y_vl)
            seed_metrics.append(clf.evaluate(X_te, y_te))

        avg_res = {
            "experiment": "E05_temporal_drift",
            "model": model_name,
            "train_date_range": f"{df_early['date'].min()} to {df_early['date'].max()}",
            "test_date_range": f"{df_late['date'].min()} to {df_late['date'].max()}",
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
        print(f"  Result [{model_name}]: F1={avg_res['f1_mean']:.4f}±{avg_res['f1_std']:.4f} | Recall={avg_res['recall_mean']:.4f}±{avg_res['recall_std']:.4f}", flush=True)

    res_df = pd.DataFrame(results_records)
    res_df.to_csv("results/E05.csv", index=False)
    print("\nSaved Experiment E5 results to results/E05.csv", flush=True)
    return res_df


if __name__ == "__main__":
    run_experiment_e5()
