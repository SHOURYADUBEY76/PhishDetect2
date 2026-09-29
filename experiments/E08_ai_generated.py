"""Experiment E8: Zero-Shot Detection of Held-Out AI-Generated Phishing URLs.
Evaluates model detection performance on synthetic phishing URLs generated to mimic LLM evasion tactics.
Enforces zero domain leakage against training data and records data provenance.
Outputs results/E08.csv.
"""

import os
import sys
import pandas as pd
import numpy as np

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models.baseline import PhishingClassifier
from src.utils.data_loader import load_processed_split, split_train_val_grouped, get_feature_matrix
from scripts.generate_ai_urls import generate_ai_phishing_urls


SEEDS = [42, 43, 44, 45, 46]


def run_experiment_e8():
    os.makedirs("results", exist_ok=True)
    print("\n=======================================================", flush=True)
    print("   EXPERIMENT E8: HELD-OUT AI-GENERATED PHISHING TEST", flush=True)
    print("=======================================================", flush=True)

    df_ai_phish = generate_ai_phishing_urls()
    provenance = df_ai_phish["provenance"].iloc[0] if "provenance" in df_ai_phish.columns else "unknown"
    is_standin = bool(df_ai_phish["is_standin"].iloc[0]) if "is_standin" in df_ai_phish.columns else False

    df_gtrain = load_processed_split("global_train")
    df_itrain = load_processed_split("indian_train")
    combined_train = pd.concat([df_gtrain, df_itrain], ignore_index=True)

    train_domains = set(combined_train["registered_domain"].unique())
    ai_domains = set(df_ai_phish["registered_domain"].unique())
    overlap = len(ai_domains.intersection(train_domains))
    print(f"Data Provenance: {provenance} (Is Stand-in: {is_standin})", flush=True)
    print(f"Registered Domain Overlap (AI URLs vs Training Data): {overlap} (Strict Zero Leakage: {overlap == 0})", flush=True)

    # Balance with legitimate Indian domains
    df_tranco_legit = pd.read_csv("data/raw/tranco_indian_legit.csv")
    df_legit_clean = df_tranco_legit[~df_tranco_legit["registered_domain"].isin(train_domains)].copy()

    n_samples = min(len(df_ai_phish), len(df_legit_clean))
    ai_eval = pd.concat([
        df_ai_phish.sample(n=n_samples, random_state=42)[["url", "label", "date"]],
        df_legit_clean.sample(n=n_samples, random_state=42)[["url", "label", "date"]]
    ], ignore_index=True).sample(frac=1.0, random_state=42).reset_index(drop=True)

    print(f"Constructed Balanced AI Test Set: {len(ai_eval)} samples (Phish: {ai_eval['label'].sum()}, Legit: {len(ai_eval)-ai_eval['label'].sum()})", flush=True)

    results_records = []

    for mode, inc_reg in [("conventional_baseline", False), ("region_aware_model", True)]:
        print(f"\n--- Testing Detection on AI Phishing with {mode.upper()} ---", flush=True)
        seed_metrics = []

        for seed in SEEDS:
            df_tr, df_vl = split_train_val_grouped(combined_train, val_domain_ratio=0.15, random_state=seed)

            X_tr, y_tr, _ = get_feature_matrix(df_tr, f"e8_tr_{mode}_seed{seed}", include_regional=inc_reg, include_path_query=True)
            X_vl, y_vl, _ = get_feature_matrix(df_vl, f"e8_vl_{mode}_seed{seed}", include_regional=inc_reg, include_path_query=True)
            X_te, y_te, _ = get_feature_matrix(ai_eval, f"e8_te_{mode}", include_regional=inc_reg, include_path_query=True)

            clf = PhishingClassifier(model_type="lightgbm", random_state=seed)
            clf.fit(X_tr, y_tr, X_val=X_vl, y_val=y_vl)

            metrics = clf.evaluate(X_te, y_te)
            seed_metrics.append(metrics)

        avg_res = {
            "experiment": "E08_ai_generated",
            "model_mode": mode,
            "data_provenance": provenance,
            "is_standin": is_standin,
            "regional_features": inc_reg,
            "sample_count": len(ai_eval),
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
        print(f"  Result [{mode}]: F1={avg_res['f1_mean']:.4f}±{avg_res['f1_std']:.4f} | Recall={avg_res['recall_mean']:.4f}±{avg_res['recall_std']:.4f} | Size={avg_res['model_size_mb']:.2f}MB", flush=True)

    res_df = pd.DataFrame(results_records)
    res_df.to_csv("results/E08.csv", index=False)
    print("\nSaved Experiment E8 results to results/E08.csv", flush=True)
    return res_df


if __name__ == "__main__":
    run_experiment_e8()
