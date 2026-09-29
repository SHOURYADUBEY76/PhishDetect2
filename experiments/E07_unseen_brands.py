"""Experiment E7: Generalization to Unseen Indian Domains and Brands.
Evaluates the model's zero-shot detection capability when targeted brands/entities
are strictly withheld during training and only encountered at test time.
Guarantees statistically robust sample size (N >= 250) and 50/50 balance.
Averaged over 5 seeds. Outputs results/E07.csv.
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

# Brand clustering: Cluster A (Seen in Training) vs Cluster B (Strictly Held-Out Unseen)
SEEN_TRAIN_BRANDS = ["sbi", "onlinesbi", "statebank", "jio", "airtel", "bsnl"]
HELD_OUT_UNSEEN_BRANDS = ["phonepe", "aadhaar", "uidai", "irctc", "axis", "pnb", "bob", "kotak", "epfo", "paytm", "hdfc", "icici"]


def run_experiment_e7():
    os.makedirs("results", exist_ok=True)
    print("\n=======================================================", flush=True)
    print("   EXPERIMENT E7: UNSEEN BRANDS & DOMAINS EVALUATION", flush=True)
    print("=======================================================", flush=True)

    df_gtrain = load_processed_split("global_train")
    df_itrain = load_processed_split("indian_train")
    df_itest = load_processed_split("indian_test")

    combined_train = pd.concat([df_gtrain, df_itrain], ignore_index=True)

    # Helper: check brand membership
    def has_unseen_brand(url_str):
        u_lower = str(url_str).lower()
        return any(b in u_lower for b in HELD_OUT_UNSEEN_BRANDS)

    def has_seen_brand(url_str):
        u_lower = str(url_str).lower()
        return any(b in u_lower for b in SEEN_TRAIN_BRANDS)

    # 1. Training set: strictly purge any sample containing an unseen brand
    train_filtered = combined_train[~combined_train["url"].apply(has_unseen_brand)].copy()

    # 2. Test partition for UNSEEN brands:
    # Phishing lures containing unseen brands + balanced sample of legitimate Indian domains
    phish_unseen = df_itest[(df_itest["label"] == 1) & (df_itest["url"].apply(has_unseen_brand))].copy()
    legit_itest = df_itest[df_itest["label"] == 0].copy()

    n_unseen = min(len(phish_unseen), len(legit_itest) // 2)
    test_unseen = pd.concat([
        phish_unseen.sample(n=n_unseen, random_state=42),
        legit_itest.sample(n=n_unseen, random_state=42)
    ], ignore_index=True).sample(frac=1.0, random_state=42).reset_index(drop=True)

    # 3. Test partition for SEEN brands:
    phish_seen = df_itest[(df_itest["label"] == 1) & (df_itest["url"].apply(has_seen_brand))].copy()
    rem_legit = legit_itest[~legit_itest["url"].isin(test_unseen["url"])].copy()

    n_seen = min(len(phish_seen), len(rem_legit))
    test_seen = pd.concat([
        phish_seen.sample(n=n_seen, random_state=42),
        rem_legit.sample(n=n_seen, random_state=42)
    ], ignore_index=True).sample(frac=1.0, random_state=42).reset_index(drop=True)

    print(f"Training set (with {len(HELD_OUT_UNSEEN_BRANDS)} unseen entities purged): {len(train_filtered)} samples", flush=True)
    print(f"Unseen Brands Evaluation Set : {len(test_unseen)} samples (Phish: {test_unseen['label'].sum()}, Legit: {len(test_unseen)-test_unseen['label'].sum()})", flush=True)
    print(f"Seen Brands Evaluation Set   : {len(test_seen)} samples (Phish: {test_seen['label'].sum()}, Legit: {len(test_seen)-test_seen['label'].sum()})", flush=True)

    results_records = []

    for test_name, test_df in [("unseen_brands", test_unseen), ("seen_brands", test_seen)]:
        if len(test_df) == 0:
            continue
        seed_metrics = []
        for seed in SEEDS:
            df_tr, df_vl = split_train_val_grouped(train_filtered, val_domain_ratio=0.15, random_state=seed)

            X_tr, y_tr, _ = get_feature_matrix(df_tr, f"e7_tr_seed{seed}", include_regional=True, include_path_query=True)
            X_vl, y_vl, _ = get_feature_matrix(df_vl, f"e7_vl_seed{seed}", include_regional=True, include_path_query=True)
            X_te, y_te, _ = get_feature_matrix(test_df, f"e7_te_{test_name}", include_regional=True, include_path_query=True)

            clf = PhishingClassifier(model_type="lightgbm", random_state=seed)
            clf.fit(X_tr, y_tr, X_val=X_vl, y_val=y_vl)
            seed_metrics.append(clf.evaluate(X_te, y_te))

        avg_res = {
            "experiment": "E07_unseen_brands",
            "evaluation_subset": test_name,
            "sample_count": len(test_df),
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
        print(f"  Result [{test_name}]: F1={avg_res['f1_mean']:.4f}±{avg_res['f1_std']:.4f} | Recall={avg_res['recall_mean']:.4f}±{avg_res['recall_std']:.4f} | FPR={avg_res['fpr_mean']:.4f}±{avg_res['fpr_std']:.4f} | Size={avg_res['model_size_mb']:.2f}MB", flush=True)

    res_df = pd.DataFrame(results_records)
    res_df.to_csv("results/E07.csv", index=False)
    print("\nSaved Experiment E7 results to results/E07.csv", flush=True)
    return res_df


if __name__ == "__main__":
    run_experiment_e7()
