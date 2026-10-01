"""BiLSTM Deep Learning Benchmark with Lion Optimizer.
Trains a character-level Bidirectional LSTM using the Lion optimizer on raw URL sequences.
Evaluates on the held-out global test set and compares directly against the LightGBM baseline.
Outputs:
  - Model checkpoint: src/models/bilstm_lion.keras and src/models/bilstm_lion_meta.joblib
  - Evaluation metrics: results/bilstm_lion_eval.json
"""

import os
import sys
import time
import json
import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score, f1_score, precision_score, recall_score,
    confusion_matrix, brier_score_loss
)

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models.bilstm import BiLSTMClassifier
from src.models.baseline import PhishingClassifier
from src.utils.data_loader import load_processed_split, split_train_val_grouped, get_feature_matrix


def compute_expected_calibration_error(y_true: np.ndarray, y_probs: np.ndarray, n_bins: int = 10) -> float:
    """Calculate Expected Calibration Error (ECE) across equal-width probability bins."""
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n = len(y_true)
    for i in range(n_bins):
        bin_lower, bin_upper = bin_boundaries[i], bin_boundaries[i + 1]
        if i == n_bins - 1:
            idx = np.where((y_probs >= bin_lower) & (y_probs <= bin_upper))[0]
        else:
            idx = np.where((y_probs >= bin_lower) & (y_probs < bin_upper))[0]
        if len(idx) > 0:
            bin_acc = np.mean(y_true[idx])
            bin_conf = np.mean(y_probs[idx])
            ece += (len(idx) / n) * np.abs(bin_acc - bin_conf)
    return float(ece)



def run_bilstm_benchmark(
    sample_train_size: int = 12000,
    epochs: int = 4,
    batch_size: int = 256,
    learning_rate: float = 1e-4,
    weight_decay: float = 1e-2,
    seed: int = 42
):
    print("=" * 65)
    print("   CHARACTER-LEVEL BiLSTM BENCHMARK (LION OPTIMIZER)")
    print("=" * 65)
    os.makedirs("results", exist_ok=True)
    os.makedirs("src/models", exist_ok=True)

    # 1. Load Data
    print("\n[1/5] Loading established dataset splits...", flush=True)
    df_train_full = load_processed_split("global_train")
    df_test = load_processed_split("global_test")
    print(f"  Full Global Train: {len(df_train_full)} URLs ({df_train_full['registered_domain'].nunique()} registered domains)")
    print(f"  Global Test Set:   {len(df_test)} URLs ({df_test['registered_domain'].nunique()} registered domains)")

    # To ensure reproducible CPU training within reasonable bounds, sample domains if requested
    if sample_train_size is not None and len(df_train_full) > sample_train_size:
        np.random.seed(seed)
        unique_domains = df_train_full["registered_domain"].unique()
        sampled_domains = np.random.choice(unique_domains, size=min(len(unique_domains), int(sample_train_size * 0.8)), replace=False)
        df_train_sub = df_train_full[df_train_full["registered_domain"].isin(set(sampled_domains))].copy()
        if len(df_train_sub) > sample_train_size:
            df_train_sub = df_train_sub.sample(n=sample_train_size, random_state=seed).reset_index(drop=True)
        print(f"  Sampled Train Subset: {len(df_train_sub)} URLs across {df_train_sub['registered_domain'].nunique()} domains")
    else:
        df_train_sub = df_train_full

    # 2. Grouped Domain Train/Val Split (85/15)
    df_train, df_val = split_train_val_grouped(df_train_sub, val_domain_ratio=0.15, random_state=seed)
    print(f"  Training Split:    {len(df_train)} URLs (Phish: {df_train['label'].sum()})")
    print(f"  Validation Split:  {len(df_val)} URLs (Phish: {df_val['label'].sum()})")

    # 3. Train BiLSTM with Lion Optimizer
    print(f"\n[2/5] Initializing BiLSTM with Lion optimizer (lr={learning_rate}, wd={weight_decay})...", flush=True)
    bilstm = BiLSTMClassifier(
        max_length=120,
        embedding_dim=32,
        lstm_units=64,
        optimizer_type="lion",
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        random_state=seed
    )

    t_train_start = time.time()
    print(f"  Training for up to {epochs} epochs (batch_size={batch_size})...", flush=True)
    bilstm.fit(
        train_urls=df_train["url"].tolist(),
        y_train=df_train["label"].to_numpy(dtype=np.int32),
        val_urls=df_val["url"].tolist(),
        y_val=df_val["label"].to_numpy(dtype=np.int32),
        epochs=epochs,
        batch_size=batch_size,
        verbose=1
    )
    train_time_sec = time.time() - t_train_start
    print(f"  BiLSTM Training completed in {train_time_sec:.2f} seconds.")

    # 4. Save Trained Checkpoint
    checkpoint_path = "src/models/bilstm_lion.keras"
    print(f"\n[3/5] Saving model checkpoint to {checkpoint_path}...", flush=True)
    bilstm.save(checkpoint_path)

    # Verify reload
    bilstm_reloaded = BiLSTMClassifier.load(checkpoint_path)
    print("  Model checkpoint verified and successfully reloaded.")

    # 5. Evaluate BiLSTM on Held-Out Test Set
    print("\n[4/5] Evaluating BiLSTM on Global Test Set (N=7,434)...", flush=True)
    t_inf_start = time.time()
    test_urls = df_test["url"].tolist()
    y_test = df_test["label"].to_numpy(dtype=np.int32)

    probs_bilstm = bilstm_reloaded.predict_proba_urls(test_urls, batch_size=512)[:, 1]
    inf_time_sec = time.time() - t_inf_start
    latency_ms_per_url = (inf_time_sec / len(test_urls)) * 1000.0

    bilstm_opt_thresh = float(bilstm.optimal_threshold)
    preds_bilstm = (probs_bilstm >= bilstm_opt_thresh).astype(int)

    auc_bilstm = roc_auc_score(y_test, probs_bilstm)
    f1_bilstm = f1_score(y_test, preds_bilstm)
    prec_bilstm = precision_score(y_test, preds_bilstm, zero_division=0)
    rec_bilstm = recall_score(y_test, preds_bilstm, zero_division=0)
    brier_bilstm = brier_score_loss(y_test, probs_bilstm)
    ece_bilstm = compute_expected_calibration_error(y_test, probs_bilstm, n_bins=10)
    cm_bilstm = confusion_matrix(y_test, preds_bilstm)
    tn, fp, fn, tp = cm_bilstm.ravel()
    spec_bilstm = float(tn / (tn + fp))

    print(f"  BiLSTM Test Metrics:")
    print(f"    AUC-ROC:      {auc_bilstm:.4f}")
    print(f"    F1-Score:     {f1_bilstm:.4f}")
    print(f"    Precision:    {prec_bilstm:.4f}")
    print(f"    Recall:       {rec_bilstm:.4f}")
    print(f"    Specificity:  {spec_bilstm:.4f}")
    print(f"    Brier Score:  {brier_bilstm:.4f}")
    print(f"    ECE:          {ece_bilstm:.4f}")
    print(f"    Latency:      {latency_ms_per_url:.3f} ms/URL")

    # 6. Train and Evaluate LightGBM Baseline on Identical Split for Fair Comparison
    print("\n[5/5] Training LightGBM Baseline on Identical Data Split for Comparison...", flush=True)
    X_tr_lgbm, y_tr_lgbm, _ = get_feature_matrix(df_train, f"bilstm_comp_tr_seed{seed}", include_regional=False, include_path_query=True)
    X_vl_lgbm, y_vl_lgbm, _ = get_feature_matrix(df_val, f"bilstm_comp_vl_seed{seed}", include_regional=False, include_path_query=True)
    X_te_lgbm, y_te_lgbm, _ = get_feature_matrix(df_test, "bilstm_comp_te", include_regional=False, include_path_query=True)

    clf_lgbm = PhishingClassifier(model_type="lightgbm", random_state=seed)
    t_lgbm_tr = time.time()
    clf_lgbm.fit(X_tr_lgbm, y_tr_lgbm, X_val=X_vl_lgbm, y_val=y_vl_lgbm)
    lgbm_train_time = time.time() - t_lgbm_tr

    probs_lgbm = clf_lgbm.predict_proba(X_te_lgbm)[:, 1]
    lgbm_thresh = float(clf_lgbm.optimal_threshold)
    preds_lgbm = (probs_lgbm >= lgbm_thresh).astype(int)

    auc_lgbm = roc_auc_score(y_test, probs_lgbm)
    f1_lgbm = f1_score(y_test, preds_lgbm)
    prec_lgbm = precision_score(y_test, preds_lgbm, zero_division=0)
    rec_lgbm = recall_score(y_test, preds_lgbm, zero_division=0)
    brier_lgbm = brier_score_loss(y_test, probs_lgbm)
    ece_lgbm = compute_expected_calibration_error(y_test, probs_lgbm, n_bins=10)
    cm_lgbm = confusion_matrix(y_test, preds_lgbm)
    tn_l, fp_l, fn_l, tp_l = cm_lgbm.ravel()
    spec_lgbm = float(tn_l / (tn_l + fp_l))

    # Latency for LightGBM
    t_lgbm_inf = time.time()
    _ = clf_lgbm.predict_proba(X_te_lgbm)
    lgbm_latency_ms = ((time.time() - t_lgbm_inf) / len(test_urls)) * 1000.0

    print(f"  LightGBM Test Metrics:")
    print(f"    AUC-ROC:      {auc_lgbm:.4f}")
    print(f"    F1-Score:     {f1_lgbm:.4f}")
    print(f"    Precision:    {prec_lgbm:.4f}")
    print(f"    Recall:       {rec_lgbm:.4f}")
    print(f"    Specificity:  {spec_lgbm:.4f}")
    print(f"    Brier Score:  {brier_lgbm:.4f}")
    print(f"    ECE:          {ece_lgbm:.4f}")
    print(f"    Latency:      {lgbm_latency_ms:.3f} ms/URL")

    # 7. Save Comprehensive Comparative Results
    summary = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "dataset": {
            "train_total_urls": len(df_train),
            "train_phishing_urls": int(df_train["label"].sum()),
            "val_total_urls": len(df_val),
            "val_phishing_urls": int(df_val["label"].sum()),
            "test_total_urls": len(df_test),
            "test_phishing_urls": int(df_test["label"].sum()),
            "test_benign_urls": int(len(df_test) - df_test["label"].sum())
        },
        "bilstm_lion": {
            "model_type": "BiLSTM",
            "optimizer": "Lion",
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "embedding_dim": 32,
            "lstm_units": 64,
            "epochs_run": len(bilstm.history.get("loss", [])),
            "training_time_sec": round(train_time_sec, 2),
            "optimal_threshold": round(bilstm_opt_thresh, 4),
            "test_metrics": {
                "auc": round(float(auc_bilstm), 4),
                "f1": round(float(f1_bilstm), 4),
                "precision": round(float(prec_bilstm), 4),
                "recall": round(float(rec_bilstm), 4),
                "specificity": round(float(spec_bilstm), 4),
                "brier_score": round(float(brier_bilstm), 4),
                "ece": round(float(ece_bilstm), 4),
                "latency_ms_per_url": round(float(latency_ms_per_url), 4),
                "confusion_matrix": {
                    "tn": int(tn),
                    "fp": int(fp),
                    "fn": int(fn),
                    "tp": int(tp)
                }
            },
            "history": {k: [round(float(v), 5) for v in vals] for k, vals in bilstm.history.items()}
        },
        "lightgbm_baseline": {
            "model_type": "LightGBM",
            "training_time_sec": round(lgbm_train_time, 2),
            "optimal_threshold": round(lgbm_thresh, 4),
            "test_metrics": {
                "auc": round(float(auc_lgbm), 4),
                "f1": round(float(f1_lgbm), 4),
                "precision": round(float(prec_lgbm), 4),
                "recall": round(float(rec_lgbm), 4),
                "specificity": round(float(spec_lgbm), 4),
                "brier_score": round(float(brier_lgbm), 4),
                "ece": round(float(ece_lgbm), 4),
                "latency_ms_per_url": round(float(lgbm_latency_ms), 4),
                "confusion_matrix": {
                    "tn": int(tn_l),
                    "fp": int(fp_l),
                    "fn": int(fn_l),
                    "tp": int(tp_l)
                }
            }
        },
        "comparison_delta": {
            "delta_auc_lgbm_minus_bilstm": round(float(auc_lgbm - auc_bilstm), 4),
            "delta_f1_lgbm_minus_bilstm": round(float(f1_lgbm - f1_bilstm), 4),
            "latency_ratio_bilstm_to_lgbm": round(float(latency_ms_per_url / max(lgbm_latency_ms, 1e-6)), 2)
        }
    }

    out_file = "results/bilstm_lion_eval.json"
    with open(out_file, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved complete benchmark summary to {out_file}.")
    print("=" * 65)
    return summary


if __name__ == "__main__":
    run_bilstm_benchmark()
