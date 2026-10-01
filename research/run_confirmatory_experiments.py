"""Confirmatory Experiment Execution Engine for Track A (A1-A5) and E06 Appendix.
Executes all approved experiments, computes domain-clustered bootstrap statistics,
and exports structured JSON reports to research/results/.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from urllib.parse import urlparse
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score

# Ensure root directory is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models.baseline import PhishingClassifier
from src.features.regional import LexiconConfig
from src.utils.data_loader import get_feature_matrix, split_train_val_grouped, load_processed_split
from src.utils.domain import extract_registered_domain
from research.stats import paired_cluster_bootstrap, cluster_bootstrap_ci

RESULTS_DIR = "research/results"
os.makedirs(RESULTS_DIR, exist_ok=True)


def calculate_metrics_dict(y_true, y_prob, threshold=0.5):
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    y_pred = (y_prob >= threshold).astype(int)
    
    auc = float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else float("nan")
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    
    # False Positive Rate: FP / (FP + TN)
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0

    return {
        "roc_auc": auc,
        "f1": f1,
        "precision": prec,
        "recall": rec,
        "fpr": fpr,
        "threshold": float(threshold)
    }


def run_a1_experiment():
    print("\n" + "="*60)
    print("  RUNNING EXPERIMENT A1: SHORTCUT SENSITIVITY & LEXICAL MASKING")
    print("="*60, flush=True)

    df_train = load_processed_split("global_train")
    df_test = pd.read_csv("data/benchmark/a1_path_matched.csv")

    df_tr, df_vl = split_train_val_grouped(df_train, val_domain_ratio=0.15, random_state=42)

    # 1. Unmasked Generic LightGBM
    X_tr_unmasked, y_tr, feat_names_unmasked = get_feature_matrix(df_tr, "a1_tr", include_regional=False, include_path_query=True)
    X_vl_unmasked, y_vl, _ = get_feature_matrix(df_vl, "a1_vl", include_regional=False, include_path_query=True)
    X_te_unmasked, y_te, _ = get_feature_matrix(df_test, "a1_te_matched", include_regional=False, include_path_query=True)

    clf_unmasked = PhishingClassifier(model_type="lightgbm", random_state=42)
    clf_unmasked.fit(X_tr_unmasked, y_tr, X_vl_unmasked, y_vl)
    prob_unmasked = clf_unmasked.predict_proba(X_te_unmasked)[:, 1]

    # 2. Masked Generic LightGBM (include_path_query=False)
    X_tr_masked, _, feat_names_masked = get_feature_matrix(df_tr, "a1_tr_masked", include_regional=False, include_path_query=False)
    X_vl_masked, _, _ = get_feature_matrix(df_vl, "a1_vl_masked", include_regional=False, include_path_query=False)
    X_te_masked, _, _ = get_feature_matrix(df_test, "a1_te_matched_masked", include_regional=False, include_path_query=False)

    clf_masked = PhishingClassifier(model_type="lightgbm", random_state=42)
    clf_masked.fit(X_tr_masked, y_tr, X_vl_masked, y_vl)
    prob_masked = clf_masked.predict_proba(X_te_masked)[:, 1]

    # 3. Shortcut-Only LightGBM
    shortcut_cols = ["path_depth", "path_length", "url_length", "is_https", "subdomain_length", "num_slashes"]
    sc_indices = [feat_names_unmasked.index(c) for c in shortcut_cols if c in feat_names_unmasked]
    
    clf_shortcut = PhishingClassifier(model_type="lightgbm", random_state=42)
    clf_shortcut.fit(X_tr_unmasked[:, sc_indices], y_tr, X_vl_unmasked[:, sc_indices], y_vl)
    prob_shortcut = clf_shortcut.predict_proba(X_te_unmasked[:, sc_indices])[:, 1]

    metrics_unmasked = calculate_metrics_dict(y_te, prob_unmasked, threshold=clf_unmasked.optimal_threshold)
    metrics_masked = calculate_metrics_dict(y_te, prob_masked, threshold=clf_masked.optimal_threshold)
    metrics_shortcut = calculate_metrics_dict(y_te, prob_shortcut, threshold=clf_shortcut.optimal_threshold)

    clusters = df_test["registered_domain"].values
    bootstrap_res = paired_cluster_bootstrap(
        y_true=y_te,
        y_pred_a=prob_unmasked,
        y_pred_b=prob_masked,
        clusters=clusters,
        metric_func=roc_auc_score,
        n_bootstraps=1000,
        random_state=42
    )

    results_a1 = {
        "experiment": "A1_shortcut_sensitivity",
        "evaluation_rows": len(df_test),
        "unmasked_generic_lgb": metrics_unmasked,
        "masked_generic_lgb": metrics_masked,
        "shortcut_only_lgb": metrics_shortcut,
        "paired_bootstrap_unmasked_vs_masked": bootstrap_res
    }

    out_file = os.path.join(RESULTS_DIR, "a1_shortcut_results.json")
    with open(out_file, "w") as f:
        json.dump(results_a1, f, indent=2)
    print(f"  [A1] Unmasked AUC: {metrics_unmasked['roc_auc']:.4f} | Masked AUC: {metrics_masked['roc_auc']:.4f} | Shortcut AUC: {metrics_shortcut['roc_auc']:.4f}")
    print(f"  [A1] Delta AUC (Unmasked - Masked): {bootstrap_res['observed_delta']:.4f} (95% CI: [{bootstrap_res['ci_lower']:.4f}, {bootstrap_res['ci_upper']:.4f}])")
    return results_a1


def run_a2_experiment():
    print("\n" + "="*60)
    print("  RUNNING EXPERIMENT A2: REPRESENTATION SENSITIVITY DIAGNOSTIC")
    print("="*60, flush=True)

    df_train = load_processed_split("global_train")
    df_rep = pd.read_csv("data/benchmark/a2_representation.csv")

    df_tr, df_vl = split_train_val_grouped(df_train, val_domain_ratio=0.15, random_state=42)
    X_tr, y_tr, _ = get_feature_matrix(df_tr, "a2_tr", include_regional=False, include_path_query=True)
    X_vl, y_vl, _ = get_feature_matrix(df_vl, "a2_vl", include_regional=False, include_path_query=True)

    clf = PhishingClassifier(model_type="lightgbm", random_state=42)
    clf.fit(X_tr, y_tr, X_vl, y_vl)

    X_rep, _, _ = get_feature_matrix(df_rep, "a2_rep", include_regional=False, include_path_query=True)
    df_rep["prob"] = clf.predict_proba(X_rep)[:, 1]

    # Calculate shifts relative to https_bare
    pivoted = df_rep.pivot(index="domain", columns="variant_type", values="prob")
    shifts = pd.DataFrame(index=pivoted.index)
    base_col = "https_bare"
    for col in pivoted.columns:
        if col != base_col:
            shifts[f"delta_{col}"] = pivoted[col] - pivoted[base_col]

    # Count false-positive flips on top benign homepages (prob jumps from <0.5 to >=0.5)
    flips_www = ((pivoted[base_col] < 0.5) & (pivoted["https_www_bare"] >= 0.5)).sum()
    flips_slash = ((pivoted[base_col] < 0.5) & (pivoted["https_trailing_slash"] >= 0.5)).sum()
    flips_www_slash = ((pivoted[base_col] < 0.5) & (pivoted["https_www_slash"] >= 0.5)).sum()

    results_a2 = {
        "experiment": "A2_representation_sensitivity",
        "evaluated_domains": len(pivoted),
        "mean_base_prob": float(pivoted[base_col].mean()),
        "mean_shifts": {col: float(shifts[col].mean()) for col in shifts.columns},
        "max_shifts": {col: float(shifts[col].max()) for col in shifts.columns},
        "min_shifts": {col: float(shifts[col].min()) for col in shifts.columns},
        "false_positive_flips": {
            "www_flip_count": int(flips_www),
            "slash_flip_count": int(flips_slash),
            "www_slash_flip_count": int(flips_www_slash),
        },
        "per_domain_sample": pivoted.head(10).to_dict()
    }

    out_file = os.path.join(RESULTS_DIR, "a2_representation_results.json")
    with open(out_file, "w") as f:
        json.dump(results_a2, f, indent=2)
    print(f"  [A2] Mean Shift (+www): {results_a2['mean_shifts'].get('delta_https_www_bare', 0.0):+.4f} | (+slash): {results_a2['mean_shifts'].get('delta_https_trailing_slash', 0.0):+.4f}")
    print(f"  [A2] Benign Flips into Phishing: www={flips_www}, slash={flips_slash}, www+slash={flips_www_slash} of 50 domains")
    return results_a2


def run_a3_experiment():
    print("\n" + "="*60)
    print("  RUNNING EXPERIMENT A3: ZERO-KNOWLEDGE BRAND GENERALIZATION")
    print("="*60, flush=True)

    df_train = load_processed_split("global_train")
    df_test = pd.read_csv("data/benchmark/a3_unseen_brands.csv")

    df_tr, df_vl = split_train_val_grouped(df_train, val_domain_ratio=0.15, random_state=42)

    # 1. Generic Baseline LightGBM
    X_tr_gen, y_tr, _ = get_feature_matrix(df_tr, "a3_tr_gen", include_regional=False, include_path_query=True)
    X_vl_gen, y_vl, _ = get_feature_matrix(df_vl, "a3_vl_gen", include_regional=False, include_path_query=True)
    X_te_gen, y_te, _ = get_feature_matrix(df_test, "a3_te_gen", include_regional=False, include_path_query=True)

    clf_gen = PhishingClassifier(model_type="lightgbm", random_state=42)
    clf_gen.fit(X_tr_gen, y_tr, X_vl_gen, y_vl)
    prob_gen = clf_gen.predict_proba(X_te_gen)[:, 1]

    # 2. Regional Model with Held-Out Brands Purged from LexiconConfig
    held_out_brands = ["sbi", "hdfc", "icici", "axis", "pnb", "bob", "kotak", "paytm", "phonepe", "irctc"]
    zk_config = LexiconConfig()
    zk_config.brands = [b for b in zk_config.brands if b not in held_out_brands]
    zk_config.official_domains = {k: v for k, v in zk_config.official_domains.items() if k not in held_out_brands}

    X_tr_zk, _, _ = get_feature_matrix(df_tr, "a3_tr_zk", include_regional=True, include_path_query=True, lexicon_config=zk_config)
    X_vl_zk, _, _ = get_feature_matrix(df_vl, "a3_vl_zk", include_regional=True, include_path_query=True, lexicon_config=zk_config)
    X_te_zk, _, _ = get_feature_matrix(df_test, "a3_te_zk", include_regional=True, include_path_query=True, lexicon_config=zk_config)

    clf_zk = PhishingClassifier(model_type="lightgbm", random_state=42)
    clf_zk.fit(X_tr_zk, y_tr, X_vl_zk, y_vl)
    prob_zk = clf_zk.predict_proba(X_te_zk)[:, 1]

    metrics_gen = calculate_metrics_dict(y_te, prob_gen, threshold=clf_gen.optimal_threshold)
    metrics_zk = calculate_metrics_dict(y_te, prob_zk, threshold=clf_zk.optimal_threshold)

    clusters = df_test["registered_domain"].values
    bootstrap_res = paired_cluster_bootstrap(
        y_true=y_te,
        y_pred_a=prob_zk,
        y_pred_b=prob_gen,
        clusters=clusters,
        metric_func=roc_auc_score,
        n_bootstraps=1000,
        random_state=42
    )

    results_a3 = {
        "experiment": "A3_unseen_brand_generalization",
        "evaluation_rows": len(df_test),
        "generic_baseline": metrics_gen,
        "zero_knowledge_regional": metrics_zk,
        "paired_bootstrap_zk_vs_gen": bootstrap_res
    }

    out_file = os.path.join(RESULTS_DIR, "a3_brand_results.json")
    with open(out_file, "w") as f:
        json.dump(results_a3, f, indent=2)
    print(f"  [A3] Generic Baseline AUC: {metrics_gen['roc_auc']:.4f} | Zero-Knowledge Regional AUC: {metrics_zk['roc_auc']:.4f}")
    print(f"  [A3] Delta AUC (ZK Regional - Generic): {bootstrap_res['observed_delta']:.4f} (95% CI: [{bootstrap_res['ci_lower']:.4f}, {bootstrap_res['ci_upper']:.4f}], Width: {bootstrap_res['interval_width']:.4f})")
    print(f"  [A3] Statistically Significant at alpha=0.05: {bootstrap_res['is_significant']}")
    return results_a3


def run_a4_experiment():
    print("\n" + "="*60)
    print("  RUNNING EXPERIMENT A4: CROSS-DISTRIBUTION & HOSTING CENSUS")
    print("="*60, flush=True)

    df_train = load_processed_split("global_train")
    df_test_in = load_processed_split("global_test")
    df_test_cross_primary = pd.read_csv("data/benchmark/a4_cross_distribution.csv")
    df_test_cross_disjoint = pd.read_csv("data/benchmark/a4_domain_disjoint.csv")
    df_census = pd.read_csv("data/benchmark/hosting_infrastructure_census.csv")

    df_tr, df_vl = split_train_val_grouped(df_train, val_domain_ratio=0.15, random_state=42)
    X_tr, y_tr, _ = get_feature_matrix(df_tr, "a4_tr", include_regional=False, include_path_query=True)
    X_vl, y_vl, _ = get_feature_matrix(df_vl, "a4_vl", include_regional=False, include_path_query=True)

    clf = PhishingClassifier(model_type="lightgbm", random_state=42)
    clf.fit(X_tr, y_tr, X_vl, y_vl)

    # In-Distribution Test
    X_te_in, y_te_in, _ = get_feature_matrix(df_test_in, "a4_te_in", include_regional=False, include_path_query=True)
    prob_in = clf.predict_proba(X_te_in)[:, 1]
    metrics_in = calculate_metrics_dict(y_te_in, prob_in, threshold=clf.optimal_threshold)

    # Primary External Test (Full PhiUSIIL)
    X_te_cross, y_te_cross, _ = get_feature_matrix(df_test_cross_primary, "a4_te_cross_primary", include_regional=False, include_path_query=True)
    prob_cross = clf.predict_proba(X_te_cross)[:, 1]
    metrics_cross_primary = calculate_metrics_dict(y_te_cross, prob_cross, threshold=clf.optimal_threshold)

    # Secondary External Test (Domain-Disjoint)
    X_te_disj, y_te_disj, _ = get_feature_matrix(df_test_cross_disjoint, "a4_te_cross_disjoint", include_regional=False, include_path_query=True)
    prob_disj = clf.predict_proba(X_te_disj)[:, 1]
    metrics_cross_disjoint = calculate_metrics_dict(y_te_disj, prob_disj, threshold=clf.optimal_threshold)

    # Clustered bootstrap for Primary Cross-Distribution Drop
    delta_auc_primary = float(metrics_in["roc_auc"] - metrics_cross_primary["roc_auc"])
    delta_auc_disjoint = float(metrics_in["roc_auc"] - metrics_cross_disjoint["roc_auc"])

    # Hosting Infrastructure Census Evaluation
    X_census, y_census, _ = get_feature_matrix(df_census, "a4_te_census", include_regional=False, include_path_query=True)
    prob_census = clf.predict_proba(X_census)[:, 1]
    metrics_census = calculate_metrics_dict(y_census, prob_census, threshold=clf.optimal_threshold)

    ci_census_auc = cluster_bootstrap_ci(y_true=y_census, y_pred=prob_census, clusters=df_census["registered_domain"].values, metric_func=roc_auc_score)

    results_a4 = {
        "experiment": "A4_cross_distribution_transfer",
        "in_distribution": metrics_in,
        "cross_distribution_primary_phiusiil": metrics_cross_primary,
        "cross_distribution_secondary_disjoint": metrics_cross_disjoint,
        "delta_auc_primary": delta_auc_primary,
        "delta_auc_disjoint": delta_auc_disjoint,
        "exceeds_operational_threshold_0_05": bool(delta_auc_primary > 0.05),
        "hosting_infrastructure_census": {
            "metrics": metrics_census,
            "bootstrap_ci_auc": ci_census_auc,
            "target_category_breakdown": df_census["target_category"].value_counts().to_dict()
        }
    }

    out_file = os.path.join(RESULTS_DIR, "a4_cross_distribution_results.json")
    with open(out_file, "w") as f:
        json.dump(results_a4, f, indent=2)

    print(f"  [A4] In-Distribution AUC: {metrics_in['roc_auc']:.4f} | Primary Cross-Dist AUC: {metrics_cross_primary['roc_auc']:.4f} (Delta: {delta_auc_primary:.4f})")
    print(f"  [A4] Disjoint Cross-Dist AUC: {metrics_cross_disjoint['roc_auc']:.4f} (Delta: {delta_auc_disjoint:.4f})")
    print(f"  [A4] Hosting Census AUC (.in ccTLD): {metrics_census['roc_auc']:.4f} (95% CI: [{ci_census_auc['ci_lower']:.4f}, {ci_census_auc['ci_upper']:.4f}])")
    return results_a4


def run_a5_experiment():
    print("\n" + "="*60)
    print("  RUNNING EXPERIMENT A5: TEMPORAL GENERALIZATION (TIME SPLIT)")
    print("="*60, flush=True)

    df_train = pd.read_csv("data/benchmark/a5_temporal_train.csv")
    df_test = pd.read_csv("data/benchmark/a5_temporal_test.csv")

    df_tr, df_vl = split_train_val_grouped(df_train, val_domain_ratio=0.15, random_state=42)

    X_tr, y_tr, _ = get_feature_matrix(df_tr, "a5_tr", include_regional=False, include_path_query=True)
    X_vl, y_vl, _ = get_feature_matrix(df_vl, "a5_vl", include_regional=False, include_path_query=True)
    X_te, y_te, _ = get_feature_matrix(df_test, "a5_te", include_regional=False, include_path_query=True)

    clf = PhishingClassifier(model_type="lightgbm", random_state=42)
    clf.fit(X_tr, y_tr, X_vl, y_vl)

    prob_vl = clf.predict_proba(X_vl)[:, 1]
    prob_te = clf.predict_proba(X_te)[:, 1]

    metrics_val = calculate_metrics_dict(y_vl, prob_vl, threshold=clf.optimal_threshold)
    metrics_test = calculate_metrics_dict(y_te, prob_te, threshold=clf.optimal_threshold)

    ci_te_auc = cluster_bootstrap_ci(y_true=y_te, y_pred=prob_te, clusters=df_test["registered_domain"].values, metric_func=roc_auc_score)

    delta_temporal_auc = float(metrics_val["roc_auc"] - metrics_test["roc_auc"])

    results_a5 = {
        "experiment": "A5_temporal_generalization",
        "in_time_validation": metrics_val,
        "out_of_time_future_test": metrics_test,
        "delta_auc_temporal_drop": delta_temporal_auc,
        "test_bootstrap_ci_auc": ci_te_auc
    }

    out_file = os.path.join(RESULTS_DIR, "a5_temporal_results.json")
    with open(out_file, "w") as f:
        json.dump(results_a5, f, indent=2)

    print(f"  [A5] In-Time Val AUC (2024): {metrics_val['roc_auc']:.4f} | Out-of-Time Test AUC (Late 2025): {metrics_test['roc_auc']:.4f}")
    print(f"  [A5] Temporal Drop Delta AUC: {delta_temporal_auc:.4f} | Test AUC 95% CI: [{ci_te_auc['ci_lower']:.4f}, {ci_te_auc['ci_upper']:.4f}]")
    return results_a5


def run_e06_appendix():
    print("\n" + "="*60)
    print("  RUNNING APPENDIX E06: ADVERSARIAL PERTURBATION DIAGNOSTIC")
    print("="*60, flush=True)

    from experiments.E06_adversarial import generate_adversarial_dataset, perturb_url_adversarial

    df_test = load_processed_split("global_test")
    df_train = load_processed_split("global_train")

    df_tr, df_vl = split_train_val_grouped(df_train, val_domain_ratio=0.15, random_state=42)
    X_tr, y_tr, _ = get_feature_matrix(df_tr, "e06_tr", include_regional=False, include_path_query=True)
    X_vl, y_vl, _ = get_feature_matrix(df_vl, "e06_vl", include_regional=False, include_path_query=True)

    clf = PhishingClassifier(model_type="lightgbm", random_state=42)
    clf.fit(X_tr, y_tr, X_vl, y_vl)

    # 1. Clean Test Performance
    X_clean, y_clean, _ = get_feature_matrix(df_test, "e06_te_clean", include_regional=False, include_path_query=True)
    prob_clean = clf.predict_proba(X_clean)[:, 1]
    metrics_clean = calculate_metrics_dict(y_clean, prob_clean, threshold=clf.optimal_threshold)

    # 2. Perturbed Test Sets across techniques
    perturbed_metrics = {}
    for tech in ["homoglyph", "token_padding", "subdomain_trick"]:
        df_perturbed = df_test.copy()
        df_perturbed["url"] = [perturb_url_adversarial(u, technique=tech) if l == 1 else u for u, l in zip(df_perturbed["url"], df_perturbed["label"])]
        X_pert, y_pert, _ = get_feature_matrix(df_perturbed, f"e06_pert_{tech}", include_regional=False, include_path_query=True, force_recompute=True)
        prob_pert = clf.predict_proba(X_pert)[:, 1]
        perturbed_metrics[tech] = calculate_metrics_dict(y_pert, prob_pert, threshold=clf.optimal_threshold)

    results_e06 = {
        "appendix": "E06_adversarial_perturbation_diagnostic",
        "clean_baseline": metrics_clean,
        "perturbation_evaluations": perturbed_metrics
    }

    out_file = os.path.join(RESULTS_DIR, "e06_adversarial_results.json")
    with open(out_file, "w") as f:
        json.dump(results_e06, f, indent=2)

    print(f"  [E06 Appendix] Clean Recall (Phish TPR): {metrics_clean['recall']:.4f}")
    for tech, m in perturbed_metrics.items():
        print(f"  [E06 Appendix] Evasion [{tech}]: Recall drops to {m['recall']:.4f} (F1: {m['f1']:.4f})")
    return results_e06


if __name__ == "__main__":
    print("STARTING TRACK A CONFIRMATORY RUNS (A1 - A5, E06 Appendix)...")
    res_a1 = run_a1_experiment()
    res_a2 = run_a2_experiment()
    res_a3 = run_a3_experiment()
    res_a4 = run_a4_experiment()
    res_a5 = run_a5_experiment()
    res_e06 = run_e06_appendix()

    summary_file = os.path.join(RESULTS_DIR, "summary_confirmatory_report.json")
    summary = {
        "a1": res_a1,
        "a2": res_a2,
        "a3": res_a3,
        "a4": res_a4,
        "a5": res_a5,
        "e06_appendix": res_e06
    }
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)
    print("\nALL CONFIRMATORY EXPERIMENTS COMPLETED SUCCESSFULLY!")
    print(f"Summary written to {summary_file}")
