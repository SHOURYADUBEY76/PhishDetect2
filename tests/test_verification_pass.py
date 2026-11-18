"""Verification Pass Test Suite.
Validates A4 dataset integrity, A3 benchmark validity, hosting census accounting, and bootstrap functionality.
"""

import os
import re
import pytest
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, f1_score

from src.utils.domain import extract_registered_domain
from src.features.regional import LexiconConfig
from src.utils.data_loader import get_feature_matrix, FEATURE_CACHE_DIR
from research.stats import paired_cluster_bootstrap, cluster_bootstrap_ci


def test_a4_dataset_integrity():
    """Verify A4 primary (N=3520) and secondary domain-disjoint (N=2888) benchmarks."""
    p_primary = "data/benchmark/a4_cross_distribution.csv"
    p_disjoint = "data/benchmark/a4_domain_disjoint.csv"
    
    assert os.path.exists(p_primary), "A4 primary benchmark missing"
    assert os.path.exists(p_disjoint), "A4 secondary domain-disjoint benchmark missing"
    
    df_prim = pd.read_csv(p_primary)
    assert len(df_prim) == 3520, f"Expected 3520 rows, got {len(df_prim)}"
    assert (df_prim["label_num"] == 0).sum() == 1760
    assert (df_prim["label_num"] == 1).sum() == 1760
    assert df_prim["registered_domain"].nunique() == 2838

    df_disj = pd.read_csv(p_disjoint)
    assert len(df_disj) == 2888, f"Expected 2888 rows, got {len(df_disj)}"
    assert (df_disj["label_num"] == 0).sum() == 1701
    assert (df_disj["label_num"] == 1).sum() == 1187
    assert df_disj["registered_domain"].nunique() == 2746

    # Verify zero domain overlap with any training dataset
    g_tr = pd.read_csv("data/processed/global_train.csv")
    a5_tr = pd.read_csv("data/benchmark/a5_temporal_train.csv")
    i_tr = pd.read_csv("data/processed/indian_train.csv")
    
    all_train_doms = set(g_tr["registered_domain"].dropna()).union(
        set(a5_tr["registered_domain"].dropna())
    ).union(set(i_tr["registered_domain"].dropna()))
    
    disj_doms = set(df_disj["registered_domain"])
    overlap = disj_doms.intersection(all_train_doms)
    assert len(overlap) == 0, f"Expected 0 overlap in domain-disjoint benchmark, found {len(overlap)}: {overlap}"


def test_a3_benchmark_validity():
    """Verify A3 brand benchmark integrity, non-polysemic attribution, and cache isolation."""
    p_a3 = "data/benchmark/a3_unseen_brands.csv"
    assert os.path.exists(p_a3), "A3 benchmark missing"
    
    df_a3 = pd.read_csv(p_a3)
    assert len(df_a3) == 658, f"Expected 658 rows, got {len(df_a3)}"
    assert (df_a3["label_num"] == 0).sum() == 329
    assert (df_a3["label_num"] == 1).sum() == 329

    # Verify no synthetic regex patterns exist
    synth_re = re.compile(r"sbi-kyc-update-portal|hdfc-reward-points|icici-netbanking|paytm-cashback|phonepe-upi|uidai-aadhaar|pan-card-update|irctc-ticket|axis-bank-netbanking|pnb-online-alert", re.IGNORECASE)
    for u in df_a3["url"]:
        assert not synth_re.search(u), f"Found synthetic URL in A3: {u}"

    # Verify no unvalidated standalone 'bob' or 'axis' substrings exist without compound naming or valid entity
    for _, row in df_a3[df_a3["label_num"] == 1].iterrows():
        b = row["brand_entity"]
        u = row["url"].lower()
        if b == "bob":
            assert "bankofbaroda" in u, f"Unverified bob URL: {u}"
        elif b == "axis":
            assert any(t in u for t in ["axisbank", "axiscard", "axis-bank"]), f"Unverified axis URL: {u}"

    # Test cache isolation in data_loader
    df_test_sample = pd.DataFrame({
        "url": ["https://sbi.co.in/portal", "https://bankofbaroda.in/login"],
        "label": [0, 0]
    })
    
    cfg1 = LexiconConfig()
    cfg2 = LexiconConfig(brands=["sbi"])
    
    X1, y1, _ = get_feature_matrix(df_test_sample, "test_cache_iso", include_regional=True, lexicon_config=cfg1, force_recompute=True)
    X2, y2, _ = get_feature_matrix(df_test_sample, "test_cache_iso", include_regional=True, lexicon_config=cfg2, force_recompute=True)
    
    # Check that different cache files were generated
    cache_files = os.listdir(FEATURE_CACHE_DIR)
    matching_files = [f for f in cache_files if f.startswith("test_cache_iso")]
    assert len(matching_files) >= 2, f"Expected at least 2 distinct cache files for different LexiconConfigs, got {matching_files}"


def test_hosting_infrastructure_census():
    """Verify dedicated hosting census accounting, class breakdown, and zero cross-class domain overlap."""
    p_census = "data/benchmark/hosting_infrastructure_census.csv"
    assert os.path.exists(p_census), "Hosting census missing"
    
    df = pd.read_csv(p_census)
    assert len(df) == 1682, f"Expected 1682 rows, got {len(df)}"
    
    n_phish = (df["label_num"] == 1).sum()
    n_benign = (df["label_num"] == 0).sum()
    assert n_phish == 1205, f"Expected 1205 phishing rows, got {n_phish}"
    assert n_benign == 477, f"Expected 477 benign rows, got {n_benign}"
    
    # Check domain overlap across classes
    d_benign = set(df[df["label_num"] == 0]["registered_domain"])
    d_phish = set(df[df["label_num"] == 1]["registered_domain"])
    assert len(d_benign.intersection(d_phish)) == 0, "Cross-class domain overlap should be 0"
    assert len(d_benign) == 263
    assert len(d_phish) == 496

    # Verify target category accounting on phishing rows
    phish_df = df[df["label_num"] == 1]
    vc = phish_df["target_category"].value_counts().to_dict()
    assert vc.get("Identified Foreign / Global Target", 0) == 713
    assert vc.get("Ambiguous / Unspecified Target", 0) == 412
    assert vc.get("Domestic Indian Target", 0) == 80


def test_bootstrap_uncertainty_handling():
    """Verify paired cluster-bootstrap handles stratified sampling and tracks invalid replicates."""
    rng = np.random.default_rng(42)
    N = 200
    clusters = np.repeat([f"dom{i}.com" for i in range(20)], 10)
    y_true = np.array([0]*100 + [1]*100)
    y_pred_a = rng.uniform(0.1, 0.9, size=N)
    y_pred_b = rng.uniform(0.1, 0.9, size=N)

    res = paired_cluster_bootstrap(
        y_true=y_true,
        y_pred_a=y_pred_a,
        y_pred_b=y_pred_b,
        clusters=clusters,
        metric_func=roc_auc_score,
        n_bootstraps=100,
        random_state=42
    )

    assert "observed_delta" in res
    assert "ci_lower" in res
    assert "ci_upper" in res
    assert "interval_width" in res
    assert res["n_valid_replicates"] + res["n_invalid_replicates"] == 100
    assert not np.isnan(res["ci_lower"])
    assert not np.isnan(res["ci_upper"])
    assert res["ci_upper"] >= res["ci_lower"]


def test_multi_tenant_and_counterfactual_innovations():
    """Verify multi-tenant cloud awareness and actionable counterfactual explanations."""
    from src.utils.domain import get_multi_tenant_info
    from src.explainer.explainer import PhishingExplainer

    # Test Cloudflare Worker multi-tenant parsing
    is_multi, platform, tenant = get_multi_tenant_info("https://sbi-kyc-update.workers.dev/verify")
    assert is_multi is True
    assert platform == "Cloudflare Workers"
    assert tenant == "sbi-kyc-update"

    # Test Vercel multi-tenant parsing
    is_multi, platform, tenant = get_multi_tenant_info("https://paytm-cashback.vercel.app/claim")
    assert is_multi is True
    assert platform == "Vercel"
    assert tenant == "paytm-cashback"

    # Test standard independent domain (not multi-tenant)
    is_multi, platform, tenant = get_multi_tenant_info("https://onlinesbi.sbi/portal")
    assert is_multi is False
    assert platform is None
    assert tenant is None

    # Test counterfactual generation
    explainer = PhishingExplainer()
    cf_phish = explainer.explain_counterfactual(
        url="http://sbi-kyc-update.top/login.php",
        features={"brand_tld_misalignment": 1.0, "is_http": 1.0},
        risk_score=0.985,
        is_phishing=True
    )
    assert "SBI" in cf_phish or "sbi" in cf_phish.lower()
    assert "predicted threat would drop" in cf_phish

    cf_safe = explainer.explain_counterfactual(
        url="https://onlinesbi.sbi/portal",
        features={"is_https": 1.0},
        risk_score=0.012,
        is_phishing=False
    )
    assert "well-formed" in cf_safe or "low" in cf_safe.lower()
