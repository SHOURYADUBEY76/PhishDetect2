"""Streamed Census of Upstream PhreshPhish across all 77 parquet shards.
Reads strictly metadata columns: url, label, target, lang, lang_score, date, sha256.
Computes dataset stats, monthly counts, target/lang distributions, sha256 duplicates,
and applies the independent Indian entity definition.
Runs stratum class-matching and shortcut-only baseline on the PhreshPhish population.
Saves results to audit/results/phreshphish_census_report.json and .txt.
"""

import os
import sys
import json
import time
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import urlparse
from collections import Counter
import pandas as pd
import numpy as np
import pyarrow.parquet as pq
import fsspec
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

# Ensure root directory is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.utils.domain import extract_registered_domain, extract_tld
from audit.independent_indian_entities import (
    match_independent_indian_target,
    is_indian_hosting_tld,
    INDIAN_HOSTING_TLDS
)

BASE_HF_URL = "https://huggingface.co/datasets/phreshphish/phreshphish/resolve/main/data"


def get_all_shard_urls():
    train_urls = [f"{BASE_HF_URL}/train-{i:03d}.parquet" for i in range(56)]
    test_urls = [f"{BASE_HF_URL}/test-{i:03d}.parquet" for i in range(21)]
    return train_urls, test_urls


def fetch_shard_metadata(url, retries=3):
    """Fetch metadata columns from a remote parquet shard via fsspec range requests."""
    fs, path = fsspec.core.url_to_fs(url)
    for attempt in range(retries):
        try:
            with fs.open(path, "rb") as f:
                pf = pq.ParquetFile(f)
                tbl = pf.read(columns=["url", "label", "target", "lang", "lang_score", "date", "sha256"])
                df = tbl.to_pandas()
                df["shard"] = os.path.basename(url)
                return df
        except Exception as e:
            if attempt == retries - 1:
                print(f"Error fetching {url}: {e}", flush=True)
                return pd.DataFrame()
            time.sleep(2)
    return pd.DataFrame()


def run_phreshphish_census():
    print("Initializing PhreshPhish Streamed Census...", flush=True)
    cache_path = "audit/results/phreshphish_metadata_streamed.parquet"
    train_urls, test_urls = get_all_shard_urls()
    all_urls = train_urls + test_urls

    # 1. Fetch train-000 specifically first to record single-shard baseline
    t0 = time.time()
    df_train000 = fetch_shard_metadata(train_urls[0])
    train000_stats = {
        "shard": "train-000.parquet",
        "rows": len(df_train000),
        "target_non_null": int(df_train000["target"].notnull().sum()),
        "target_distribution": {str(k): int(v) for k, v in df_train000["target"].value_counts().head(10).items()},
        "lang_non_null": int(df_train000["lang"].notnull().sum()),
        "lang_distribution": {str(k): int(v) for k, v in df_train000["lang"].value_counts().head(5).items()},
        "distinct_registered_domains": int(df_train000["url"].apply(extract_registered_domain).nunique()),
    }
    print(f"train-000 profiled in {time.time()-t0:.2f}s", flush=True)

    # Check local cache for all shards
    if os.path.exists(cache_path):
        print(f"Loading cached full PhreshPhish metadata from {cache_path}...", flush=True)
        df_all = pd.read_parquet(cache_path)
    else:
        print("Streaming all 77 shards with ThreadPoolExecutor (max_workers=8)...", flush=True)
        dfs = []
        t_start = time.time()
        with ThreadPoolExecutor(max_workers=8) as executor:
            future_to_url = {executor.submit(fetch_shard_metadata, u): u for u in all_urls}
            completed_count = 0
            for future in as_completed(future_to_url):
                df_part = future.result()
                if not df_part.empty:
                    dfs.append(df_part)
                completed_count += 1
                if completed_count % 10 == 0 or completed_count == len(all_urls):
                    print(f"  Streamed {completed_count}/{len(all_urls)} shards ({time.time()-t_start:.1f}s)...", flush=True)

        df_all = pd.concat(dfs, ignore_index=True)
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        df_all.to_parquet(cache_path, index=False)
        print(f"Cached {len(df_all)} records to {cache_path} ({time.time()-t_start:.1f}s total)", flush=True)

    # 2. Compute full census metrics
    total_rows = len(df_all)
    df_all["label_num"] = df_all["label"].apply(lambda x: 1 if str(x).lower().startswith("phish") or x == 1 else 0)
    class_balance = {str(k): int(v) for k, v in df_all["label"].value_counts().items()}
    class_balance_num = {int(k): int(v) for k, v in df_all["label_num"].value_counts().items()}
    sha256_duplicates = int(total_rows - df_all["sha256"].nunique()) if "sha256" in df_all.columns else 0
    unique_urls = int(df_all["url"].nunique())

    print("Extracting registered domains...", flush=True)
    df_all["registered_domain"] = df_all["url"].apply(extract_registered_domain)
    unique_domains = int(df_all["registered_domain"].nunique())

    # Monthly counts
    df_all["date_dt"] = pd.to_datetime(df_all["date"], errors="coerce")
    df_all["month"] = df_all["date_dt"].dt.to_period("M").astype(str)
    monthly_counts = {str(k): int(v) for k, v in df_all["month"].value_counts().sort_index().items()}

    # Target & Lang distribution across all shards
    target_non_null = int(df_all["target"].notnull().sum())
    top_targets = {str(k): int(v) for k, v in df_all["target"].value_counts().head(20).items()}
    lang_non_null = int(df_all["lang"].notnull().sum())
    top_langs = {str(k): int(v) for k, v in df_all["lang"].value_counts().head(15).items()}

    # Path & TLD properties
    print("Computing URL path & structural flags...", flush=True)
    def get_url_path_info(u):
        try:
            p = urlparse(u if "://" in u else f"http://{u}")
            has_p = (len(p.path) > 1) or bool(p.query)
            return has_p, len(p.path), len(u), u.lower().startswith("https://")
        except Exception:
            return False, 0, len(str(u)), False

    path_tuples = [get_url_path_info(u) for u in df_all["url"]]
    df_all["has_path_query"] = [x[0] for x in path_tuples]
    df_all["path_len"] = [x[1] for x in path_tuples]
    df_all["url_len"] = [x[2] for x in path_tuples]
    df_all["is_https"] = [x[3] for x in path_tuples]

    # Hosting TLD analysis
    print("Evaluating Indian hosting TLDs...", flush=True)
    df_all["is_indian_hosting"] = [is_indian_hosting_tld(u) for u in df_all["url"]]

    # Independent Target Analysis
    print("Evaluating Independent Indian Target matches...", flush=True)
    indep_results = [
        match_independent_indian_target(u, t)
        for u, t in zip(df_all["url"], df_all["target"])
    ]
    df_all["is_indian_targeted"] = [x[0] for x in indep_results]
    df_all["matched_indian_entity"] = [x[1] for x in indep_results]
    df_all["match_source"] = [x[2] for x in indep_results]

    # Stratum definition: (has_path_query, length_bin, https_binary)
    df_all["len_bin"] = pd.qcut(df_all["url_len"], q=5, labels=False, duplicates="drop")
    df_all["stratum"] = df_all["has_path_query"].astype(str) + "_" + df_all["len_bin"].astype(str) + "_" + df_all["is_https"].astype(str)

    # Specific breakdowns:
    # (a) Indian-targeted phishing under the independent definition
    df_ind_phish = df_all[(df_all["label_num"] == 1) & (df_all["is_indian_targeted"])].copy()
    ind_phish_count = len(df_ind_phish)
    ind_phish_domains = int(df_ind_phish["registered_domain"].nunique())

    # (b) Phishing hosted on Indian TLDs
    df_in_hosted_phish = df_all[(df_all["label_num"] == 1) & (df_all["is_indian_hosting"])].copy()
    in_hosted_phish_count = len(df_in_hosted_phish)
    in_hosted_phish_domains = int(df_in_hosted_phish["registered_domain"].nunique())

    # (c) Benign URLs with paths on Indian TLDs
    df_in_benign_paths = df_all[(df_all["label_num"] == 0) & (df_all["is_indian_hosting"]) & (df_all["has_path_query"])].copy()
    in_benign_paths_count = len(df_in_benign_paths)
    in_benign_paths_domains = int(df_in_benign_paths["registered_domain"].nunique())

    # 3. Class-matching & Shortcut-only Baseline on PhreshPhish population
    print("Running stratum class-matching and shortcut baseline on PhreshPhish...", flush=True)

    # Subsample for regression if full dataset is large to guarantee fast computation
    sample_size = min(len(df_all), 100000)
    df_eval_sub = df_all.sample(n=sample_size, random_state=42)
    X_shortcut_full = df_eval_sub[["has_path_query", "is_https", "url_len", "path_len"]].astype(float).values
    y_full = df_eval_sub["label_num"].values
    lr_full = LogisticRegression(random_state=42, max_iter=200).fit(X_shortcut_full, y_full)
    auc_full = float(roc_auc_score(y_full, lr_full.predict_proba(X_shortcut_full)[:, 1]))

    # Stratum matching: balanced per stratum
    matched_dfs = []
    strata_report = {}
    for st, group in df_eval_sub.groupby("stratum"):
        g0 = group[group["label_num"] == 0]
        g1 = group[group["label_num"] == 1]
        n_match = min(len(g0), len(g1))
        if n_match > 0:
            s0 = g0.sample(n=n_match, random_state=42)
            s1 = g1.sample(n=n_match, random_state=42)
            matched_dfs.extend([s0, s1])
            strata_report[st] = {
                "raw_legit": len(g0),
                "raw_phish": len(g1),
                "matched_per_class": n_match
            }

    df_matched = pd.concat(matched_dfs, ignore_index=True) if matched_dfs else pd.DataFrame()
    matched_surviving_total = len(df_matched)
    matched_class_dist = {int(k): int(v) for k, v in df_matched["label_num"].value_counts().items()} if not df_matched.empty else {}

    # Shortcut-only baseline AUC on matched population
    if not df_matched.empty:
        X_shortcut_matched = df_matched[["has_path_query", "is_https", "url_len", "path_len"]].astype(float).values
        y_matched = df_matched["label_num"].values
        lr_matched = LogisticRegression(random_state=42, max_iter=200).fit(X_shortcut_matched, y_matched)
        auc_matched = float(roc_auc_score(y_matched, lr_matched.predict_proba(X_shortcut_matched)[:, 1]))
    else:
        auc_matched = 0.5

    # 4. Study-specific Feasibility Criteria Evaluation
    c1_passed = (ind_phish_count >= 500) and (in_benign_paths_count >= 500) and (ind_phish_domains >= 100) and (in_benign_paths_domains >= 100)
    n_months_phish = df_ind_phish["month"].nunique() if not df_ind_phish.empty else 0
    n_months_benign = df_in_benign_paths["month"].nunique() if not df_in_benign_paths.empty else 0
    c2_passed = (n_months_phish >= 3) and (n_months_benign >= 3)

    df_ind_pool = pd.concat([df_ind_phish, df_in_benign_paths], ignore_index=True)
    if not df_ind_pool.empty:
        ind_matched = []
        for st, grp in df_ind_pool.groupby("stratum"):
            g0 = grp[grp["label_num"] == 0]
            g1 = grp[grp["label_num"] == 1]
            n_m = min(len(g0), len(g1))
            if n_m > 0:
                ind_matched.extend([g0.sample(n=n_m, random_state=42), g1.sample(n=n_m, random_state=42)])
        df_ind_matched = pd.concat(ind_matched, ignore_index=True) if ind_matched else pd.DataFrame()
        ind_matched_pairs = len(df_ind_matched) // 2 if not df_ind_matched.empty else 0
        ind_matched_doms_0 = df_ind_matched[df_ind_matched["label_num"] == 0]["registered_domain"].nunique() if not df_ind_matched.empty else 0
        ind_matched_doms_1 = df_ind_matched[df_ind_matched["label_num"] == 1]["registered_domain"].nunique() if not df_ind_matched.empty else 0
        c3_passed = (ind_matched_pairs >= 250) and (ind_matched_doms_0 >= 50) and (ind_matched_doms_1 >= 50)
    else:
        ind_matched_pairs = 0
        ind_matched_doms_0 = 0
        ind_matched_doms_1 = 0
        c3_passed = False

    decision = "Track A ONLY (Track A+R Unfeasible)" if not (c1_passed and c2_passed and c3_passed) else "Track A+R Feasible"

    census_report = {
        "train000_baseline": train000_stats,
        "full_phreshphish_census": {
            "total_shards": len(all_urls),
            "total_rows": total_rows,
            "unique_urls": unique_urls,
            "unique_registered_domains": unique_domains,
            "class_balance_raw": class_balance,
            "class_balance_num": class_balance_num,
            "sha256_duplicates": sha256_duplicates,
            "monthly_counts": monthly_counts,
            "target_non_null": target_non_null,
            "top_20_targets": top_targets,
            "lang_non_null": lang_non_null,
            "top_15_languages": top_langs,
        },
        "regional_subpopulations": {
            "a_indian_targeted_phishing_independent": {
                "count": ind_phish_count,
                "distinct_registered_domains": ind_phish_domains,
                "months_covered": n_months_phish,
                "matched_entities_sample": {str(k): int(v) for k, v in df_ind_phish["matched_indian_entity"].value_counts().head(5).items()}
            },
            "b_phishing_hosted_on_indian_tlds": {
                "count": in_hosted_phish_count,
                "distinct_registered_domains": in_hosted_phish_domains,
                "top_5_hosting_domains": {str(k): int(v) for k, v in df_in_hosted_phish["registered_domain"].value_counts().head(5).items()}
            },
            "c_benign_urls_with_paths_on_indian_tlds": {
                "count": in_benign_paths_count,
                "distinct_registered_domains": in_benign_paths_domains,
                "months_covered": n_months_benign,
                "top_5_domains": {str(k): int(v) for k, v in df_in_benign_paths["registered_domain"].value_counts().head(5).items()}
            }
        },
        "phreshphish_matching_and_shortcut_baseline": {
            "evaluated_sample_size": sample_size,
            "unmatched_shortcut_only_auc": round(auc_full, 4),
            "matched_surviving_total_rows": matched_surviving_total,
            "matched_class_distribution": matched_class_dist,
            "matched_shortcut_only_auc": round(auc_matched, 4),
            "strata_summary": {k: v for k, v in list(strata_report.items())[:8]}
        },
        "study_specific_feasibility_criteria": {
            "c1_sample_size_check": {
                "passed": c1_passed,
                "ind_phish_count": ind_phish_count, "req_phish": 500,
                "ind_benign_count": in_benign_paths_count, "req_benign": 500,
                "ind_phish_domains": ind_phish_domains, "req_phish_dom": 100,
                "ind_benign_domains": in_benign_paths_domains, "req_benign_dom": 100
            },
            "c2_temporal_span_check": {
                "passed": c2_passed,
                "phish_months": n_months_phish, "benign_months": n_months_benign, "req_months": 3
            },
            "c3_matching_survival_check": {
                "passed": c3_passed,
                "surviving_pairs": ind_matched_pairs, "req_pairs": 250,
                "surviving_doms_legit": ind_matched_doms_0, "surviving_doms_phish": ind_matched_doms_1, "req_doms": 50
            },
            "final_track_decision": decision
        }
    }

    os.makedirs("audit/results", exist_ok=True)
    json_path = "audit/results/phreshphish_census_report.json"
    txt_path = "audit/results/phreshphish_census_report.txt"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(census_report, f, indent=2)

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("=== FULL STREAMED PHRESHPHISH CENSUS & FEASIBILITY REPORT ===\n\n")
        f.write("1. Shard train-000 Baseline (Initial Probe):\n")
        f.write(f"  Rows: {train000_stats['rows']}, Domains: {train000_stats['distinct_registered_domains']}\n")
        f.write(f"  Target Non-null: {train000_stats['target_non_null']}, Lang Non-null: {train000_stats['lang_non_null']}\n")
        f.write(f"  Top Targets: {train000_stats['target_distribution']}\n")
        f.write(f"  Top Languages: {train000_stats['lang_distribution']}\n\n")

        f.write("2. Full Streamed Census (All 77 Shards):\n")
        f.write(f"  Total Rows: {total_rows}\n")
        f.write(f"  Unique URLs: {unique_urls}\n")
        f.write(f"  Unique Registered Domains: {unique_domains}\n")
        f.write(f"  Class Balance (Raw): {class_balance}\n")
        f.write(f"  Class Balance (Numeric): {class_balance_num}\n")
        f.write(f"  SHA256 Duplicates: {sha256_duplicates}\n")
        f.write(f"  Target Non-null: {target_non_null} of {total_rows} ({target_non_null/total_rows:.2%})\n")
        f.write(f"  Lang Non-null: {lang_non_null} of {total_rows} ({lang_non_null/total_rows:.2%})\n")
        f.write(f"  Top Targets: {top_targets}\n")
        f.write(f"  Top Languages: {top_langs}\n")
        f.write(f"  Monthly Counts: {monthly_counts}\n\n")

        f.write("3. Regional Subpopulations (Independent Definition):\n")
        f.write(f"  (a) Indian-targeted Phishing: {ind_phish_count} URLs across {ind_phish_domains} distinct domains (Months: {n_months_phish})\n")
        f.write(f"  (b) Phishing Hosted on Indian TLDs: {in_hosted_phish_count} URLs across {in_hosted_phish_domains} distinct domains\n")
        f.write(f"  (c) Benign URLs with Paths on Indian TLDs: {in_benign_paths_count} URLs across {in_benign_paths_domains} distinct domains (Months: {n_months_benign})\n\n")

        f.write("4. PhreshPhish Population Stratum Matching & Shortcut Baseline:\n")
        f.write(f"  Evaluated Sample: {sample_size} rows | Shortcut-Only Baseline AUC: {auc_full:.4f}\n")
        f.write(f"  Stratum-Matched Population: {matched_surviving_total} rows ({matched_class_dist}) | Shortcut-Only Baseline AUC: {auc_matched:.4f}\n\n")

        f.write("5. Study-Specific Feasibility Criteria Evaluation:\n")
        f.write(f"  Criterion 1 (Sample Size & Domains): {'PASSED' if c1_passed else 'FAILED'} (Phish: {ind_phish_count}/{ind_phish_domains} doms; Benign: {in_benign_paths_count}/{in_benign_paths_domains} doms)\n")
        f.write(f"  Criterion 2 (Temporal Span >= 3 months): {'PASSED' if c2_passed else 'FAILED'} (Phish months: {n_months_phish}, Benign months: {n_months_benign})\n")
        f.write(f"  Criterion 3 (Matching Survival >= 250 pairs): {'PASSED' if c3_passed else 'FAILED'} (Pairs: {ind_matched_pairs}, Legit doms: {ind_matched_doms_0}, Phish doms: {ind_matched_doms_1})\n")
        f.write(f"  FINAL DIRECTION DECISION: {decision}\n")

    print(f"Census completed. Reports saved to:\n  - {json_path}\n  - {txt_path}", flush=True)


if __name__ == "__main__":
    run_phreshphish_census()
