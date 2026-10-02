"""Comprehensive Data Census Script for all raw and processed datasets.
Computes dataset sizes, unique domains, class balances, train/test leaks,
path/protocol distributions, TLD distributions, and brand presence.
Saves results to audit/results/data_census_report.json and data_census_report.txt.
"""

import os
import sys
import json
import glob
import re
from urllib.parse import urlparse
from collections import Counter
import pandas as pd
import numpy as np

# Ensure root directory is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.utils.domain import extract_registered_domain, extract_tld
from src.features.regional import INDIAN_BRANDS, INDIAN_TLDS

# Independent list of Indian financial and critical public entities
INDEPENDENT_INDIAN_TARGETS = {
    "state bank of india", "sbi", "hdfc", "icici", "punjab national bank", "pnb",
    "bank of baroda", "bob", "axis bank", "axis", "kotak", "canara bank", "union bank",
    "reserve bank of india", "rbi", "national payments corporation", "npci", "upi",
    "paytm", "phonepe", "google pay", "gpay", "bhim", "uidai", "aadhaar",
    "income tax department", "pan card", "nsdl", "utiitsl", "epfo", "irctc",
    "india post", "digilocker", "parivahan", "voter id", "eci"
}

INDEPENDENT_INDIAN_TOKENS = {
    "sbi", "onlinesbi", "hdfc", "hdfcbank", "icici", "icicibank", "pnb", "bob",
    "bankofbaroda", "axisbank", "kotak", "canarabank", "unionbank", "rbi", "npci",
    "upi", "paytm", "phonepe", "gpay", "bhim", "uidai", "aadhaar", "incometax",
    "nsdl", "utiitsl", "epfo", "epfindia", "irctc", "indiapost", "digilocker", "parivahan"
}


def analyze_dataset(file_path):
    """Analyze a single CSV dataset file."""
    df = pd.read_csv(file_path)
    filename = os.path.basename(file_path)
    
    if "url" not in df.columns:
        return {"file": file_path, "error": "No 'url' column"}

    total_rows = len(df)
    unique_urls = int(df["url"].nunique())
    
    # Registered domain
    if "registered_domain" in df.columns:
        domains = df["registered_domain"].fillna("")
    else:
        domains = df["url"].apply(extract_registered_domain)
    unique_domains = int(domains.nunique())
    top_5_domains = domains.value_counts().head(5).to_dict()
    top_1_domain_share = float(domains.value_counts().iloc[0] / total_rows) if total_rows > 0 else 0.0

    # Labels
    has_label = "label" in df.columns
    label_dist = df["label"].value_counts().to_dict() if has_label else {}
    
    # Path & Query presence
    def get_path_query_info(u):
        try:
            p = urlparse(u if "://" in u else f"http://{u}")
            has_pq = (len(p.path) > 1) or bool(p.query)
            is_https = u.lower().startswith("https://")
            is_http = u.lower().startswith("http://")
            return has_pq, is_https, is_http, len(u), len(p.netloc), len(p.path)
        except Exception:
            return False, False, False, len(str(u)), 0, 0

    pq_info = [get_path_query_info(u) for u in df["url"]]
    has_pq_arr = [x[0] for x in pq_info]
    is_https_arr = [x[1] for x in pq_info]
    is_http_arr = [x[2] for x in pq_info]
    url_lens = [x[3] for x in pq_info]
    path_lens = [x[5] for x in pq_info]

    # Breakdown by label if available
    path_by_class = {}
    https_by_class = {}
    if has_label:
        for lbl in sorted(df["label"].unique()):
            mask = (df["label"] == lbl).values
            path_by_class[int(lbl)] = {
                "count": int(mask.sum()),
                "has_path_query_ratio": float(np.mean([has_pq_arr[i] for i in range(len(mask)) if mask[i]])),
                "https_ratio": float(np.mean([is_https_arr[i] for i in range(len(mask)) if mask[i]])),
                "mean_path_length": float(np.mean([path_lens[i] for i in range(len(mask)) if mask[i]])),
            }

    # TLD distribution
    tlds = df["url"].apply(extract_tld)
    top_5_tlds = tlds.value_counts().head(5).to_dict()
    indian_tld_ratio = float(tlds.isin(INDIAN_TLDS).mean())

    # Brand matching
    def match_brands(u, brand_set):
        tokens = re.findall(r"[a-zA-Z0-9]+", str(u).lower())
        return any(t in brand_set for t in tokens)

    repo_brand_matches = int(df["url"].apply(lambda u: match_brands(u, set(INDIAN_BRANDS))).sum())
    repo_brand_ratio = float(repo_brand_matches / total_rows) if total_rows > 0 else 0.0

    indep_brand_matches = int(df["url"].apply(lambda u: match_brands(u, INDEPENDENT_INDIAN_TOKENS)).sum())
    indep_brand_ratio = float(indep_brand_matches / total_rows) if total_rows > 0 else 0.0

    # Date unique counts if available
    date_info = {}
    if "date" in df.columns:
        date_info = {
            "unique_dates": int(df["date"].nunique()),
            "min_date": str(df["date"].min()),
            "max_date": str(df["date"].max()),
        }

    return {
        "file": file_path.replace("\\", "/"),
        "total_rows": total_rows,
        "unique_urls": unique_urls,
        "unique_domains": unique_domains,
        "top_5_domains": {str(k): int(v) for k, v in top_5_domains.items()},
        "top_1_domain_share": top_1_domain_share,
        "label_distribution": {int(k): int(v) for k, v in label_dist.items()},
        "overall_path_query_ratio": float(np.mean(has_pq_arr)),
        "overall_https_ratio": float(np.mean(is_https_arr)),
        "mean_url_length": float(np.mean(url_lens)),
        "path_by_class": path_by_class,
        "top_5_tlds": {str(k): int(v) for k, v in top_5_tlds.items()},
        "indian_tld_ratio": indian_tld_ratio,
        "repo_brand_match_count": repo_brand_matches,
        "repo_brand_match_ratio": repo_brand_ratio,
        "independent_brand_match_count": indep_brand_matches,
        "independent_brand_match_ratio": indep_brand_ratio,
        "date_info": date_info
    }


def compute_leakages():
    """Compute exact URL and domain overlaps across train and test partitions."""
    splits = {
        "global_train": pd.read_csv("data/processed/global_train.csv"),
        "global_test": pd.read_csv("data/processed/global_test.csv"),
        "indian_train": pd.read_csv("data/processed/indian_train.csv"),
        "indian_test": pd.read_csv("data/processed/indian_test.csv"),
        "phiusiil_train": pd.read_csv("data/processed/phiusiil_train.csv"),
        "phiusiil_test": pd.read_csv("data/processed/phiusiil_test.csv"),
    }

    pairs = [
        ("global_train", "global_test"),
        ("indian_train", "indian_test"),
        ("phiusiil_train", "phiusiil_test"),
        ("global_train", "indian_test"),
        ("global_train", "phiusiil_test"),
    ]

    leakage_results = {}
    for s1, s2 in pairs:
        df1 = splits[s1]
        df2 = splits[s2]
        
        urls1 = set(df1["url"])
        urls2 = set(df2["url"])
        url_overlap = len(urls1.intersection(urls2))

        doms1 = set(df1["registered_domain"]) if "registered_domain" in df1.columns else set(df1["url"].apply(extract_registered_domain))
        doms2 = set(df2["registered_domain"]) if "registered_domain" in df2.columns else set(df2["url"].apply(extract_registered_domain))
        dom_overlap = len(doms1.intersection(doms2))

        leakage_results[f"{s1}_vs_{s2}"] = {
            "url_overlap": url_overlap,
            "registered_domain_overlap": dom_overlap,
            "domain_overlap_ratio_of_test": float(dom_overlap / len(doms2)) if len(doms2) > 0 else 0.0
        }

    return leakage_results


def main():
    print("Running Full Data Census across all local datasets...")
    target_files = sorted(glob.glob("data/**/*.csv", recursive=True))
    # Exclude tranco_cache large files
    target_files = [f for f in target_files if "tranco_cache" not in f]

    dataset_summaries = {}
    for f in target_files:
        print(f"  Auditing: {f}...", flush=True)
        summary = analyze_dataset(f)
        dataset_summaries[f.replace("\\", "/")] = summary

    leakages = compute_leakages()

    census_report = {
        "census_version": "1.0.0",
        "datasets": dataset_summaries,
        "train_test_leakage_audit": leakages,
    }

    os.makedirs("audit/results", exist_ok=True)
    json_path = "audit/results/data_census_report.json"
    txt_path = "audit/results/data_census_report.txt"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(census_report, f, indent=2)

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("=======================================================\n")
        f.write("             FULL DATA CENSUS REPORT                   \n")
        f.write("=======================================================\n\n")

        f.write("--- 1. DATASET SUMMARIES ---\n")
        for fpath, d in dataset_summaries.items():
            f.write(f"\n[DATASET] {fpath}\n")
            f.write(f"  Rows: {d.get('total_rows')}, Unique URLs: {d.get('unique_urls')}, Unique Domains: {d.get('unique_domains')}\n")
            f.write(f"  Labels: {d.get('label_distribution')}\n")
            f.write(f"  Path/Query Ratio: {d.get('overall_path_query_ratio'):.4f}, HTTPS Ratio: {d.get('overall_https_ratio'):.4f}\n")
            f.write(f"  Top 1 Domain Share: {d.get('top_1_domain_share'):.4%}\n")
            f.write(f"  Indian TLD Ratio: {d.get('indian_tld_ratio'):.4%}\n")
            f.write(f"  Repo Brand Match Ratio: {d.get('repo_brand_match_ratio'):.4%}\n")
            f.write(f"  Independent Brand Match Ratio: {d.get('independent_brand_match_ratio'):.4%}\n")
            if d.get("path_by_class"):
                f.write("  Breakdown by Class:\n")
                for lbl, cinfo in d["path_by_class"].items():
                    f.write(f"    Class {lbl}: Count={cinfo['count']}, HasPath={cinfo['has_path_query_ratio']:.4f}, HTTPS={cinfo['https_ratio']:.4f}, MeanPathLen={cinfo['mean_path_length']:.1f}\n")
            if d.get("date_info"):
                f.write(f"  Dates: Unique={d['date_info']['unique_dates']}, Range=[{d['date_info']['min_date']} to {d['date_info']['max_date']}]\n")

        f.write("\n\n--- 2. TRAIN / TEST LEAKAGE AUDIT ---\n")
        for k, v in leakages.items():
            f.write(f"\n[{k}]\n")
            f.write(f"  Exact URL overlap: {v['url_overlap']}\n")
            f.write(f"  Registered domain overlap: {v['registered_domain_overlap']} ({v['domain_overlap_ratio_of_test']:.4%} of test domains)\n")

    print(f"\nCensus completed successfully. Reports saved to:\n  - {json_path}\n  - {txt_path}")


if __name__ == "__main__":
    main()
