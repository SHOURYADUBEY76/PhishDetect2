"""Benchmark Construction Script for Track A Experiments (A1-A5).
Creates leakage-free, balanced, and hardened evaluation partitions.
Adds row-level is_synthetic column to Indian splits.
Computes cryptographic SHA-256 hashes for all evaluation files.
"""

import os
import sys
import json
import hashlib
import re
from urllib.parse import urlparse
import pandas as pd
import numpy as np

# Ensure root directory is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.utils.domain import extract_registered_domain, extract_tld
from audit.independent_indian_entities import match_independent_indian_target, is_indian_hosting_tld

# The 10 brand-padding regex patterns from download_data.py
SYNTHETIC_TEMPLATES_REGEX = [
    r"sbi-kyc-update-portal-\d+\.info",
    r"hdfc-reward-points-claim-\d+\.xyz",
    r"icici-netbanking-verify-\d+\.site",
    r"paytm-cashback-offer-\d+\.club",
    r"phonepe-upi-verification-\d+\.link",
    r"uidai-aadhaar-link-urgent-\d+\.site",
    r"pan-card-update-kyc-\d+\.top",
    r"irctc-ticket-refund-\d+\.online",
    r"axis-bank-netbanking-verify-\d+\.info",
    r"pnb-online-alert-notice-\d+\.cloud",
]
COMBINED_SYNTH_REGEX = "|".join(SYNTHETIC_TEMPLATES_REGEX)


def compute_sha256(filepath: str) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def flag_synthetic_rows():
    """Add row-level is_synthetic column to processed Indian splits."""
    for path in ["data/processed/indian_train.csv", "data/processed/indian_test.csv"]:
        if os.path.exists(path):
            df = pd.read_csv(path)
            is_synth = df["url"].str.contains(COMBINED_SYNTH_REGEX, regex=True).astype(int)
            df["is_synthetic"] = is_synth
            df.to_csv(path, index=False)
            print(f"Added 'is_synthetic' to {path} ({is_synth.sum()} synthetic rows flagged of {len(df)})", flush=True)


def build_a1_benchmark():
    """Build A1 Path-Matched Evaluation Benchmark from PhreshPhish."""
    cache_path = "audit/results/phreshphish_metadata_streamed.parquet"
    if not os.path.exists(cache_path):
        print(f"Warning: {cache_path} not found. Skipping A1 build.")
        return

    df = pd.read_parquet(cache_path)
    df["label_num"] = df["label"].apply(lambda x: 1 if str(x).lower().startswith("phish") or x == 1 else 0)

    # Subsample 40k for clean benchmark
    df_sub = df.sample(n=min(len(df), 40000), random_state=42).copy()
    
    def parse_path_info(u):
        try:
            p = urlparse(u if "://" in u else f"http://{u}")
            has_pq = (len(p.path) > 1) or bool(p.query)
            p_depth = len([x for x in p.path.split("/") if x])
            is_https = u.lower().startswith("https://")
            return has_pq, p_depth, is_https, len(u)
        except Exception:
            return False, 0, False, len(str(u))

    pinfo = [parse_path_info(u) for u in df_sub["url"]]
    df_sub["has_path_query"] = [x[0] for x in pinfo]
    df_sub["path_depth"] = [x[1] for x in pinfo]
    df_sub["is_https"] = [x[2] for x in pinfo]
    df_sub["url_len"] = [x[3] for x in pinfo]

    df_sub["len_bin"] = pd.qcut(df_sub["url_len"], q=5, labels=False, duplicates="drop")
    df_sub["stratum"] = df_sub["has_path_query"].astype(str) + "_" + df_sub["len_bin"].astype(str) + "_" + df_sub["is_https"].astype(str)

    matched = []
    for _, grp in df_sub.groupby("stratum"):
        g0 = grp[grp["label_num"] == 0]
        g1 = grp[grp["label_num"] == 1]
        nm = min(len(g0), len(g1))
        if nm > 0:
            matched.extend([g0.sample(n=nm, random_state=42), g1.sample(n=nm, random_state=42)])

    df_matched = pd.concat(matched, ignore_index=True).sample(frac=1.0, random_state=42).reset_index(drop=True)
    df_matched["registered_domain"] = df_matched["url"].apply(extract_registered_domain)
    
    out_path = "data/benchmark/a1_path_matched.csv"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df_matched[["url", "label_num", "date", "registered_domain", "has_path_query", "path_depth", "is_https", "stratum"]].to_csv(out_path, index=False)
    print(f"Saved A1 Path-Matched Benchmark to {out_path} ({len(df_matched)} rows)", flush=True)


def build_a2_benchmark():
    """Build A2 Representation Diagnostic Benchmark across 50 top domains."""
    top_domains = [
        "google.com", "youtube.com", "facebook.com", "wikipedia.org", "yahoo.com",
        "amazon.com", "twitter.com", "instagram.com", "linkedin.com", "reddit.com",
        "netflix.com", "live.com", "microsoft.com", "apple.com", "bing.com",
        "spotify.com", "github.com", "cnn.com", "nytimes.com", "zoom.us",
        "sbi.co.in", "hdfcbank.com", "icicibank.com", "paytm.com", "phonepe.com",
        "irctc.co.in", "uidai.gov.in", "incometax.gov.in", "jio.com", "airtel.in",
        "quora.com", "dropbox.com", "adobe.com", "salesforce.com", "wordpress.org",
        "medium.com", "stackoverflow.com", "ebay.com", "whatsapp.com", "paypal.com",
        "telegram.org", "weather.com", "bbc.com", "walmart.com", "office.com",
        "pinterest.com", "tumblr.com", "imdb.com", "vimeo.com", "craigslist.org"
    ]
    records = []
    for dom in top_domains:
        records.append({"domain": dom, "url": f"https://{dom}", "variant_type": "https_bare"})
        records.append({"domain": dom, "url": f"https://{dom}/", "variant_type": "https_trailing_slash"})
        records.append({"domain": dom, "url": f"http://{dom}", "variant_type": "http_bare"})
        records.append({"domain": dom, "url": f"https://www.{dom}", "variant_type": "https_www_bare"})
        records.append({"domain": dom, "url": f"https://www.{dom}/", "variant_type": "https_www_slash"})
        records.append({"domain": dom, "url": dom, "variant_type": "raw_bare"})

    df_a2 = pd.DataFrame(records)
    out_path = "data/benchmark/a2_representation.csv"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df_a2.to_csv(out_path, index=False)
    print(f"Saved A2 Representation Diagnostic Benchmark to {out_path} ({len(df_a2)} rows)", flush=True)


def build_a3_benchmark():
    """Build A3 Lexicon-Clean Unseen Brands Holdout Benchmark."""
    # Held out brand entities strictly excluded from zero-knowledge feature extractor
    held_out_brands = ["phonepe", "aadhaar", "uidai", "irctc", "axis", "pnb", "bob", "kotak", "epfo", "paytm"]
    
    # Load PhreshPhish and identify genuine URLs targeting these brands or legitimate domains
    cache_path = "audit/results/phreshphish_metadata_streamed.parquet"
    if not os.path.exists(cache_path):
        return
    df = pd.read_parquet(cache_path)
    df["label_num"] = df["label"].apply(lambda x: 1 if str(x).lower().startswith("phish") or x == 1 else 0)

    def matches_held_out(u):
        u_low = str(u).lower()
        return any(b in u_low for b in held_out_brands)

    df_held = df[df["url"].apply(matches_held_out)].copy()
    df_held["registered_domain"] = df_held["url"].apply(extract_registered_domain)
    
    # Balance with generic benign URLs
    df_benign = df[(df["label_num"] == 0) & (~df["url"].apply(matches_held_out))].sample(n=min(len(df_held), 2000), random_state=42).copy()
    df_benign["registered_domain"] = df_benign["url"].apply(extract_registered_domain)

    df_a3 = pd.concat([df_held, df_benign], ignore_index=True).sample(frac=1.0, random_state=42).reset_index(drop=True)
    out_path = "data/benchmark/a3_unseen_brands.csv"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df_a3[["url", "label_num", "date", "registered_domain"]].to_csv(out_path, index=False)
    print(f"Saved A3 Unseen Brands Benchmark to {out_path} ({len(df_a3)} rows)", flush=True)


def build_a4_a5_benchmarks():
    """Build A4 Cross-Distribution & A5 Temporal Splits Benchmarks."""
    cache_path = "audit/results/phreshphish_metadata_streamed.parquet"
    if not os.path.exists(cache_path):
        return
    df = pd.read_parquet(cache_path)
    df["label_num"] = df["label"].apply(lambda x: 1 if str(x).lower().startswith("phish") or x == 1 else 0)
    df["registered_domain"] = df["url"].apply(extract_registered_domain)
    df["is_indian_hosting"] = df["url"].apply(is_indian_hosting_tld).astype(int)

    # A4: Cross-distribution hosting set
    out_path_a4 = "data/benchmark/a4_cross_distribution.csv"
    df_sample = df.sample(n=min(len(df), 20000), random_state=42).copy()
    df_sample[["url", "label_num", "registered_domain", "is_indian_hosting", "date"]].to_csv(out_path_a4, index=False)
    print(f"Saved A4 Cross-Distribution Benchmark to {out_path_a4} ({len(df_sample)} rows)", flush=True)

    # A5: Temporal Splits (Train: 2024, Test: 2025)
    df["date_dt"] = pd.to_datetime(df["date"], errors="coerce")
    df_valid_dates = df.dropna(subset=["date_dt"]).copy()
    
    train_mask = df_valid_dates["date_dt"] < pd.Timestamp("2025-01-01")
    test_mask = df_valid_dates["date_dt"] >= pd.Timestamp("2025-06-01")

    df_train_temp = df_valid_dates[train_mask].sample(n=min(train_mask.sum(), 25000), random_state=42).copy()
    df_test_temp = df_valid_dates[test_mask].sample(n=min(test_mask.sum(), 15000), random_state=42).copy()

    # Enforce domain separation between temporal train and test
    train_domains = set(df_train_temp["registered_domain"])
    df_test_temp = df_test_temp[~df_test_temp["registered_domain"].isin(train_domains)].copy()

    out_temp_train = "data/benchmark/a5_temporal_train.csv"
    out_temp_test = "data/benchmark/a5_temporal_test.csv"
    df_train_temp[["url", "label_num", "date", "registered_domain"]].to_csv(out_temp_train, index=False)
    df_test_temp[["url", "label_num", "date", "registered_domain"]].to_csv(out_temp_test, index=False)
    print(f"Saved A5 Temporal Splits to {out_temp_train} ({len(df_train_temp)} rows) and {out_temp_test} ({len(df_test_temp)} rows)", flush=True)


def generate_manifest():
    """Generate SHA-256 hash manifest for all benchmark files."""
    benchmark_dir = "data/benchmark"
    manifest = {}
    if os.path.exists(benchmark_dir):
        for fname in sorted(os.listdir(benchmark_dir)):
            if fname.endswith(".csv"):
                fpath = os.path.join(benchmark_dir, fname)
                df = pd.read_csv(fpath)
                sha = compute_sha256(fpath)
                manifest[fname] = {
                    "path": fpath.replace("\\", "/"),
                    "sha256": sha,
                    "rows": len(df),
                    "columns": list(df.columns),
                    "label_distribution": df["label_num"].value_counts().to_dict() if "label_num" in df.columns else "N/A"
                }

    manifest_path = "research/split_manifest.json"
    os.makedirs(os.path.dirname(manifest_path), exist_ok=True)
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"Generated Benchmark Manifest at {manifest_path}", flush=True)


if __name__ == "__main__":
    print("Building Track A benchmarks...")
    flag_synthetic_rows()
    build_a1_benchmark()
    build_a2_benchmark()
    build_a3_benchmark()
    build_a4_a5_benchmarks()
    generate_manifest()
    print("All benchmarks built successfully!")
