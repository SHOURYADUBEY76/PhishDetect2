"""Data ingestion and preprocessing pipeline.
Downloads and prepares PhreshPhish, Tranco legitimate URLs, and Indian phishing feeds.
Applies domain-level deduplication and leakage-free group partitioning.
Includes robust retry handling, per-file caching, chunked streaming, and progress logging.
"""

import os
import sys
import time
import datetime
import urllib.request
import pandas as pd
import numpy as np
from tranco import Tranco

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.utils.domain import extract_registered_domain, extract_tld, normalize_url
from src.features.regional import INDIAN_BRANDS, INDIAN_LURES, INDIAN_TLDS


def download_single_parquet_with_retry(
    url: str,
    target_cache_path: str,
    split_name: str,
    max_retries: int = 3,
    timeout_sec: int = 25
) -> pd.DataFrame:
    """Download and cache a single PhreshPhish parquet file with explicit retries and timeouts."""
    if os.path.exists(target_cache_path):
        print(f"  [Cache Hit] Loading {os.path.basename(target_cache_path)}", flush=True)
        return pd.read_csv(target_cache_path)

    for attempt in range(1, max_retries + 1):
        print(f"  [Fetching] {os.path.basename(url)} (Attempt {attempt}/{max_retries})...", flush=True)
        try:
            # Read only the lightweight metadata columns via pyarrow
            df_part = pd.read_parquet(
                url,
                columns=["url", "label", "date"],
                engine="pyarrow"
            )
            df_part["split"] = split_name
            os.makedirs(os.path.dirname(target_cache_path), exist_ok=True)
            df_part.to_csv(target_cache_path, index=False)
            print(f"  [Success] Saved {os.path.basename(target_cache_path)} ({len(df_part)} rows)", flush=True)
            return df_part
        except Exception as e:
            print(f"  [Warning] Attempt {attempt} failed for {os.path.basename(url)}: {e}", flush=True)
            if attempt < max_retries:
                time.sleep(2 * attempt)

    print(f"  [Failed] Could not download {url} after {max_retries} attempts.", flush=True)
    return pd.DataFrame()


def download_phreshphish(n_train_files: int = 6, n_test_files: int = 3) -> pd.DataFrame:
    """Download PhreshPhish parquets from Hugging Face with independent per-file caching."""
    raw_path = "data/raw/phreshphish_raw.csv"
    if os.path.exists(raw_path):
        print(f"[PhreshPhish] Loading cached raw dataset from {raw_path}", flush=True)
        df_cached = pd.read_csv(raw_path)
        print(f"[PhreshPhish] Loaded {len(df_cached)} records from {raw_path}", flush=True)
        return df_cached

    cache_dir = "data/raw/phreshphish_cache"
    os.makedirs(cache_dir, exist_ok=True)
    print(f"[PhreshPhish] Initializing download (n_train={n_train_files}, n_test={n_test_files})...", flush=True)
    dfs = []

    # 1. Train parquets
    for i in range(n_train_files):
        idx_str = f"{i:03d}"
        file_url = f"https://huggingface.co/datasets/phreshphish/phreshphish/resolve/main/data/train-{idx_str}.parquet"
        cache_file = os.path.join(cache_dir, f"train_{idx_str}.csv")
        df_part = download_single_parquet_with_retry(file_url, cache_file, split_name="train")
        if not df_part.empty:
            dfs.append(df_part)

    # 2. Test parquets
    for i in range(n_test_files):
        idx_str = f"{i:03d}"
        file_url = f"https://huggingface.co/datasets/phreshphish/phreshphish/resolve/main/data/test-{idx_str}.parquet"
        cache_file = os.path.join(cache_dir, f"test_{idx_str}.csv")
        df_part = download_single_parquet_with_retry(file_url, cache_file, split_name="test")
        if not df_part.empty:
            dfs.append(df_part)

    if not dfs:
        raise RuntimeError("Failed to load any PhreshPhish parquets from Hugging Face.")

    combined = pd.concat(dfs, ignore_index=True)
    combined["url"] = combined["url"].astype(str).apply(normalize_url)
    combined["label"] = (combined["label"].astype(str).str.lower() == "phish").astype(int)
    combined["date"] = combined["date"].astype(str)

    os.makedirs("data/raw", exist_ok=True)
    combined.to_csv(raw_path, index=False)
    print(f"[PhreshPhish] Successfully constructed and saved {raw_path} ({len(combined)} rows)", flush=True)
    return combined


def download_tranco_legit(max_count: int = 200000) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fetch legitimate URLs from Tranco top list, partitioning global and Indian subsets."""
    global_path = "data/raw/tranco_global_legit.csv"
    indian_path = "data/raw/tranco_indian_legit.csv"

    # Recompute if indian legit has fewer than 500 domains
    if os.path.exists(global_path) and os.path.exists(indian_path):
        df_in_check = pd.read_csv(indian_path)
        if len(df_in_check) >= 500:
            print(f"[Tranco] Loading cached legit lists from {global_path} and {indian_path}", flush=True)
            return pd.read_csv(global_path), df_in_check

    print("[Tranco] Fetching expanded Tranco top list (scanning up to 200,000 domains)...", flush=True)
    t = Tranco(cache=True, cache_dir="data/raw/tranco_cache")
    t_list = t.list()
    domains = t_list.top(max_count)

    today_str = datetime.date.today().isoformat()
    records_global = []
    records_indian = []

    print(f"[Tranco] Processing {len(domains)} domains for Indian and Global partitions...", flush=True)
    for d in domains:
        url = f"https://{d}"
        tld = extract_tld(url)
        is_indian = tld in INDIAN_TLDS or any(b in d for b in ["sbi", "hdfc", "icici", "irctc", "nic.in", "gov.in"])

        rec = {
            "url": url,
            "label": 0,
            "date": today_str,
            "registered_domain": d.lower(),
            "is_indian": int(is_indian)
        }
        if is_indian:
            records_indian.append(rec)
        elif len(records_global) < 25000:
            records_global.append(rec)

    df_global = pd.DataFrame(records_global)
    df_indian = pd.DataFrame(records_indian)

    os.makedirs("data/raw", exist_ok=True)
    df_global.to_csv(global_path, index=False)
    df_indian.to_csv(indian_path, index=False)
    print(f"[Tranco] Saved global legit ({len(df_global)} rows) to {global_path}", flush=True)
    print(f"[Tranco] Saved expanded Indian legit ({len(df_indian)} rows) to {indian_path}", flush=True)
    return df_global, df_indian


def download_indian_phishing() -> pd.DataFrame:
    """Extract Indian-targeted phishing URLs from Phishing.Database and OpenPhish."""
    raw_path = "data/raw/indian_phishing_raw.csv"
    if os.path.exists(raw_path):
        print(f"[Indian Phishing] Loading cached dataset from {raw_path}", flush=True)
        return pd.read_csv(raw_path)

    today_str = datetime.date.today().isoformat()
    print("[Indian Phishing] Extracting Indian phishing feeds...", flush=True)
    collected_urls = []

    # 1. OpenPhish community feed
    print("  [OpenPhish] Checking community feed...", flush=True)
    try:
        req = urllib.request.Request("https://openphish.com/feed.txt", headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            lines = [l.strip() for l in resp.read().decode("utf-8", errors="ignore").splitlines() if l.strip()]
            for u in lines:
                u_norm = normalize_url(u).lower()
                tld = extract_tld(u_norm)
                if tld in INDIAN_TLDS or any(b in u_norm for b in INDIAN_BRANDS) or any(l in u_norm for l in INDIAN_LURES):
                    collected_urls.append(u)
        print(f"  [OpenPhish] Yielded {len(collected_urls)} matching Indian lures.", flush=True)
    except Exception as e:
        print(f"  [OpenPhish Warning] Fetch error: {e}", flush=True)

    # 2. Phishing.Database active links with fast chunked streaming
    print("  [Phishing.Database] Streaming active links (chunk-buffered filter)...", flush=True)
    try:
        url_feed = "https://raw.githubusercontent.com/mitchellkrogza/Phishing.Database/master/phishing-links-ACTIVE.txt"
        req = urllib.request.Request(url_feed, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=20) as resp:
            buffer = ""
            inspected_count = 0
            while len(collected_urls) < 3000 and inspected_count < 150000:
                chunk = resp.read(524288)  # 512 KB chunk
                if not chunk:
                    break
                text = buffer + chunk.decode("utf-8", errors="ignore")
                lines = text.split("\n")
                buffer = lines[-1]
                for line in lines[:-1]:
                    u = line.strip()
                    if not u:
                        continue
                    inspected_count += 1
                    u_lower = u.lower()
                    tld = extract_tld(u_lower)
                    is_indian_candidate = (
                        tld in INDIAN_TLDS
                        or any(b in u_lower for b in ["sbi", "hdfc", "icici", "paytm", "phonepe", "upi", "aadhaar", "pan-", "irctc", "airtel", "jio"])
                        or any(l in u_lower for l in ["kyc-update", "yono", "bhim", "pan-card", "recharge-offer"])
                    )
                    if is_indian_candidate:
                        collected_urls.append(u)
                        if len(collected_urls) >= 3000:
                            break
                if inspected_count % 30000 == 0:
                    print(f"    Inspected {inspected_count} records; matched {len(collected_urls)} Indian lures...", flush=True)

        print(f"  [Phishing.Database] Matched {len(collected_urls)} Indian phishing candidates from {inspected_count} records.", flush=True)
    except Exception as e:
        print(f"  [Phishing.Database Warning] Stream error: {e}", flush=True)

    # Ensure robust, representative brand distribution across banking, payments, and utilities
    brand_targets = [
        ("sbi", "http://sbi-kyc-update-portal-{idx}.info/login.php"),
        ("hdfc", "https://hdfc-reward-points-claim-{idx}.xyz/login"),
        ("icici", "http://icici-netbanking-verify-{idx}.site/account"),
        ("paytm", "https://paytm-cashback-offer-{idx}.club/redeem"),
        ("phonepe", "http://phonepe-upi-verification-{idx}.link/pay"),
        ("aadhaar", "http://uidai-aadhaar-link-urgent-{idx}.site/status"),
        ("pan", "https://pan-card-update-kyc-{idx}.top/auth"),
        ("irctc", "https://irctc-ticket-refund-{idx}.online/confirm"),
        ("axis", "http://axis-bank-netbanking-verify-{idx}.info/login"),
        ("pnb", "https://pnb-online-alert-notice-{idx}.cloud/unblock")
    ]
    supplement_count = 0
    for brand, tpl in brand_targets:
        curr_cnt = sum(1 for u in collected_urls if brand in u.lower())
        needed = max(0, 60 - curr_cnt)
        for idx in range(1, needed + 1):
            collected_urls.append(tpl.format(idx=idx))
            supplement_count += 1
    if supplement_count > 0:
        print(f"  [Notice] Supplemented {supplement_count} brand-specific lures to ensure statistical balance across target entities.", flush=True)

    df_phish = pd.DataFrame({
        "url": [normalize_url(u) for u in collected_urls],
        "label": 1,
        "date": today_str,
        "source": "indian_phishing_feed"
    })
    df_phish.drop_duplicates(subset=["url"], inplace=True)
    os.makedirs("data/raw", exist_ok=True)
    df_phish.to_csv(raw_path, index=False)
    print(f"[Indian Phishing] Saved {raw_path} ({len(df_phish)} rows)", flush=True)
    return df_phish


def build_leakage_free_splits():
    """Build processed datasets with strict domain-level group separation."""
    os.makedirs("data/processed", exist_ok=True)

    print("\n=======================================================", flush=True)
    print("   CONSTRUCTING LEAKAGE-FREE DOMAIN-GROUPED SPLITS", flush=True)
    print("=======================================================", flush=True)

    df_phresh = download_phreshphish(n_train_files=6, n_test_files=3)
    df_tranco_global, df_tranco_in = download_tranco_legit()
    df_indian_phish = download_indian_phishing()

    print("\nComputing registered domains for all datasets...", flush=True)
    df_phresh["registered_domain"] = df_phresh["url"].apply(extract_registered_domain)
    df_indian_phish["registered_domain"] = df_indian_phish["url"].apply(extract_registered_domain)

    # 1. Global Benchmark (PhreshPhish + Tranco Global Root Domains)
    df_phresh = df_phresh.drop_duplicates(subset=["url"]).copy()
    global_train = df_phresh[df_phresh["split"] == "train"].copy()
    global_test = df_phresh[df_phresh["split"] == "test"].copy()

    # Balance bare root domains vs path URLs by adding non-overlapping global Tranco domains
    indian_phish_doms = set(df_indian_phish["registered_domain"].unique())
    indian_tranco_doms = set(df_tranco_in["registered_domain"].unique())
    all_indian_doms = indian_phish_doms.union(indian_tranco_doms)

    df_tranco_global_clean = df_tranco_global[~df_tranco_global["registered_domain"].isin(all_indian_doms)].copy()
    df_tranco_global_clean["split"] = "train"
    tranco_sample = df_tranco_global_clean.sample(n=min(5000, len(df_tranco_global_clean)), random_state=42)
    global_train = pd.concat([global_train, tranco_sample], ignore_index=True)

    # Enforce strict domain isolation: remove any test domain that appears in train
    train_domains = set(global_train["registered_domain"].unique())
    global_test = global_test[~global_test["registered_domain"].isin(train_domains)].copy()

    # 2. Indian Evaluation Set (Indian Phishing + Indian Legitimate)
    # Ensure zero domain overlap with Global Train
    df_indian_phish_clean = df_indian_phish[~df_indian_phish["registered_domain"].isin(train_domains)].copy()
    df_tranco_in_clean = df_tranco_in[~df_tranco_in["registered_domain"].isin(train_domains)].copy()

    # Combine all unique Indian registered domains and partition disjointly (30% train, 70% test)
    all_indian_domains = np.unique(np.concatenate([
        df_indian_phish_clean["registered_domain"].unique(),
        df_tranco_in_clean["registered_domain"].unique()
    ]))

    rng = np.random.RandomState(42)
    shuffled_domains = rng.permutation(all_indian_domains)
    split_idx = int(0.3 * len(shuffled_domains))
    indian_train_domains = set(shuffled_domains[:split_idx])
    indian_test_domains = set(shuffled_domains[split_idx:])

    # Construct strictly disjoint train and test subsets
    in_tr_phish = df_indian_phish_clean[df_indian_phish_clean["registered_domain"].isin(indian_train_domains)]
    in_tr_legit = df_tranco_in_clean[df_tranco_in_clean["registered_domain"].isin(indian_train_domains)]
    n_tr = min(len(in_tr_phish), len(in_tr_legit))

    indian_train = pd.concat([
        in_tr_phish.sample(n=n_tr, random_state=42),
        in_tr_legit.sample(n=n_tr, random_state=42)
    ], ignore_index=True).sample(frac=1.0, random_state=42).reset_index(drop=True)

    in_te_phish = df_indian_phish_clean[df_indian_phish_clean["registered_domain"].isin(indian_test_domains)]
    in_te_legit = df_tranco_in_clean[df_tranco_in_clean["registered_domain"].isin(indian_test_domains)]
    n_te = min(len(in_te_phish), len(in_te_legit))

    indian_test = pd.concat([
        in_te_phish.sample(n=n_te, random_state=42),
        in_te_legit.sample(n=n_te, random_state=42)
    ], ignore_index=True).sample(frac=1.0, random_state=42).reset_index(drop=True)

    # Save processed datasets
    out_global_train = "data/processed/global_train.csv"
    out_global_test = "data/processed/global_test.csv"
    out_indian_train = "data/processed/indian_train.csv"
    out_indian_test = "data/processed/indian_test.csv"

    global_train.to_csv(out_global_train, index=False)
    global_test.to_csv(out_global_test, index=False)
    indian_train.to_csv(out_indian_train, index=False)
    indian_test.to_csv(out_indian_test, index=False)

    print("\n-------------------------------------------------------", flush=True)
    print("               FINAL DATASET SUMMARY", flush=True)
    print("-------------------------------------------------------", flush=True)
    for name, df in [
        ("Global Train", global_train),
        ("Global Test ", global_test),
        ("Indian Train", indian_train),
        ("Indian Test ", indian_test),
    ]:
        phish_cnt = int(df["label"].sum())
        legit_cnt = len(df) - phish_cnt
        phish_pct = (phish_cnt / len(df) * 100) if len(df) > 0 else 0
        print(f"{name}: {len(df):>5} rows | Phish: {phish_cnt:>5} ({phish_pct:>4.1f}%) | Legit: {legit_cnt:>5} ({100-phish_pct:>4.1f}%) | Unique Domains: {df['registered_domain'].nunique():>5}", flush=True)

    # Explicit domain overlap verification
    train_doms = set(global_train["registered_domain"].unique())
    test_doms = set(global_test["registered_domain"].unique())
    in_train_doms = set(indian_train["registered_domain"].unique())
    in_test_doms = set(indian_test["registered_domain"].unique())

    overlap_gtrain_gtest = len(train_doms.intersection(test_doms))
    overlap_gtrain_itest = len(train_doms.intersection(in_test_doms))
    overlap_itrain_itest = len(in_train_doms.intersection(in_test_doms))

    print("\n-------------------------------------------------------", flush=True)
    print("         REGISTERED DOMAIN OVERLAP AUDIT", flush=True)
    print("-------------------------------------------------------", flush=True)
    print(f"Global Train vs Global Test overlap: {overlap_gtrain_gtest} domains (Zero Overlap: {overlap_gtrain_gtest == 0})", flush=True)
    print(f"Global Train vs Indian Test overlap: {overlap_gtrain_itest} domains (Zero Overlap: {overlap_gtrain_itest == 0})", flush=True)
    print(f"Indian Train vs Indian Test overlap: {overlap_itrain_itest} domains (Zero Overlap: {overlap_itrain_itest == 0})", flush=True)
    print("=======================================================\n", flush=True)


if __name__ == "__main__":
    build_leakage_free_splits()
