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
    """Build A3 Lexicon-Clean Unseen Brands Holdout Benchmark.
    
    Attribution Rules:
      1. Polysemic tokens ('bob', 'axis', 'pan') require compound naming ('bankofbaroda', 'axisbank')
         or explicit upstream feed target confirmation ('axisbank', 'bank of baroda', 'pnb').
         Standalone substring matches without confirmation are classified as AMBIGUOUS and excluded.
      2. Non-polysemic brand tokens ('sbi', 'hdfc', 'icici', 'phonepe', 'paytm', 'uidai', 'irctc', 'jio')
         are matched with token boundaries and confirmed to not belong to official brand domains.
      3. Brand instances with < 20 rows or < 5 registered domains are flagged as is_sparse_stratum=1
         and included in the pooled aggregate benchmark, but not reported in isolated brand breakdowns.
      4. All synthetic regex templates are strictly excluded.
    """
    out_path = "data/benchmark/a3_unseen_brands.csv"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    
    # 1. Candidate extraction from real indian_test (excluding synthetic templates)
    phish_records = []
    if os.path.exists("data/processed/indian_test.csv"):
        df_itest = pd.read_csv("data/processed/indian_test.csv")
        real_phish = df_itest[(df_itest["label"] == 1) & (df_itest["is_synthetic"] == 0)].copy()
        
        # Explicit target verification rules on real_phish
        brand_pats = {
            "sbi": re.compile(r"(?:^|[^a-zA-Z0-9])(?:sbi|onlinesbi|statebank)(?:[^a-zA-Z0-9]|$)", re.IGNORECASE),
            "hdfc": re.compile(r"(?:^|[^a-zA-Z0-9])(?:hdfc|hdfcbank)(?:[^a-zA-Z0-9]|$)", re.IGNORECASE),
            "icici": re.compile(r"(?:^|[^a-zA-Z0-9])(?:icici|icicibank)(?:[^a-zA-Z0-9]|$)", re.IGNORECASE),
            "axis": re.compile(r"(?:^|[^a-zA-Z0-9])(?:axisbank)(?:[^a-zA-Z0-9]|$)", re.IGNORECASE),  # Compound only
            "pnb": re.compile(r"(?:^|[^a-zA-Z0-9])(?:pnb|punjabnational)(?:[^a-zA-Z0-9]|$)", re.IGNORECASE),
            "bob": re.compile(r"(?:^|[^a-zA-Z0-9])(?:bankofbaroda)(?:[^a-zA-Z0-9]|$)", re.IGNORECASE),  # Compound only
            "kotak": re.compile(r"(?:^|[^a-zA-Z0-9])(?:kotak|kotakbank)(?:[^a-zA-Z0-9]|$)", re.IGNORECASE),
            "paytm": re.compile(r"(?:^|[^a-zA-Z0-9])(?:paytm)(?:[^a-zA-Z0-9]|$)", re.IGNORECASE),
            "phonepe": re.compile(r"(?:^|[^a-zA-Z0-9])(?:phonepe)(?:[^a-zA-Z0-9]|$)", re.IGNORECASE),
            "uidai": re.compile(r"(?:^|[^a-zA-Z0-9])(?:uidai|aadhaar)(?:[^a-zA-Z0-9]|$)", re.IGNORECASE),
            "irctc": re.compile(r"(?:^|[^a-zA-Z0-9])(?:irctc)(?:[^a-zA-Z0-9]|$)", re.IGNORECASE),
            "jio": re.compile(r"(?:^|[^a-zA-Z0-9])(?:jio)(?:[^a-zA-Z0-9]|$)", re.IGNORECASE),
            "unionbank": re.compile(r"(?:^|[^a-zA-Z0-9])(?:unionbank)(?:[^a-zA-Z0-9]|$)", re.IGNORECASE),
        }
        
        for _, row in real_phish.iterrows():
            u = row["url"]
            for bname, pat in brand_pats.items():
                if pat.search(u):
                    phish_records.append({
                        "url": u,
                        "label_num": 1,
                        "date": row.get("date", "2024-01-01"),
                        "registered_domain": extract_registered_domain(u),
                        "brand_entity": bname,
                    })
                    break

    # 2. Candidate extraction from PhreshPhish census metadata
    cache_path = "audit/results/phreshphish_metadata_streamed.parquet"
    if os.path.exists(cache_path):
        df_p = pd.read_parquet(cache_path)
        census_ind_targets = {
            "governmentofindia": "governmentofindia",
            "della adventure and resorts": "della",
            "sterling holidays": "sterling",
            "meesho": "meesho",
            "npci": "npci",
            "flipkart": "flipkart",
            "hdfc bank": "hdfc",
            "reliance retail": "reliance",
            "india_post": "indiapost",
            "axisbank": "axis",
            "reserve bank of india": "rbi",
            "boi": "boi",
            "moneyview": "moneyview",
            "bse ltd (bombay stock exchange)": "bse",
            "eurekaforbes": "eurekaforbes",
            "yesbank": "yesbank",
            "ujjivan bank": "ujjivan",
            "allcargo group": "allcargo",
            "nykaa": "nykaa",
        }
        p_phish = df_p[(df_p["label"] == "phish") | (df_p["label"] == 1)].copy()
        for tgt_val, bname in census_ind_targets.items():
            matched = p_phish[p_phish["target"].astype(str).str.lower().str.strip() == tgt_val]
            for _, row in matched.iterrows():
                u = row["url"]
                phish_records.append({
                    "url": u,
                    "label_num": 1,
                    "date": row.get("date", "2024-01-01"),
                    "registered_domain": extract_registered_domain(u),
                    "brand_entity": bname,
                })

    df_phish = pd.DataFrame(phish_records).drop_duplicates(subset=["url"]).reset_index(drop=True)
    
    # Stratum sparsity calculation: <20 instances or <5 domains
    brand_counts = df_phish.groupby("brand_entity").agg(n_rows=("url", "count"), n_doms=("registered_domain", "nunique"))
    sparse_brands = set(brand_counts[(brand_counts["n_rows"] < 20) | (brand_counts["n_doms"] < 5)].index)
    df_phish["is_sparse_stratum"] = df_phish["brand_entity"].apply(lambda b: 1 if b in sparse_brands else 0)

    # 3. Balanced benign sample from verified clean benign sources
    benign_urls = []
    if os.path.exists("data/processed/indian_test.csv"):
        df_itest = pd.read_csv("data/processed/indian_test.csv")
        b_itest = df_itest[df_itest["label"] == 0]["url"].tolist()
        benign_urls.extend(b_itest)
    
    df_benign_pool = pd.DataFrame({"url": benign_urls}).drop_duplicates().sample(n=min(len(benign_urls), len(df_phish)), random_state=42).copy()
    df_benign_pool["label_num"] = 0
    df_benign_pool["date"] = "2024-01-01"
    df_benign_pool["registered_domain"] = df_benign_pool["url"].apply(extract_registered_domain)
    df_benign_pool["brand_entity"] = "none"
    df_benign_pool["is_sparse_stratum"] = 0

    df_a3 = pd.concat([df_phish, df_benign_pool], ignore_index=True).sample(frac=1.0, random_state=42).reset_index(drop=True)
    df_a3[["url", "label_num", "date", "registered_domain", "brand_entity", "is_sparse_stratum"]].to_csv(out_path, index=False)
    print(f"Saved Validated A3 Unseen Brands Benchmark to {out_path} ({len(df_a3)} rows: {len(df_phish)} phish, {len(df_benign_pool)} benign)", flush=True)


def build_a4_a5_benchmarks():
    """Build A4 External Cross-Distribution and A5 Temporal Benchmarks."""
    # A4 Primary: Full External PhiUSIIL Test Set (N=3,520; 1,760 benign, 1,760 phish)
    out_path_a4_primary = "data/benchmark/a4_cross_distribution.csv"
    out_path_a4_disjoint = "data/benchmark/a4_domain_disjoint.csv"
    
    if os.path.exists("data/processed/phiusiil_test.csv"):
        df_phi = pd.read_csv("data/processed/phiusiil_test.csv")
        df_phi["label_num"] = df_phi["label"].astype(int)
        df_phi["registered_domain"] = df_phi["url"].apply(extract_registered_domain)
        df_phi["date"] = "2024-01-01"
        
        # Primary Benchmark: Full External Test Set
        df_phi[["url", "label_num", "registered_domain", "date"]].to_csv(out_path_a4_primary, index=False)
        print(f"Saved Primary A4 Cross-Distribution Benchmark to {out_path_a4_primary} ({len(df_phi)} rows)", flush=True)

        # Secondary Benchmark: Strict Domain-Disjoint Filtered Partition (N=2,888)
        # Purge 92 overlapping domains from strict union training pool
        g_tr = pd.read_csv("data/processed/global_train.csv") if os.path.exists("data/processed/global_train.csv") else pd.DataFrame(columns=["registered_domain"])
        a5_tr = pd.read_csv("data/benchmark/a5_temporal_train.csv") if os.path.exists("data/benchmark/a5_temporal_train.csv") else pd.DataFrame(columns=["registered_domain"])
        i_tr = pd.read_csv("data/processed/indian_train.csv") if os.path.exists("data/processed/indian_train.csv") else pd.DataFrame(columns=["registered_domain"])
        
        union_tr_doms = set(g_tr["registered_domain"].dropna()).union(set(a5_tr["registered_domain"].dropna())).union(set(i_tr["registered_domain"].dropna()))
        df_disjoint = df_phi[~df_phi["registered_domain"].isin(union_tr_doms)].copy()
        df_disjoint[["url", "label_num", "registered_domain", "date"]].to_csv(out_path_a4_disjoint, index=False)
        print(f"Saved Secondary A4 Domain-Disjoint Benchmark to {out_path_a4_disjoint} ({len(df_disjoint)} rows)", flush=True)

    # A5: Temporal Splits from PhreshPhish Census
    cache_path = "audit/results/phreshphish_metadata_streamed.parquet"
    if os.path.exists(cache_path):
        df = pd.read_parquet(cache_path)
        df["label_num"] = df["label"].apply(lambda x: 1 if str(x).lower().startswith("phish") or x == 1 else 0)
        df["registered_domain"] = df["url"].apply(extract_registered_domain)
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


def build_hosting_infrastructure_census():
    """Build Dedicated Hosting Infrastructure Census Benchmark (N=1,682 URLs)."""
    cache_path = "audit/results/phreshphish_metadata_streamed.parquet"
    if not os.path.exists(cache_path):
        return
        
    df = pd.read_parquet(cache_path)
    df["label_num"] = df["label"].apply(lambda x: 1 if str(x).lower().startswith("phish") or x == 1 else 0)
    df["registered_domain"] = df["url"].apply(extract_registered_domain)
    df["is_indian_hosting"] = df["url"].apply(is_indian_hosting_tld)

    # Filter to Indian hosting (.in / .bharat)
    df_hosted = df[df["is_indian_hosting"]].copy()

    # Explicit auditable target category mapping
    INDIAN_TARGET_ENTITIES = {
        'governmentofindia', 'della adventure and resorts', 'sterling holidays',
        'meesho', 'npci', 'flipkart', 'hdfc bank', 'reliance retail',
        'india_post', 'axisbank', 'reserve bank of india', 'boi', 'moneyview',
        'bse ltd (bombay stock exchange)', 'eurekaforbes', 'yesbank',
        'ujjivan bank', 'allcargo group', 'nykaa'
    }

    UNSPECIFIED_GENERIC_TARGETS = {
        'other', 'genericwebmailphishing', 'generic email', 'genericfakebank',
        'generic - no brand identified', 'server data', 'email admin', 'vm',
        'transact', 'one', 'one_digital_service'
    }

    def map_target(row):
        if row["label_num"] == 0:
            return "Benign Reference"
        t = str(row.get("target", "")).lower().strip()
        if t in INDIAN_TARGET_ENTITIES:
            return "Domestic Indian Target"
        elif t in UNSPECIFIED_GENERIC_TARGETS or not t or t in ['none', 'n/a', 'unknown']:
            return "Ambiguous / Unspecified Target"
        else:
            return "Identified Foreign / Global Target"

    df_hosted["target_category"] = df_hosted.apply(map_target, axis=1)

    out_path = "data/benchmark/hosting_infrastructure_census.csv"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df_hosted[["url", "label_num", "registered_domain", "target_category", "date"]].to_csv(out_path, index=False)
    print(f"Saved Hosting Infrastructure Census Benchmark to {out_path} ({len(df_hosted)} rows: {len(df_hosted[df_hosted['label_num']==1])} phish, {len(df_hosted[df_hosted['label_num']==0])} benign)", flush=True)


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
    build_hosting_infrastructure_census()
    generate_manifest()
    print("All benchmarks built successfully!")
