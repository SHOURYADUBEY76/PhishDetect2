"""Download and normalize the UCI PhiUSIIL Phishing URL Dataset (UCI ID 967).
Extracts raw URL strings only.
Normalizes inverted labels: original (1 = legit, 0 = phish) -> standardized (1 = phish, 0 = legit).
Enforces registered domain grouping and saves clean splits to data/processed/.
"""

import os
import sys
import io
import zipfile
import urllib.request
import pandas as pd
import numpy as np

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.utils.domain import normalize_url, extract_registered_domain


def download_and_process_phiusiil(max_samples: int = 15000) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Download PhiUSIIL from UCI archive and build domain-grouped splits."""
    raw_path = "data/raw/phiusiil_raw.csv"
    train_path = "data/processed/phiusiil_train.csv"
    test_path = "data/processed/phiusiil_test.csv"

    if os.path.exists(train_path) and os.path.exists(test_path):
        print(f"[PhiUSIIL] Loading cached splits from {train_path} and {test_path}", flush=True)
        return pd.read_csv(train_path), pd.read_csv(test_path)

    os.makedirs("data/raw", exist_ok=True)
    os.makedirs("data/processed", exist_ok=True)

    if not os.path.exists(raw_path):
        print("[PhiUSIIL] Fetching archive from UCI repository...", flush=True)
        url = "https://archive.ics.uci.edu/static/public/967/phiusiil+phishing+url+dataset.zip"
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=45) as resp:
            zip_bytes = resp.read()

        with zipfile.ZipFile(io.BytesIO(zip_bytes)) as z:
            with z.open("PhiUSIIL_Phishing_URL_Dataset.csv") as f:
                print("[PhiUSIIL] Reading raw CSV from zip archive...", flush=True)
                df_raw = pd.read_csv(f, usecols=["URL", "label"])

        # Label normalization: original 1=legit, 0=phish -> our standard: 1=phish, 0=legit
        df_raw["url"] = df_raw["URL"].astype(str).apply(normalize_url)
        df_raw["label"] = (df_raw["label"] == 0).astype(int)
        df_raw = df_raw[["url", "label"]].drop_duplicates(subset=["url"]).reset_index(drop=True)

        # Sample for balanced benchmark
        if len(df_raw) > max_samples:
            df_raw = df_raw.sample(n=max_samples, random_state=42).reset_index(drop=True)

        df_raw.to_csv(raw_path, index=False)
        print(f"[PhiUSIIL] Saved raw normalized URLs ({len(df_raw)} samples) to {raw_path}", flush=True)
    else:
        df_raw = pd.read_csv(raw_path)

    # Domain-grouped partition
    df_raw["registered_domain"] = df_raw["url"].apply(extract_registered_domain)
    unique_domains = df_raw["registered_domain"].unique()

    rng = np.random.RandomState(42)
    shuffled_domains = rng.permutation(unique_domains)
    split_idx = int(0.7 * len(shuffled_domains))

    train_domains = set(shuffled_domains[:split_idx])
    test_domains = set(shuffled_domains[split_idx:])

    df_train = df_raw[df_raw["registered_domain"].isin(train_domains)].copy()
    df_test = df_raw[df_raw["registered_domain"].isin(test_domains)].copy()

    # Balance train and test splits to 50/50
    def balance_df(df):
        phish = df[df["label"] == 1]
        legit = df[df["label"] == 0]
        n = min(len(phish), len(legit))
        return pd.concat([
            phish.sample(n=n, random_state=42),
            legit.sample(n=n, random_state=42)
        ], ignore_index=True).sample(frac=1.0, random_state=42).reset_index(drop=True)

    df_train_bal = balance_df(df_train)
    df_test_bal = balance_df(df_test)

    df_train_bal.to_csv(train_path, index=False)
    df_test_bal.to_csv(test_path, index=False)
    print(f"[PhiUSIIL] Built train ({len(df_train_bal)} samples) and test ({len(df_test_bal)} samples) with zero domain leakage.", flush=True)
    return df_train_bal, df_test_bal


if __name__ == "__main__":
    download_and_process_phiusiil()
