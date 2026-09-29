"""Dataset loading and cached feature matrix retrieval.
Supports domain-grouped validation splitting for strict leakage-free threshold calibration.
"""

import os
import pandas as pd
import numpy as np
from src.features.extractor import extract_features_df, get_feature_names
from src.utils.domain import extract_registered_domain


FEATURE_CACHE_DIR = "data/processed/feature_cache"


def get_feature_matrix(
    df: pd.DataFrame,
    cache_key: str,
    include_regional: bool = True,
    include_path_query: bool = True,
    force_recompute: bool = False
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Extract or load cached feature matrix from a dataframe of URLs.
    
    Args:
        df: DataFrame containing 'url' and 'label' columns.
        cache_key: Unique identifier for the dataset split.
        include_regional: Whether regional features are included.
        include_path_query: Whether path/query features are included.
        force_recompute: Bypass cache if True.
        
    Returns:
        (X, y, feature_names)
    """
    os.makedirs(FEATURE_CACHE_DIR, exist_ok=True)
    suffix = f"reg_{int(include_regional)}_pq_{int(include_path_query)}"
    cache_path = os.path.join(FEATURE_CACHE_DIR, f"{cache_key}_{suffix}.parquet")

    if os.path.exists(cache_path) and not force_recompute:
        feat_df = pd.read_parquet(cache_path)
    else:
        print(f"  Extracting features for {cache_key} ({len(df)} URLs)...", flush=True)
        feat_df = extract_features_df(
            df["url"].tolist(),
            include_regional=include_regional,
            include_path_query=include_path_query
        )
        feat_df.to_parquet(cache_path, index=False)

    feature_names = get_feature_names(include_regional=include_regional)
    X = feat_df[feature_names].values.astype(np.float32)
    # Replace any potential NaN/Inf
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    y = df["label"].values.astype(int)

    return X, y, feature_names


def load_processed_split(split_name: str) -> pd.DataFrame:
    """Load a processed split CSV (e.g. 'global_train', 'global_test', 'indian_test')."""
    path = f"data/processed/{split_name}.csv"
    if not os.path.exists(path):
        raise FileNotFoundError(f"Processed dataset split not found: {path}")
    return pd.read_csv(path)


def split_train_val_grouped(
    df_train: pd.DataFrame,
    val_domain_ratio: float = 0.15,
    random_state: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split training dataframe into train and validation sets using registered domain grouping.
    
    Ensures zero domain leakage between train and validation for threshold tuning.
    """
    if "registered_domain" not in df_train.columns:
        df_train["registered_domain"] = df_train["url"].apply(extract_registered_domain)

    unique_domains = df_train["registered_domain"].unique()
    rng = np.random.RandomState(random_state)
    shuffled = rng.permutation(unique_domains)
    val_count = int(len(shuffled) * val_domain_ratio)

    val_domains = set(shuffled[:val_count])
    train_domains = set(shuffled[val_count:])

    df_tr = df_train[df_train["registered_domain"].isin(train_domains)].copy()
    df_vl = df_train[df_train["registered_domain"].isin(val_domains)].copy()

    return df_tr, df_vl
