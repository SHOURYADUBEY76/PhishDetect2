"""Unified feature extraction interface for training and real-time inference.
Safe, zero-network, sub-millisecond execution.
"""

from typing import Dict, List, Optional
import numpy as np
import pandas as pd
from src.features.lexical import extract_lexical_features
from src.features.structural import extract_structural_features
from src.features.regional import extract_regional_features


# Canonical feature orders to maintain strict consistency across training and inference
LEXICAL_FEATURE_NAMES = [
    "url_length", "hostname_length", "path_length", "query_length",
    "num_digits", "digit_ratio", "host_num_digits", "host_digit_ratio",
    "num_dots", "host_num_dots", "num_hyphens", "host_num_hyphens",
    "num_underscores", "num_slashes", "num_questions", "num_equals",
    "num_ats", "num_percents", "num_ampersands", "num_tildes",
    "special_char_ratio", "url_entropy", "host_entropy", "path_entropy"
]

STRUCTURAL_FEATURE_NAMES = [
    "num_subdomains", "has_excessive_subdomains", "has_ip", "is_https",
    "is_http", "has_port", "is_non_standard_port", "path_depth",
    "num_query_params", "suspicious_keyword_count", "has_double_slash_redirect",
    "percent_encoded_count", "max_host_token_len", "tld_length", "subdomain_length"
]

REGIONAL_FEATURE_NAMES = [
    "is_indian_tld", "has_indian_brand_token", "brand_token_count",
    "has_indian_lure", "lure_token_count", "brand_tld_misalignment",
    "max_brand_similarity", "has_homoglyph_or_punycode"
]


def get_feature_names(include_regional: bool = True) -> List[str]:
    """Get ordered list of feature names based on feature regime."""
    names = list(LEXICAL_FEATURE_NAMES) + list(STRUCTURAL_FEATURE_NAMES)
    if include_regional:
        names.extend(REGIONAL_FEATURE_NAMES)
    return names


def extract_features_dict(
    url: str,
    include_regional: bool = True,
    include_path_query: bool = True
) -> Dict[str, float]:
    """Extract full feature dictionary from a static URL string.
    
    Args:
        url: Static URL string.
        include_regional: Whether to include India-specific regional features.
        include_path_query: Whether to include path/query components (used for source bias audit).
        
    Returns:
        Dictionary mapping feature name to numeric value.
    """
    feats = {}
    feats.update(extract_lexical_features(url, include_path_query=include_path_query))
    feats.update(extract_structural_features(url, include_path_query=include_path_query))
    if include_regional:
        feats.update(extract_regional_features(url, include_path_query=include_path_query))
    return feats


def extract_features_vector(
    url: str,
    include_regional: bool = True,
    include_path_query: bool = True
) -> np.ndarray:
    """Extract features as a 1D NumPy array matching canonical order."""
    fdict = extract_features_dict(
        url,
        include_regional=include_regional,
        include_path_query=include_path_query
    )
    feature_names = get_feature_names(include_regional=include_regional)
    return np.array([fdict.get(col, 0.0) for col in feature_names], dtype=np.float32)


def extract_features_df(
    urls: List[str],
    include_regional: bool = True,
    include_path_query: bool = True
) -> pd.DataFrame:
    """Extract features for a list of URLs into a pandas DataFrame."""
    rows = [
        extract_features_dict(
            u,
            include_regional=include_regional,
            include_path_query=include_path_query
        )
        for u in urls
    ]
    df = pd.DataFrame(rows)
    feature_names = get_feature_names(include_regional=include_regional)
    return df[feature_names]
