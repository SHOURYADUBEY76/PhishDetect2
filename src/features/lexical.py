"""Lexical feature extraction from static URL strings.
Calculates statistical, character-level, and entropy distributions.
"""

import math
from collections import Counter
from urllib.parse import urlparse


def calculate_entropy(text: str) -> float:
    """Calculate Shannon entropy of a string."""
    if not text:
        return 0.0
    counts = Counter(text)
    length = len(text)
    return -sum((cnt / length) * math.log2(cnt / length) for cnt in counts.values())


def extract_lexical_features(url: str, include_path_query: bool = True) -> dict:
    """Extract lexical features from static URL string.
    
    Args:
        url: Static URL string.
        include_path_query: Whether to include path/query features (used for source bias audit).
        
    Returns:
        Dictionary of lexical feature names and float values.
    """
    url_str = str(url).strip()
    parsed = urlparse(url_str if "://" in url_str else f"http://{url_str}")
    hostname = parsed.netloc.split(":")[0] if parsed.netloc else ""
    path = parsed.path if include_path_query else ""
    query = parsed.query if include_path_query else ""

    url_len = len(url_str)
    host_len = len(hostname)
    path_len = len(path)
    query_len = len(query)

    # Character counts in full URL
    num_digits = sum(c.isdigit() for c in url_str)
    num_alpha = sum(c.isalpha() for c in url_str)
    num_dots = url_str.count(".")
    num_hyphens = url_str.count("-")
    num_underscores = url_str.count("_")
    num_slashes = url_str.count("/")
    num_questions = url_str.count("?") if include_path_query else 0
    num_equals = url_str.count("=") if include_path_query else 0
    num_ats = url_str.count("@")
    num_percents = url_str.count("%")
    num_ampersands = url_str.count("&") if include_path_query else 0
    num_tildes = url_str.count("~")

    # Hostname character counts
    host_digits = sum(c.isdigit() for c in hostname)
    host_hyphens = hostname.count("-")
    host_dots = hostname.count(".")

    # Ratios
    digit_ratio = num_digits / url_len if url_len > 0 else 0.0
    host_digit_ratio = host_digits / host_len if host_len > 0 else 0.0
    special_char_count = (
        num_dots + num_hyphens + num_underscores + num_slashes +
        num_questions + num_equals + num_ats + num_percents + num_ampersands
    )
    special_char_ratio = special_char_count / url_len if url_len > 0 else 0.0

    # Entropies
    url_entropy = calculate_entropy(url_str)
    host_entropy = calculate_entropy(hostname)
    path_entropy = calculate_entropy(path) if include_path_query and path else 0.0

    return {
        "url_length": float(url_len),
        "hostname_length": float(host_len),
        "path_length": float(path_len),
        "query_length": float(query_len),
        "num_digits": float(num_digits),
        "digit_ratio": float(digit_ratio),
        "host_num_digits": float(host_digits),
        "host_digit_ratio": float(host_digit_ratio),
        "num_dots": float(num_dots),
        "host_num_dots": float(host_dots),
        "num_hyphens": float(num_hyphens),
        "host_num_hyphens": float(host_hyphens),
        "num_underscores": float(num_underscores),
        "num_slashes": float(num_slashes),
        "num_questions": float(num_questions),
        "num_equals": float(num_equals),
        "num_ats": float(num_ats),
        "num_percents": float(num_percents),
        "num_ampersands": float(num_ampersands),
        "num_tildes": float(num_tildes),
        "special_char_ratio": float(special_char_ratio),
        "url_entropy": float(url_entropy),
        "host_entropy": float(host_entropy),
        "path_entropy": float(path_entropy),
    }
