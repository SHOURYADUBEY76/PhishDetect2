"""Domain extraction and URL normalization utilities.
Safe string-only analysis without making any network calls.
"""

from urllib.parse import urlparse
import tldextract


# Cache tldextract instance to optimize performance across large datasets
_extractor = tldextract.TLDExtract(cache_dir="data/raw/tld_cache")


def extract_registered_domain(url: str) -> str:
    """Extract registered domain (domain + suffix, e.g. 'google.com', 'sbi.co.in').
    
    Args:
        url: Static URL string.
        
    Returns:
        Registered domain string in lowercase.
    """
    if not isinstance(url, str):
        return ""
    try:
        extracted = _extractor(url)
        if extracted.registered_domain:
            return extracted.registered_domain.lower()
        if extracted.domain:
            return extracted.domain.lower()
        # Fallback to netloc
        parsed = urlparse(url if "://" in url else f"http://{url}")
        return (parsed.netloc or "").split(":")[0].lower()
    except Exception:
        return ""


def extract_tld(url: str) -> str:
    """Extract top-level domain suffix (e.g. 'in', 'co.in', 'com')."""
    if not isinstance(url, str):
        return ""
    try:
        extracted = _extractor(url)
        return extracted.suffix.lower() if extracted.suffix else ""
    except Exception:
        return ""


def extract_subdomain(url: str) -> str:
    """Extract subdomain string (e.g. 'secure.login')."""
    if not isinstance(url, str):
        return ""
    try:
        extracted = _extractor(url)
        return extracted.subdomain.lower() if extracted.subdomain else ""
    except Exception:
        return ""


def normalize_url(url: str) -> str:
    """Standardize URL string representation."""
    if not isinstance(url, str):
        return ""
    url = url.strip()
    if not url.startswith(("http://", "https://", "ftp://")):
        url = "http://" + url
    return url
