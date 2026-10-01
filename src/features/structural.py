"""Structural feature extraction from static URL strings.
Extracts protocol, subdomain hierarchy, path depth, IP presence, and structural anomalies.
Hardened with defensive parsing for out-of-range ports and malformed schemes.
"""

import re
from urllib.parse import urlparse
from src.utils.domain import extract_subdomain, extract_registered_domain, extract_tld


# Standard IPv4 pattern
IPV4_PATTERN = re.compile(r"^(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)$")

# Common phishing intent keywords (matched as discrete tokens or bound words)
SUSPICIOUS_KEYWORDS = [
    "login", "signin", "verify", "verification", "secure", "security",
    "account", "update", "banking", "wallet", "confirm", "confirmation",
    "password", "credential", "auth", "authenticate", "portal", "support",
    "billing", "payment", "invoice", "recover", "unlock", "alert"
]


def is_ip_address(hostname: str) -> bool:
    """Check if the hostname is an IPv4 or hexadecimal IP representation."""
    if not hostname:
        return False
    # Strip port if present
    host = hostname.split(":")[0].strip("[]")
    if IPV4_PATTERN.match(host):
        return True
    # Hex or octal IP detection (e.g. 0x7f000001 or 017700000001)
    if host.startswith("0x") or (host.isdigit() and len(host) >= 8):
        return True
    return False


def extract_structural_features(url: str, include_path_query: bool = True) -> dict:
    """Extract structural URL features with hardened defensive parsing.
    
    Args:
        url: Static URL string.
        include_path_query: Whether to include path and query components.
        
    Returns:
        Dictionary of structural feature names and float values.
    """
    url_str = str(url).strip()
    try:
        parsed = urlparse(url_str if "://" in url_str else f"http://{url_str}")
        hostname = (parsed.netloc or "").split(":")[0]
        scheme = (parsed.scheme or "").lower()
    except Exception:
        parsed = None
        hostname = ""
        scheme = ""

    path = (parsed.path if parsed else "") if include_path_query else ""
    query = (parsed.query if parsed else "") if include_path_query else ""

    subdomain = extract_subdomain(url_str)
    registered_domain = extract_registered_domain(url_str)
    tld = extract_tld(url_str)

    # Subdomain depth
    subdomain_tokens = [t for t in subdomain.split(".") if t]
    num_subdomains = len(subdomain_tokens)
    has_excessive_subdomains = 1.0 if num_subdomains >= 3 else 0.0

    # IP detection
    has_ip = 1.0 if is_ip_address(hostname) else 0.0

    # Scheme
    is_https = 1.0 if scheme == "https" else 0.0
    is_http = 1.0 if scheme == "http" else 0.0

    # Hardened Port Evaluation (prevents unhandled ValueError on malformed ports)
    has_port = 0.0
    is_non_standard_port = 0.0
    if parsed is not None:
        try:
            port = parsed.port
            if port is not None:
                has_port = 1.0
                if (scheme == "https" and port != 443) or (scheme == "http" and port != 80):
                    is_non_standard_port = 1.0
        except ValueError:
            # Out of range or non-integer port is an anomaly/malformed attempt
            has_port = 1.0
            is_non_standard_port = 1.0

    # Path depth
    path_segments = [s for s in path.split("/") if s]
    path_depth = len(path_segments)

    # Query param count
    query_params = [q for q in query.split("&") if q]
    num_query_params = len(query_params)

    # Suspicious keywords presence (tokenized to avoid false substring matches)
    search_tokens = set(re.findall(r"[a-z0-9]+", f"{hostname} {path} {query}".lower()))
    keyword_count = sum(1 for kw in SUSPICIOUS_KEYWORDS if kw in search_tokens)

    # Redirection trick: double slash in path (e.g., http://legit.com//phish.com)
    has_double_slash_redirect = 1.0 if "//" in path else 0.0

    # Hex/percent encoding count
    percent_encoded_count = url_str.count("%")

    # Longest token in hostname
    host_tokens = re.split(r"[.\-_]", hostname)
    max_host_token_len = max((len(t) for t in host_tokens), default=0)

    # TLD length
    tld_length = len(tld)

    return {
        "num_subdomains": float(num_subdomains),
        "has_excessive_subdomains": float(has_excessive_subdomains),
        "has_ip": float(has_ip),
        "is_https": float(is_https),
        "is_http": float(is_http),
        "has_port": float(has_port),
        "is_non_standard_port": float(is_non_standard_port),
        "path_depth": float(path_depth),
        "num_query_params": float(num_query_params),
        "suspicious_keyword_count": float(keyword_count),
        "has_double_slash_redirect": float(has_double_slash_redirect),
        "percent_encoded_count": float(percent_encoded_count),
        "max_host_token_len": float(max_host_token_len),
        "tld_length": float(tld_length),
        "subdomain_length": float(len(subdomain)),
    }
