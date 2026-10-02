"""Domain extraction, URL validation, and canonicalization utilities.
Safe string-only analysis without making any network calls.
"""

import re
from typing import Optional, Tuple
from urllib.parse import urlparse, urlsplit
import ipaddress
import tldextract


# Cache tldextract instance to optimize performance across large datasets
_extractor = tldextract.TLDExtract(cache_dir="data/raw/tld_cache")

# Supported web schemes
VALID_SCHEMES = {"http", "https"}


def validate_url_syntax(url: str) -> Tuple[bool, Optional[str]]:
    """Validate URL syntax and scheme strictly.
    
    Returns:
        (is_valid, error_message)
    """
    if not isinstance(url, str) or not url.strip():
        return False, "URL string cannot be empty or whitespace."

    u = url.strip()

    # Reject dangerous or unsupported schemes immediately
    scheme_match = re.match(r"^([a-zA-Z][a-zA-Z0-9+.-]*):", u)
    if scheme_match:
        scheme = scheme_match.group(1).lower()
        if scheme not in VALID_SCHEMES:
            return False, f"Unsupported URL scheme '{scheme}'. Only HTTP and HTTPS are permitted."

    # Parse structure defensively
    target_url = u if "://" in u else f"https://{u}"
    try:
        split_res = urlsplit(target_url)
    except ValueError as e:
        return False, f"Malformed URL syntax: {str(e)}"
    except Exception as e:
        return False, f"Failed to parse URL: {str(e)}"

    # Validate hostname
    netloc = split_res.netloc
    if not netloc:
        return False, "URL must contain a valid host / authority."

    # Validate port if present
    if ":" in netloc:
        host_part, _, port_part = netloc.rpartition(":")
        # Check if port_part is bracketed IPv6 continuation
        if not host_part.endswith("]"):
            try:
                port_int = int(port_part)
                if not (0 <= port_int <= 65535):
                    return False, f"Port '{port_int}' is out of valid range (0-65535)."
            except ValueError:
                return False, f"Invalid port '{port_part}': Port must be an integer."

    # Validate IPv6 brackets if present
    if "[" in netloc or "]" in netloc:
        if not (netloc.startswith("[") and "]" in netloc):
            return False, "Malformed IPv6 address syntax in host."
        ipv6_str = netloc[1:netloc.index("]")]
        try:
            ipaddress.IPv6Address(ipv6_str)
        except ValueError:
            return False, f"Invalid IPv6 address literal '[{ipv6_str}]'."

    return True, None


def has_idn_punycode(url_or_host: str) -> bool:
    """Detect whether host contains IDN homoglyphs or Punycode encoding."""
    u = str(url_or_host).strip().lower()
    if "://" in u:
        try:
            parsed = urlparse(u)
            host = parsed.netloc.split(":")[0]
        except Exception:
            host = u.split("/")[0].split(":")[0]
    else:
        host = u.split("/")[0].split(":")[0]

    # Check for Punycode prefix
    if "xn--" in host:
        return True
    # Check for non-ASCII characters in host
    if any(ord(c) > 127 for c in host):
        return True
    return False


def canonicalize_url(url: str, default_scheme: str = "https") -> Tuple[str, bool]:
    """Standardize URL string representation.
    
    Design Choices (Explicit & Tested):
    1. Scheme Handling: When missing, default to 'https://' (modern web standard).
       Returns scheme_assumed=True so callers and APIs can distinguish explicit from inferred schemes.
    2. Host Lowercasing: Netloc is converted to lowercase per RFC 3986.
    3. Trailing Slash: For bare domain roots without paths, a single trailing slash '/'
       is standardized (e.g. 'https://example.com/' instead of ''), aligning with browser address bar conventions.
    4. Subdomain & 'www' Preservation:
       - 'www.' is PRESERVED in the canonical URL string to allow structural features to measure
         exact presented host complexity.
       - 'www.' is stripped separately during registered domain extraction (extract_registered_domain).
    5. IDN Homoglyphs: Preserved in host (not silently resolved away), allowing homoglyph detectors
       to flag spoofing attempts.

    Returns:
        (canonical_url, scheme_assumed)
    """
    if not isinstance(url, str):
        return "", False

    u = url.strip()
    scheme_assumed = False
    if not u.startswith(("http://", "https://")):
        u = f"{default_scheme}://" + u
        scheme_assumed = True

    try:
        parsed = urlparse(u)
        scheme = parsed.scheme.lower()
        netloc = parsed.netloc.lower()
        path = parsed.path
        if not path:
            path = "/"
        query = f"?{parsed.query}" if parsed.query else ""
        fragment = f"#{parsed.fragment}" if parsed.fragment else ""
        canonical = f"{scheme}://{netloc}{path}{query}{fragment}"
        return canonical, scheme_assumed
    except Exception:
        return u, scheme_assumed


def normalize_url(url: str) -> str:
    """Backward-compatible wrapper for canonicalize_url."""
    canonical, _ = canonicalize_url(url)
    return canonical


def extract_registered_domain(url: str) -> str:
    """Extract registered domain (domain + suffix, e.g. 'google.com', 'sbi.co.in').
    
    Explicitly strips 'www.' subdomain to isolate organizational registered domain.
    """
    if not isinstance(url, str):
        return ""
    try:
        extracted = _extractor(url)
        if extracted.domain and extracted.suffix:
            return f"{extracted.domain}.{extracted.suffix}".lower()
        if extracted.domain:
            return extracted.domain.lower()
        # Fallback to netloc
        parsed = urlparse(url if "://" in url else f"http://{url}")
        host = (parsed.netloc or "").split(":")[0].lower()
        if host.startswith("www."):
            host = host[4:]
        return host
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
