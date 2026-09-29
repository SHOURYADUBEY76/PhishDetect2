"""Region-aware feature extraction targeting the Indian phishing threat landscape.
Analyzes brand impersonation, regional TLD misalignment, lure actions, and homoglyphs.
"""

import re
from difflib import SequenceMatcher
from urllib.parse import urlparse
from src.utils.domain import extract_registered_domain, extract_tld, extract_subdomain


# Indian high-value targets across banking, payments, telecom, and citizen services
INDIAN_BRANDS = [
    "sbi", "onlinesbi", "statebank", "hdfc", "hdfcbank", "icici", "icicibank",
    "axis", "axisbank", "pnb", "punjabnational", "bob", "bankofbaroda",
    "canarabank", "kotak", "yesbank", "indusind", "unionbank", "rbi",
    "paytm", "phonepe", "gpay", "googlepay", "bharatpe", "bhim", "upi",
    "mobikwik", "freecharge", "cred", "aadhaar", "uidai", "pan", "incometax",
    "nsdl", "utiitsl", "epfo", "epfindia", "irctc", "indiapost", "postinfo",
    "digilocker", "parivahan", "voter", "echallan", "jio", "airtel", "bsnl", "fastag"
]

# Canonical mapping of aliases to brand entity
BRAND_ALIAS_MAP = {
    "onlinesbi": "sbi",
    "statebank": "sbi",
    "hdfcbank": "hdfc",
    "icicibank": "icici",
    "axisbank": "axis",
    "punjabnational": "pnb",
    "bankofbaroda": "bob",
    "googlepay": "gpay",
    "uidai": "aadhaar",
    "nsdl": "pan",
    "utiitsl": "pan",
    "incometax": "pan",
    "epfindia": "epfo",
    "postinfo": "indiapost",
}

# Legitimate official registered domains for Indian entities
OFFICIAL_DOMAINS = {
    "sbi": {"sbi.co.in", "onlinesbi.sbi", "onlinesbi.com", "statebankofindia.com"},
    "hdfc": {"hdfcbank.com", "hdfc.com", "hdfcbank.net"},
    "icici": {"icicibank.com", "icici.com"},
    "axis": {"axisbank.com"},
    "pnb": {"pnbindia.in", "pnb.in"},
    "bob": {"bankofbaroda.in", "bankofbaroda.com"},
    "kotak": {"kotak.com"},
    "rbi": {"rbi.org.in"},
    "paytm": {"paytm.com", "paytmbank.com"},
    "phonepe": {"phonepe.com"},
    "gpay": {"google.com", "pay.google.com"},
    "upi": {"npci.org.in", "bhimupi.org.in"},
    "bhim": {"bhimupi.org.in", "npci.org.in"},
    "aadhaar": {"uidai.gov.in"},
    "pan": {"incometax.gov.in", "tin-nsdl.com", "utiitsl.com"},
    "irctc": {"irctc.co.in"},
    "indiapost": {"indiapost.gov.in"},
    "epfo": {"epfindia.gov.in"},
    "digilocker": {"digilocker.gov.in"},
    "parivahan": {"parivahan.gov.in"},
    "jio": {"jio.com", "reliancedigital.in"},
    "airtel": {"airtel.in", "airtel.com"},
    "bsnl": {"bsnl.co.in", "bsnl.in"},
}

# Regional top-level domains for India
INDIAN_TLDS = {
    "in", "co.in", "net.in", "org.in", "gov.in", "nic.in", "res.in", "ac.in",
    "edu.in", "mil.in", "ind.in", "gen.in", "firm.in", "bharat"
}

# Action lures prevalent in regional attacks
INDIAN_LURES = [
    "kyc", "kycupdate", "kyc-verify", "panlink", "pan-update", "lottery",
    "recharge", "subsidy", "reward", "refund", "cashback", "electricity",
    "biometric", "block-card", "unblock", "yono", "rewards-point"
]

# Common Cyrillic homoglyphs mapping to Latin
HOMOGLYPH_MAP = {
    'а': 'a', 'с': 'c', 'е': 'e', 'о': 'o', 'р': 'p',
    'ѕ': 's', 'ԁ': 'd', 'ԛ': 'q', 'і': 'i', 'ј': 'j'
}


def detect_homoglyph_or_punycode(url_str: str) -> float:
    """Detect punycode (xn--) or non-ASCII homoglyphs in hostname."""
    if "xn--" in url_str.lower():
        return 1.0
    for char in url_str:
        if char in HOMOGLYPH_MAP or ord(char) > 127:
            return 1.0
    return 0.0


def string_similarity(a: str, b: str) -> float:
    """Calculate SequenceMatcher similarity ratio between two tokens."""
    return SequenceMatcher(None, a, b).ratio()


def extract_regional_features(url: str, include_path_query: bool = True) -> dict:
    """Extract region-aware indicators for Indian threat landscape.
    
    Args:
        url: Static URL string.
        include_path_query: Whether to inspect path/query for lures.
        
    Returns:
        Dictionary of regional feature values.
    """
    url_str = str(url).strip()
    parsed = urlparse(url_str if "://" in url_str else f"http://{url_str}")
    hostname = (parsed.netloc or "").split(":")[0].lower()
    path = parsed.path.lower() if include_path_query else ""
    query = parsed.query.lower() if include_path_query else ""
    
    registered_domain = extract_registered_domain(url_str)
    tld = extract_tld(url_str)
    search_context = f"{hostname} {path} {query}".lower()
    norm_context = "".join(HOMOGLYPH_MAP.get(c, c) for c in search_context)
    norm_hostname = "".join(HOMOGLYPH_MAP.get(c, c) for c in hostname)

    # 1. Regional TLD signal
    is_indian_tld = 1.0 if tld in INDIAN_TLDS else 0.0

    # 2. Indian brand token matching (normalized to canonical entity)
    matched_canonical = set()
    for brand in INDIAN_BRANDS:
        # Check boundary or delimiter match in both raw and homoglyph-normalized contexts
        pattern = r"(?:^|[^a-z0-9])" + re.escape(brand) + r"(?:[^a-z0-9]|$)"
        if (
            re.search(pattern, search_context) or brand in hostname
            or re.search(pattern, norm_context) or brand in norm_hostname
        ):
            canonical = BRAND_ALIAS_MAP.get(brand, brand)
            matched_canonical.add(canonical)

    has_indian_brand_token = 1.0 if len(matched_canonical) > 0 else 0.0
    brand_token_count = float(len(matched_canonical))

    # 3. Indian lure keywords
    matched_lures = [lure for lure in INDIAN_LURES if lure in search_context]
    has_indian_lure = 1.0 if len(matched_lures) > 0 else 0.0
    lure_token_count = float(len(matched_lures))

    # 4. Target vs TLD misalignment:
    # High risk signal: Indian banking/gov brand token is present, but registered domain is NOT the official domain!
    brand_tld_misalignment = 0.0
    for entity in matched_canonical:
        official_domains = OFFICIAL_DOMAINS.get(entity, set())
        if official_domains:
            if registered_domain not in official_domains:
                brand_tld_misalignment = 1.0
                break
        else:
            # Generic brand not in registry, check if domain ends with expected domain
            if entity in registered_domain and registered_domain not in {f"{entity}.com", f"{entity}.in", f"{entity}.co.in"}:
                brand_tld_misalignment = 1.0
                break

    # 5. Fuzzy similarity to nearest Indian brand in hostname tokens
    host_tokens = re.split(r"[.\-_/]", hostname)
    max_similarity = 0.0
    for token in host_tokens:
        if len(token) >= 3:
            for b in INDIAN_BRANDS:
                sim = string_similarity(token, b)
                if sim > max_similarity and sim < 1.0:  # Exclude exact matches from typo-similarity
                    max_similarity = sim
    
    # 6. Homoglyph / Punycode
    has_homoglyph = detect_homoglyph_or_punycode(hostname)

    return {
        "is_indian_tld": float(is_indian_tld),
        "has_indian_brand_token": float(has_indian_brand_token),
        "brand_token_count": float(brand_token_count),
        "has_indian_lure": float(has_indian_lure),
        "lure_token_count": float(lure_token_count),
        "brand_tld_misalignment": float(brand_tld_misalignment),
        "max_brand_similarity": float(max_similarity),
        "has_homoglyph_or_punycode": float(has_homoglyph),
    }
