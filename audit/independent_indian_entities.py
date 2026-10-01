"""Independent Definition of Indian Entities and Indian-Targeted URLs.
Authored and committed independently of the repository's internal feature detector lexicon.

Sources:
1. Reserve Bank of India (RBI): Scheduled Commercial Banks, Payments Banks, and Regulated FinTech Entities.
2. National Payments Corporation of India (NPCI): UPI, BHIM, RuPay, FASTag, IMPS.
3. Government of India Citizen Infrastructure: UIDAI (Aadhaar), CBDT / Income Tax (PAN, NSDL, UTIITSL),
   EPFO, IRCTC, Department of Posts (India Post), MoRTH (Parivahan, eChallan), DigiLocker, ECI (Voter ID).
4. Major Indian Telecom Service Providers: Reliance Jio, Bharti Airtel, BSNL, Vodafone Idea (Vi).
5. Major Regulated Indian Digital Payments Providers: Paytm, PhonePe, Google Pay (India), Razorpay, CRED, BharatPe, MobiKwik.

Principle of Separation:
- Target Region: The intended victim / impersonated brand or institution.
- Hosting Region: The registration or infrastructure location (e.g. .in ccTLD or Indian ASNs).
These two dimensions MUST NOT be conflated. A generic phishing site targeting Netflix hosted on a .in domain
is classified as "Phishing Hosted on Indian TLD", NOT "Indian-Targeted Phishing".
"""

import re
from typing import Dict, List, Optional, Set, Tuple
from urllib.parse import urlparse

# Authoritative list of independent Indian target entities
INDEPENDENT_INDIAN_ENTITIES: Dict[str, Dict] = {
    # 1. Major Banking & Regulatory Entities
    "state_bank_of_india": {
        "canonical_name": "State Bank of India",
        "tokens": ["sbi", "onlinesbi", "statebank", "statebankofindia"],
        "official_domains": {"sbi.co.in", "onlinesbi.sbi", "onlinesbi.com", "statebankofindia.com"},
        "sector": "banking"
    },
    "hdfc_bank": {
        "canonical_name": "HDFC Bank",
        "tokens": ["hdfc", "hdfcbank"],
        "official_domains": {"hdfcbank.com", "hdfc.com"},
        "sector": "banking"
    },
    "icici_bank": {
        "canonical_name": "ICICI Bank",
        "tokens": ["icici", "icicibank"],
        "official_domains": {"icicibank.com", "icici.com"},
        "sector": "banking"
    },
    "punjab_national_bank": {
        "canonical_name": "Punjab National Bank",
        "tokens": ["pnb", "punjabnational", "pnbindia"],
        "official_domains": {"pnbindia.in", "pnb.in"},
        "sector": "banking"
    },
    "bank_of_baroda": {
        "canonical_name": "Bank of Baroda",
        "tokens": ["bob", "bankofbaroda"],
        "official_domains": {"bankofbaroda.in", "bankofbaroda.com"},
        "sector": "banking"
    },
    "axis_bank": {
        "canonical_name": "Axis Bank",
        "tokens": ["axis", "axisbank"],
        "official_domains": {"axisbank.com"},
        "sector": "banking"
    },
    "kotak_mahindra_bank": {
        "canonical_name": "Kotak Mahindra Bank",
        "tokens": ["kotak", "kotakbank"],
        "official_domains": {"kotak.com"},
        "sector": "banking"
    },
    "canara_bank": {
        "canonical_name": "Canara Bank",
        "tokens": ["canarabank", "canara"],
        "official_domains": {"canarabank.com"},
        "sector": "banking"
    },
    "union_bank_of_india": {
        "canonical_name": "Union Bank of India",
        "tokens": ["unionbank", "unionbankofindia"],
        "official_domains": {"unionbankofindia.co.in"},
        "sector": "banking"
    },
    "indusind_bank": {
        "canonical_name": "IndusInd Bank",
        "tokens": ["indusind", "indusindbank"],
        "official_domains": {"indusind.com"},
        "sector": "banking"
    },
    "reserve_bank_of_india": {
        "canonical_name": "Reserve Bank of India",
        "tokens": ["rbi"],
        "official_domains": {"rbi.org.in"},
        "sector": "regulatory"
    },

    # 2. National Payments & Infrastructure (NPCI & UPI Ecosystem)
    "npci_upi": {
        "canonical_name": "NPCI / Unified Payments Interface / BHIM",
        "tokens": ["npci", "upi", "bhim", "rupay", "fastag"],
        "official_domains": {"npci.org.in", "bhimupi.org.in"},
        "sector": "payments_infrastructure"
    },
    "paytm": {
        "canonical_name": "Paytm / One97 Communications",
        "tokens": ["paytm", "paytmbank"],
        "official_domains": {"paytm.com", "paytmbank.com"},
        "sector": "fintech"
    },
    "phonepe": {
        "canonical_name": "PhonePe",
        "tokens": ["phonepe"],
        "official_domains": {"phonepe.com"},
        "sector": "fintech"
    },
    "google_pay_india": {
        "canonical_name": "Google Pay India (Tez / UPI)",
        "tokens": ["googlepay", "gpay"],
        "official_domains": {"pay.google.com"},
        "sector": "fintech"
    },
    "cred": {
        "canonical_name": "CRED",
        "tokens": ["cred"],
        "official_domains": {"cred.club"},
        "sector": "fintech"
    },
    "bharatpe": {
        "canonical_name": "BharatPe",
        "tokens": ["bharatpe"],
        "official_domains": {"bharatpe.com"},
        "sector": "fintech"
    },
    "mobikwik": {
        "canonical_name": "MobiKwik",
        "tokens": ["mobikwik"],
        "official_domains": {"mobikwik.com"},
        "sector": "fintech"
    },

    # 3. Government & Public Citizen Services
    "uidai_aadhaar": {
        "canonical_name": "UIDAI / Aadhaar",
        "tokens": ["uidai", "aadhaar", "myaadhaar"],
        "official_domains": {"uidai.gov.in", "myaadhaar.uidai.gov.in"},
        "sector": "government_identity"
    },
    "income_tax_pan": {
        "canonical_name": "Income Tax Department / PAN Card",
        "tokens": ["incometax", "nsdl", "utiitsl", "tin-nsdl"],
        "official_domains": {"incometax.gov.in", "tin-nsdl.com", "utiitsl.com"},
        "sector": "government_tax"
    },
    "epfo": {
        "canonical_name": "Employees Provident Fund Organisation",
        "tokens": ["epfo", "epfindia", "passbook.epfindia"],
        "official_domains": {"epfindia.gov.in"},
        "sector": "government_welfare"
    },
    "irctc": {
        "canonical_name": "Indian Railway Catering and Tourism Corporation",
        "tokens": ["irctc"],
        "official_domains": {"irctc.co.in"},
        "sector": "government_transport"
    },
    "india_post": {
        "canonical_name": "Department of Posts / India Post",
        "tokens": ["indiapost", "postinfo", "ippb"],
        "official_domains": {"indiapost.gov.in", "ippbonline.com"},
        "sector": "government_postal"
    },
    "parivahan": {
        "canonical_name": "MoRTH Parivahan Sewa / eChallan",
        "tokens": ["parivahan", "echallan", "sarathi", "vahan"],
        "official_domains": {"parivahan.gov.in", "echallan.parivahan.gov.in"},
        "sector": "government_transport"
    },
    "digilocker": {
        "canonical_name": "DigiLocker",
        "tokens": ["digilocker"],
        "official_domains": {"digilocker.gov.in"},
        "sector": "government_digital"
    },
    "election_commission": {
        "canonical_name": "Election Commission of India / Voter Services",
        "tokens": ["eci", "nvsp", "voterportal"],
        "official_domains": {"eci.gov.in", "voters.eci.gov.in"},
        "sector": "government_civic"
    },

    # 4. Telecommunications
    "reliance_jio": {
        "canonical_name": "Reliance Jio",
        "tokens": ["jio"],
        "official_domains": {"jio.com"},
        "sector": "telecom"
    },
    "bharti_airtel": {
        "canonical_name": "Bharti Airtel",
        "tokens": ["airtel"],
        "official_domains": {"airtel.in", "airtel.com"},
        "sector": "telecom"
    },
    "bsnl": {
        "canonical_name": "Bharat Sanchar Nigam Limited",
        "tokens": ["bsnl"],
        "official_domains": {"bsnl.co.in", "bsnl.in"},
        "sector": "telecom"
    },
}

# Recognized Indian TLDs for hosting region analysis
INDIAN_HOSTING_TLDS: Set[str] = {
    "in", "co.in", "net.in", "org.in", "gov.in", "nic.in", "res.in", "ac.in",
    "edu.in", "mil.in", "ind.in", "gen.in", "firm.in", "bharat"
}


def is_indian_hosting_tld(url_or_hostname: str) -> bool:
    """Determine whether the URL's hostname has an Indian top-level domain."""
    u = str(url_or_hostname).strip().lower()
    if "://" in u:
        parsed = urlparse(u)
        hostname = parsed.netloc.split(":")[0]
    else:
        hostname = u.split("/")[0].split(":")[0]

    parts = hostname.split(".")
    if len(parts) >= 2:
        if parts[-1] == "in":
            return True
        if parts[-1] == "bharat":
            return True
        if len(parts) >= 3 and ".".join(parts[-2:]) in INDIAN_HOSTING_TLDS:
            return True
    return False


def match_independent_indian_target(
    url: str,
    target_tag: Optional[str] = None
) -> Tuple[bool, Optional[str], str]:
    """Determine if a URL targets an independent Indian entity.
    
    Returns:
        (is_indian_targeted, matched_entity_key, match_source)
        where match_source is 'explicit_target_tag', 'token_match', or 'none'.
    """
    # 1. Check explicit target tag from upstream dataset annotation
    if target_tag and str(target_tag).strip().lower() not in {"", "none", "nan", "other"}:
        t_clean = str(target_tag).strip().lower()
        for entity_key, meta in INDEPENDENT_INDIAN_ENTITIES.items():
            if t_clean == entity_key or t_clean in meta["tokens"] or t_clean == meta["canonical_name"].lower():
                return True, entity_key, "explicit_target_tag"

    # 2. Tokenize URL into alphanumeric words
    u_lower = str(url).lower()
    tokens = set(re.findall(r"[a-z0-9]+", u_lower))

    # Match tokens against entity identifiers
    for entity_key, meta in INDEPENDENT_INDIAN_ENTITIES.items():
        for tok in meta["tokens"]:
            if tok in tokens:
                # Disambiguate short tokens (e.g. 'bob', 'pan') to avoid common English word collisions
                if tok == "bob":
                    # Require bank context or baroda
                    if any(c in tokens for c in ["bank", "baroda", "banking", "netbanking", "login", "kyc"]):
                        return True, entity_key, "token_match"
                elif tok == "axis":
                    if any(c in tokens for c in ["bank", "banking", "netbanking", "login", "card", "kyc"]):
                        return True, entity_key, "token_match"
                elif tok == "pan":
                    if any(c in tokens for c in ["card", "kyc", "update", "link", "aadhaar", "incometax", "uti", "nsdl"]):
                        return True, entity_key, "token_match"
                elif tok == "sbi":
                    return True, entity_key, "token_match"
                else:
                    return True, entity_key, "token_match"

    return False, None, "none"
