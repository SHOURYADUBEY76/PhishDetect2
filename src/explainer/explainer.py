"""Lightweight, human-readable explanation layer for phishing URL detection.
Translates mathematical feature attributions and rule heuristics into plain-English reasons
without requiring a slow or heavy LLM.
"""

from typing import List, Dict, Any
from urllib.parse import urlparse
from src.features.regional import (
    INDIAN_BRANDS, INDIAN_LURES, OFFICIAL_DOMAINS, INDIAN_TLDS, BRAND_ALIAS_MAP
)
from src.utils.domain import extract_registered_domain, extract_tld, extract_subdomain


class PhishingExplainer:
    """Generates human-readable explanations based on extracted threat signals and model weights."""

    def __init__(self, model=None):
        self.model = model

    def explain(self, url: str, features: Dict[str, float], risk_score: float, is_phishing: bool) -> List[str]:
        """Generate 2 to 4 human-readable justification bullet points for a verdict.
        
        Args:
            url: Evaluated URL string.
            features: Dictionary of extracted features.
            risk_score: Continuous probability of being phishing.
            is_phishing: Binary verdict flag.
            
        Returns:
            List of concise, plain-English explanation strings.
        """
        reasons = []
        parsed = urlparse(url if "://" in url else f"http://{url}")
        hostname = (parsed.netloc or "").split(":")[0].lower()
        path = parsed.path.lower()
        reg_domain = extract_registered_domain(url)
        tld = extract_tld(url)

        if is_phishing:
            # 1. Check brand-TLD misalignment (highest priority threat indicator)
            if features.get("brand_tld_misalignment", 0.0) > 0.5:
                # Find which brand triggered it
                matched_b = [b for b in INDIAN_BRANDS if b in hostname or b in path]
                brand_label = matched_b[0].upper() if matched_b else "Indian institution"
                reasons.append(
                    f"Brand Impersonation: Displays '{brand_label}' lure, but '{reg_domain}' is not an official authorized portal."
                )

            # 2. Check Indian action lure keywords
            if features.get("has_indian_lure", 0.0) > 0.5:
                matched_l = [l for l in INDIAN_LURES if l in path or l in hostname]
                lure_str = f"'{matched_l[0]}'" if matched_l else "urgent action"
                reasons.append(
                    f"Credential Lure: Contains high-risk lure keyword ({lure_str}) frequently used in regional financial phishing."
                )

            # 3. Check IP address in hostname
            if features.get("has_ip", 0.0) > 0.5:
                reasons.append("Raw IP Address: Hostname uses a direct IP instead of a registered domain, a common evasion tactic.")

            # 3b. Check homoglyph or punycode
            if features.get("has_homoglyph_or_punycode", 0.0) > 0.5:
                reasons.append("Homoglyph Obfuscation: Uses deceptive non-ASCII character lookalikes or punycode in hostname to visually spoof a legitimate domain.")

            # 4. Check excessive subdomains
            if features.get("num_subdomains", 0.0) >= 3:
                reasons.append(f"Subdomain Masking: Excessive subdomain depth ({int(features['num_subdomains'])} levels) used to conceal the actual destination domain.")

            # 5. Non-standard port or unencrypted
            if features.get("is_non_standard_port", 0.0) > 0.5:
                reasons.append("Non-Standard Port: The connection routes through an unusual network port.")
            elif features.get("is_http", 0.0) > 0.5:
                reasons.append("Insecure Protocol: Transmits over unencrypted HTTP without TLS verification.")

            # 6. High entropy or length
            if features.get("url_entropy", 0.0) > 4.5:
                reasons.append("Obfuscation Detected: High character randomness and entropy suggestive of auto-generated attack paths.")
            elif features.get("url_length", 0.0) > 85:
                reasons.append(f"Abnormal Length: Unusually long URL ({int(features['url_length'])} chars) with dense query token padding.")

            # Fallback if specific heuristics were subtle
            if not reasons:
                reasons.append("Statistical Anomaly: High concentration of structural and lexical signals typical of credential harvesting links.")

        else:
            # Benign explanation
            # Check official registry
            is_official = False
            for brand, official_set in OFFICIAL_DOMAINS.items():
                if reg_domain in official_set:
                    reasons.append(f"Official Domain: Verified as an authentic registered portal for {brand.upper()}.")
                    is_official = True
                    break

            if not is_official:
                if features.get("is_indian_tld", 0.0) > 0.5:
                    reasons.append(f"Standard Regional Domain: Established '.{tld}' top-level domain with no brand impersonation markers.")
                else:
                    reasons.append(f"Standard Domain Structure: Registered under legitimate '{reg_domain}' host.")

            if features.get("is_https", 0.0) > 0.5:
                reasons.append("Secure Transport: Valid standard HTTPS protocol configuration.")

            if features.get("num_subdomains", 0.0) <= 1:
                reasons.append("Clean Architecture: Direct domain routing with standard hierarchical depth.")

            reasons.append(f"Low Threat Metric: Extracted lexical and structural markers align with benign traffic (Risk: {risk_score*100:.1f}%).")

        return reasons[:4]
