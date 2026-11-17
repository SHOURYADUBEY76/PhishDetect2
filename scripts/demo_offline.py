"""Offline Demonstration & CLI Inspector for Region-Aware Phishing URL Detector.
Performs 100% offline static URL analysis with zero network, DNS, or WHOIS dependencies.
Evaluates threat level, phishing probability, and plain-English explainability in <3ms.
"""

import os
import sys
import time
import argparse
import numpy as np

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Reconfigure stdout to UTF-8 for Windows console support with Unicode/homoglyphs
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src.features.extractor import extract_features_dict, get_feature_names
from src.features.regional import OFFICIAL_DOMAINS, INDIAN_BRANDS
from src.models.baseline import PhishingClassifier
from src.explainer.explainer import PhishingExplainer
from src.utils.domain import normalize_url, extract_registered_domain, extract_tld

# Standard authoritative global web properties
GLOBAL_SAFE_DOMAINS = {
    "google.com", "github.com", "microsoft.com", "apple.com", "amazon.com",
    "wikipedia.org", "youtube.com", "linkedin.com", "cloudflare.com"
}


OFFLINE_SHOWCASE_CASES = [
    {
        "category": "Official Legitimate Indian Banking",
        "url": "https://www.onlinesbi.sbi",
        "expected": "Legitimate",
        "notes": "Legitimate State Bank of India portal on official .sbi gTLD"
    },
    {
        "category": "Official Legitimate Indian Government",
        "url": "https://uidai.gov.in/en/my-aadhaar/get-aadhaar.html",
        "expected": "Legitimate",
        "notes": "Official Unique Identification Authority of India on .gov.in"
    },
    {
        "category": "Official Legitimate Global Property",
        "url": "https://github.com/explore",
        "expected": "Legitimate",
        "notes": "Global developer platform on registered .com domain"
    },
    {
        "category": "Blatant Brand Impersonation Lure",
        "url": "http://sbi-kyc-verification-update-urgent.xyz/login.php",
        "expected": "Phishing",
        "notes": "Target brand 'sbi' misaligned with deceptive .xyz TLD and urgent KYC lure"
    },
    {
        "category": "Government Scheme Phishing (Ayushman Bharat)",
        "url": "http://ayushman-bharat-goldencard-apply-8.top/beneficiary/download-card.php",
        "expected": "Phishing",
        "notes": "Exploiting health welfare scheme with .top TLD and credential harvesting path"
    },
    {
        "category": "Adversarial Subdomain Injection",
        "url": "http://hdfcbank.com.auth-service-92.buzz/netbanking/secure/verify",
        "expected": "Phishing",
        "notes": "Prepending brand domain as subdomain of an attacker root (.buzz)"
    },
    {
        "category": "Adversarial Homoglyph Attack",
        "url": "https://www.ісісіbаnk.com/login",
        "expected": "Phishing",
        "notes": "Hostname uses Cyrillic 'і' and 'а' lookalikes to visually spoof icicibank"
    },
    {
        "category": "Zero-Shot AI Lure (PM Surya Ghar)",
        "url": "http://pmsuryaghar-gov-subsidy-claim-1.top/apply/kyc-verify.php?state=delhi",
        "expected": "Phishing",
        "notes": "Synthetic lure targeting newly launched 2024 solar subsidy scheme"
    },
    {
        "category": "Zero-Shot AI Lure (Digi Yatra Biometrics)",
        "url": "https://digiyatra-airport-fastpass-enroll-3.cloud/passenger-identity/verify",
        "expected": "Phishing",
        "notes": "Synthetic lure targeting airport digital boarding credential harvesting"
    },
    {
        "category": "Emerging Neobank Phishing",
        "url": "https://fi-money-neobank-account-unblock-5.site/kyc-center/login",
        "expected": "Phishing",
        "notes": "Financial lure impersonating emerging fintech neobank"
    }
]


def run_offline_inspection(url: str, clf: PhishingClassifier, explainer: PhishingExplainer) -> dict:
    """Run 100% offline static evaluation on a single URL string."""
    t0 = time.perf_counter()
    url_clean = normalize_url(url)
    reg_dom = extract_registered_domain(url_clean)
    tld = extract_tld(url_clean)

    # 1. Feature extraction (completely offline, zero network)
    t_feat0 = time.perf_counter()
    feat_dict = extract_features_dict(url_clean, include_regional=True, include_path_query=True)
    cols = get_feature_names(include_regional=True)
    vec = np.array([[feat_dict[c] for c in cols]])
    t_feat1 = time.perf_counter()

    # 2. Model prediction
    t_inf0 = time.perf_counter()
    probs = clf.predict_proba(vec)[0]
    p_phish = float(probs[1])
    is_phish = bool(p_phish >= clf.optimal_threshold)
    t_inf1 = time.perf_counter()

    # Check official registry safelist
    is_safelisted = False
    if reg_dom in GLOBAL_SAFE_DOMAINS or reg_dom.endswith((".gov.in", ".nic.in")):
        is_safelisted = True
        is_phish = False
        p_phish = min(p_phish, 0.02)
    else:
        for brand, doms in OFFICIAL_DOMAINS.items():
            if reg_dom in doms:
                is_safelisted = True
                is_phish = False
                p_phish = min(p_phish, 0.02)
                break

    if not is_safelisted and feat_dict.get("has_homoglyph_or_punycode", 0.0) > 0.5:
        p_phish = max(p_phish, 0.98)
        is_phish = True

    # 3. Rule-based explanation
    reasons = explainer.explain(
        url=url_clean,
        features=feat_dict,
        risk_score=p_phish,
        is_phishing=is_phish
    )

    if p_phish >= 0.85:
        risk_level = "CRITICAL"
    elif p_phish >= 0.65:
        risk_level = "HIGH"
    elif p_phish >= 0.40:
        risk_level = "MEDIUM"
    else:
        risk_level = "LOW"

    # Identify any matched brand
    matched_brand = None
    for b in INDIAN_BRANDS:
        if b in url_clean.lower():
            matched_brand = b
            break

    t_total = (time.perf_counter() - t0) * 1000.0

    return {
        "url": url,
        "normalized_url": url_clean,
        "registered_domain": reg_dom,
        "tld": tld,
        "is_phishing": is_phish,
        "is_safelisted": is_safelisted,
        "phishing_prob": p_phish,
        "risk_level": risk_level,
        "reasons": reasons,
        "matched_brand": matched_brand,
        "feature_extraction_ms": (t_feat1 - t_feat0) * 1000.0,
        "inference_ms": (t_inf1 - t_inf0) * 1000.0,
        "total_latency_ms": t_total
    }


def print_verdict_card(result: dict, index: int = None, category: str = None, notes: str = None):
    """Render a clean ASCII formatted threat card in terminal."""
    is_phish = result["is_phishing"]
    badge = "[ DANGER / PHISHING ]" if is_phish else "[ SAFE / LEGITIMATE ]"
    border_char = "=" if is_phish else "-"
    
    print("\n" + border_char * 78)
    if index is not None and category:
        print(f" CASE #{index}: {category.upper()}")
        if notes:
            print(f" Context: {notes}")
        print(border_char * 78)

    print(f" URL             : {result['url']}")
    print(f" Registered Dom  : {result['registered_domain']}  (TLD: .{result['tld']})")
    print(f" Verdict         : {badge}")
    print(f" Phishing Risk   : {result['phishing_prob']*100:5.2f}% | Risk Level: {result['risk_level']}")
    print(f" Offline Latency : {result['total_latency_ms']:.2f} ms (Feature: {result['feature_extraction_ms']:.2f}ms, Inference: {result['inference_ms']:.2f}ms)")
    
    if result["is_safelisted"]:
        print(" Official Status : Verified Official Registry Entity (Safelisted)")
    elif result["matched_brand"]:
        print(f" Target Entity   : Impersonating '{result['matched_brand'].upper()}'")

    print("\n Plain-English Threat Explanations:")
    for reason in result["reasons"]:
        print(f"   * {reason}")
    print(border_char * 78)


def main():
    parser = argparse.ArgumentParser(description="Offline Phishing URL Detector Inspector")
    parser.add_argument("--url", type=str, help="Specific URL to evaluate offline")
    parser.add_argument("--interactive", action="store_true", help="Launch interactive CLI prompt")
    args = parser.parse_args()

    model_path = os.path.join(os.path.dirname(__file__), "..", "backend", "model_region_aware.joblib")
    if not os.path.exists(model_path):
        model_path = os.path.join(os.path.dirname(__file__), "..", "backend", "model_baseline.joblib")

    print("\n" + "=" * 78)
    print("   PHISHDETECT: AI-DRIVEN PHISHING URL DETECTION (100% OFFLINE ENGINE)")
    print("   Zero Network / Zero DNS / Zero WHOIS / Sub-Millisecond Analysis")
    print("=" * 78)
    print(f"Loading local model: {os.path.basename(model_path)}...", flush=True)

    clf = PhishingClassifier.load(model_path)
    explainer = PhishingExplainer()
    print("Model initialized and ready for offline inference.\n", flush=True)

    if args.url:
        res = run_offline_inspection(args.url, clf, explainer)
        print_verdict_card(res)
        return

    if args.interactive:
        print("Entering interactive mode. Enter any URL to inspect (or 'exit' to quit):")
        while True:
            try:
                raw = input("\nInspect URL > ").strip()
                if raw.lower() in ["exit", "quit", "q"]:
                    break
                if not raw:
                    continue
                res = run_offline_inspection(raw, clf, explainer)
                print_verdict_card(res)
            except (KeyboardInterrupt, EOFError):
                break
        print("\nExiting interactive inspector.")
        return

    # Default: Run curated showcase battery
    print(f"Evaluating {len(OFFLINE_SHOWCASE_CASES)} diverse test cases across official and malicious categories...\n")
    latencies = []
    
    for idx, item in enumerate(OFFLINE_SHOWCASE_CASES, 1):
        res = run_offline_inspection(item["url"], clf, explainer)
        latencies.append(res["total_latency_ms"])
        print_verdict_card(res, index=idx, category=item["category"], notes=item["notes"])

    avg_lat = np.mean(latencies)
    print("\n" + "=" * 78)
    print("                      OFFLINE SHOWCASE SUMMARY")
    print("=" * 78)
    print(f" Total Cases Evaluated   : {len(OFFLINE_SHOWCASE_CASES)}")
    print(f" External Network Calls  : 0 (Strictly Offline String Parsing)")
    print(f" Average Total Latency   : {avg_lat:.2f} ms / URL")
    print(f" Model In-Memory Size    : {clf.get_model_size_mb():.2f} MB")
    print("=" * 78 + "\n")


if __name__ == "__main__":
    main()
