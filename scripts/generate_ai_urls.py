"""AI-Generated Phishing URL Generator with Explicit Provenance and Zero-Leakage Guarantee.
Uses a documented red-team prompt targeting novel Indian digital initiatives and financial lures.
Supports live LLM generation via GEMINI_API_KEY / OPENAI_API_KEY with clean separation
between genuine LLM outputs (data/raw/ai_phishing_llm.csv) and template stand-ins (data/raw/ai_phishing_standin.csv).
"""

import os
import sys
import datetime
import pandas as pd

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.utils.domain import normalize_url, extract_registered_domain


# Documented fixed prompt targeting novel/unseen Indian digital initiatives and red-team evasion
FIXED_GENERATION_PROMPT = """
You are a cybersecurity red-team researcher assessing URL detection robustness against emerging distribution shifts.
Generate 50 realistic, deceptive phishing URLs targeting novel or emerging Indian public and financial services:
1. PM Surya Ghar Muft Bijli Yojana (solar subsidy lures)
2. Digi Yatra airport enrollment lures
3. Ayushman Bharat golden card download lures
4. Kisan Credit card relief portal lures
5. RuPay Express Rewards and cashback claim portals
6. Emerging Indian Neobanks (e.g., Fi, Jupiter, Niyo)

Apply modern adversarial evasion techniques:
- Multi-level brand padding in subdomains
- Typosquatting and hyphen insertion
- Urgent credential harvesting paths (/kyc-status/aadhaar-verify/auth.php)
- Generic misaligned TLDs (.top, .xyz, .site, .cloud, .link, .buzz, .info)

Output strictly valid URLs, one per line. Do not include markdown or explanations.
"""

# High-fidelity red-team evasion templates targeting novel/unseen Indian initiatives
NOVEL_UNSEEN_TEMPLATES = [
    # 1. PM Surya Ghar Solar Subsidy Lures
    "http://pmsuryaghar-gov-subsidy-claim-{idx}.top/apply/kyc-verify.php?state=delhi",
    "https://pm-suryaghar-muftbijli-portal-{idx}.xyz/registration/bank-account/update",
    "http://surya-ghar-yojana-beneficiary-{idx}.site/auth/aadhaar-link.html?ref=gov",
    # 2. Digi Yatra Lures
    "https://digiyatra-airport-fastpass-enroll-{idx}.cloud/passenger-identity/verify",
    "http://digi-yatra-credential-verification-{idx}.link/facial-biometric/confirm.jsp",
    "https://digiyatra-official-pass-download-{idx}.buzz/boarding-pass/auth?token=98{idx}",
    # 3. Ayushman Bharat Golden Card Lures
    "http://ayushman-bharat-goldencard-apply-{idx}.top/beneficiary/download-card.php",
    "https://pmjay-ayushman-hospital-scheme-{idx}.info/claim-insurance/upload-pan",
    "http://ayushman-card-free-treatment-{idx}.site/portal/verify-eligibility.html",
    # 4. Kisan Credit Portal Lures
    "https://pm-kisan-samman-nidhi-urgent-{idx}.net/installment-credit/bank-update.php",
    "http://kisan-credit-card-waiver-dept-{idx}.org/farmer-relief/status-check?id={idx}",
    # 5. RuPay Express Rewards
    "https://rupay-platinum-cashback-claim-{idx}.xyz/rewards/redeem-points.php",
    "http://rupay-card-instant-settlement-{idx}.live/verify-card-pin?session=active",
    # 6. Emerging Neobanks & Novel Fintech Lures
    "https://fi-money-neobank-account-unblock-{idx}.site/kyc-center/login",
    "http://jupiter-edge-card-reward-portal-{idx}.club/claim-bonus/auth",
    "https://niyo-global-forex-travel-card-{idx}.link/activate-card/verify-passport"
]


def generate_ai_phishing_urls(n_samples: int = 250, force_regenerate: bool = False) -> pd.DataFrame:
    """Generate or retrieve AI phishing URLs with strict provenance tracking."""
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("OPENAI_API_KEY")
    today_str = datetime.date.today().isoformat()
    os.makedirs("data/raw", exist_ok=True)

    llm_path = "data/raw/ai_phishing_llm.csv"
    standin_path = "data/raw/ai_phishing_standin.csv"

    # 1. Attempt Live LLM Generation if API Key is available
    if api_key:
        if os.path.exists(llm_path) and not force_regenerate:
            print(f"[AI URLs] Loading existing live LLM-generated URLs from {llm_path}", flush=True)
            return pd.read_csv(llm_path)

        print("[AI URLs] API key detected. Querying Gemini LLM with documented red-team prompt...", flush=True)
        try:
            import google.generativeai as genai
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel("gemini-1.5-flash")
            response = model.generate_content(FIXED_GENERATION_PROMPT)
            raw_lines = [l.strip() for l in response.text.splitlines() if l.strip().startswith(("http://", "https://"))]

            if len(raw_lines) >= 15:
                df_llm = pd.DataFrame({
                    "url": [normalize_url(u) for u in raw_lines],
                    "label": 1,
                    "date": today_str,
                    "provenance": "llm_gemini_api",
                    "is_standin": False,
                    "prompt_used": FIXED_GENERATION_PROMPT.strip()
                })
                df_llm["registered_domain"] = df_llm["url"].apply(extract_registered_domain)
                df_llm.drop_duplicates(subset=["url"], inplace=True)
                df_llm.to_csv(llm_path, index=False)
                print(f"[AI URLs] Successfully generated and saved {len(df_llm)} LLM URLs to {llm_path}", flush=True)
                return df_llm
        except Exception as e:
            print(f"[AI URLs Warning] Live LLM call failed: {e}. Falling back to stand-in set.", flush=True)

    # 2. Generate or Load Heuristic Red-Team Stand-In Set
    if os.path.exists(standin_path) and not force_regenerate:
        print(f"[AI URLs] Loading existing stand-in dataset from {standin_path}", flush=True)
        return pd.read_csv(standin_path)

    print("[AI URLs Notice] Generating structured red-team heuristic stand-in set (marked: template_standin)...", flush=True)
    generated_urls = []
    for idx in range(1, (n_samples // len(NOVEL_UNSEEN_TEMPLATES)) + 3):
        for tpl in NOVEL_UNSEEN_TEMPLATES:
            if len(generated_urls) < n_samples:
                generated_urls.append(tpl.format(idx=idx))

    df_standin = pd.DataFrame({
        "url": [normalize_url(u) for u in generated_urls],
        "label": 1,
        "date": today_str,
        "provenance": "synthetic_template_standin",
        "is_standin": True,
        "prompt_used": FIXED_GENERATION_PROMPT.strip()
    })
    df_standin["registered_domain"] = df_standin["url"].apply(extract_registered_domain)
    df_standin.drop_duplicates(subset=["url"], inplace=True)
    df_standin.to_csv(standin_path, index=False)
    print(f"[AI URLs] Saved {len(df_standin)} stand-in URLs (provenance=synthetic_template_standin) to {standin_path}", flush=True)
    return df_standin


if __name__ == "__main__":
    generate_ai_phishing_urls()
