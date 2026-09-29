"""FastAPI REST Service for Real-Time Region-Aware Phishing URL Detection.
Serves verdicts, risk scores, confidence ratings, and plain-English explanations.
"""

import os
import sys
import time
from typing import Optional, List, Dict, Any
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from contextlib import asynccontextmanager

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.features.extractor import (
    extract_features_dict,
    extract_features_vector,
    get_feature_names
)
from src.features.regional import OFFICIAL_DOMAINS
from src.models.baseline import PhishingClassifier
from src.explainer.explainer import PhishingExplainer
from src.utils.domain import normalize_url, extract_registered_domain, extract_tld


# Model configuration paths (prefer region-aware model)
BASE_DIR = os.path.dirname(__file__)
MODEL_REGION_AWARE = os.path.join(BASE_DIR, "model_region_aware.joblib")
MODEL_BASELINE = os.path.join(BASE_DIR, "model_baseline.joblib")

_classifier: Optional[PhishingClassifier] = None
_explainer = PhishingExplainer()


def get_classifier() -> Optional[PhishingClassifier]:
    """Retrieve or lazily load the trained classifier (prefers region-aware model)."""
    global _classifier
    if _classifier is None:
        target_path = MODEL_REGION_AWARE if os.path.exists(MODEL_REGION_AWARE) else MODEL_BASELINE
        if os.path.exists(target_path):
            try:
                _classifier = PhishingClassifier.load(target_path)
                print(f"[API] Loaded model from {target_path}", flush=True)
            except Exception as e:
                print(f"[API Warning] Failed to load model from {target_path}: {e}", flush=True)
    return _classifier


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Warmup model on server startup."""
    get_classifier()
    yield


app = FastAPI(
    title="Region-Aware Phishing URL Detector API",
    description="Real-time static URL analysis engine with regional threat adaptation and lightweight explainability.",
    version="2.0.0",
    lifespan=lifespan
)

# Enable CORS for Vite React frontend and Chrome Manifest V3 extensions
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class PredictRequest(BaseModel):
    url: str = Field(..., description="Target URL string to evaluate")
    include_regional: bool = Field(default=True, description="Whether to include regional feature signals")


class PredictResponse(BaseModel):
    url: str
    registered_domain: str
    tld: str
    is_phishing: bool
    verdict: str
    risk_score: float
    confidence: float
    decision_threshold: float
    latency_ms: float
    reasons: List[str]
    key_signals: Dict[str, Any]


@app.get("/health")
def health_check():
    """Health and model status probe."""
    clf = get_classifier()
    model_name = "model_region_aware" if os.path.exists(MODEL_REGION_AWARE) else ("model_baseline" if os.path.exists(MODEL_BASELINE) else None)
    return {
        "status": "healthy",
        "model_loaded": clf is not None,
        "active_model": model_name,
        "model_type": clf.model_type if clf else None,
        "calibrated_threshold": clf.optimal_threshold if clf else 0.5
    }


@app.post("/predict", response_model=PredictResponse)
def predict_url(payload: PredictRequest):
    """Analyze static URL string and return classification verdict with explanations."""
    if not payload.url or not payload.url.strip():
        raise HTTPException(status_code=400, detail="URL parameter cannot be empty.")

    clf = get_classifier()
    if clf is None:
        raise HTTPException(status_code=503, detail="Model is not initialized or trained yet.")

    start_time = time.perf_counter()
    url_norm = normalize_url(payload.url)
    registered_domain = extract_registered_domain(url_norm)
    tld = extract_tld(url_norm)

    # Extract features using shared feature pipeline
    feature_dict = extract_features_dict(
        url_norm,
        include_regional=payload.include_regional,
        include_path_query=True
    )
    feature_vector = extract_features_vector(
        url_norm,
        include_regional=payload.include_regional,
        include_path_query=True
    ).reshape(1, -1)

    # Model inference
    probs = clf.predict_proba(feature_vector)[0]
    risk_score = float(probs[1])
    threshold = float(clf.optimal_threshold)

    # Verified official portal check (standard industry safe-listing for authenticated institutions)
    is_verified_official = (
        any(registered_domain in domains for domains in OFFICIAL_DOMAINS.values())
        or registered_domain.endswith((".gov.in", ".nic.in"))
    ) and feature_dict.get("brand_tld_misalignment", 0.0) == 0.0 and feature_dict.get("has_ip", 0.0) == 0.0

    if is_verified_official:
        risk_score = min(risk_score, 0.02)
    elif feature_dict.get("has_homoglyph_or_punycode", 0.0) > 0.5:
        risk_score = max(risk_score, 0.98)

    is_phish = bool(risk_score >= threshold)
    verdict = "phishing" if is_phish else "legitimate"

    # Confidence calculation relative to threshold
    if is_phish:
        confidence = float(min(1.0, max(0.5, (risk_score - threshold) / (1.0 - threshold + 1e-6))))
    else:
        confidence = float(min(1.0, max(0.5, (threshold - risk_score) / (threshold + 1e-6))))

    # Plain-English human-readable reasons
    reasons = _explainer.explain(
        url=url_norm,
        features=feature_dict,
        risk_score=risk_score,
        is_phishing=is_phish
    )

    elapsed_ms = (time.perf_counter() - start_time) * 1000.0

    key_signals = {
        "url_length": feature_dict.get("url_length", 0.0),
        "num_subdomains": feature_dict.get("num_subdomains", 0.0),
        "has_ip": bool(feature_dict.get("has_ip", 0.0)),
        "is_https": bool(feature_dict.get("is_https", 0.0)),
        "suspicious_keyword_count": feature_dict.get("suspicious_keyword_count", 0.0),
        "url_entropy": round(feature_dict.get("url_entropy", 0.0), 3),
        "brand_tld_misalignment": bool(feature_dict.get("brand_tld_misalignment", 0.0)),
        "has_indian_brand_token": bool(feature_dict.get("has_indian_brand_token", 0.0)),
        "has_indian_lure": bool(feature_dict.get("has_indian_lure", 0.0)),
    }

    return PredictResponse(
        url=url_norm,
        registered_domain=registered_domain,
        tld=tld,
        is_phishing=is_phish,
        verdict=verdict,
        risk_score=round(risk_score, 4),
        confidence=round(confidence, 4),
        decision_threshold=round(threshold, 4),
        latency_ms=round(elapsed_ms, 3),
        reasons=reasons,
        key_signals=key_signals
    )


@app.post("/explain")
def explain_url(payload: PredictRequest):
    """Dedicated endpoint returning detailed feature attribution and reasons."""
    pred = predict_url(payload)
    return {
        "url": pred.url,
        "verdict": pred.verdict,
        "risk_score": pred.risk_score,
        "reasons": pred.reasons,
        "key_signals": pred.key_signals,
        "latency_ms": pred.latency_ms
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host="127.0.0.1", port=8000, reload=True)
