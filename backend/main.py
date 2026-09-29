"""FastAPI REST Service for Real-Time Phishing URL Detection.
Serves verdicts, risk scores, confidence, and latency to web tools and browser extensions.
"""

import os
import sys
import time
from typing import Optional, List, Dict, Any
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.features.extractor import (
    extract_features_dict,
    extract_features_vector,
    get_feature_names
)
from src.models.baseline import PhishingClassifier
from src.utils.domain import normalize_url, extract_registered_domain, extract_tld


app = FastAPI(
    title="Region-Aware Phishing URL Detector API",
    description="Real-time static URL analysis engine with regional threat adaptation.",
    version="1.0.0"
)

# Enable CORS for Vite React frontend and Chrome Manifest V3 extensions
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model state
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model_baseline.joblib")
_classifier: Optional[PhishingClassifier] = None


def get_classifier() -> Optional[PhishingClassifier]:
    """Retrieve or lazily load the trained classifier."""
    global _classifier
    if _classifier is None and os.path.exists(MODEL_PATH):
        try:
            _classifier = PhishingClassifier.load(MODEL_PATH)
            print(f"[API] Loaded baseline model from {MODEL_PATH}", flush=True)
        except Exception as e:
            print(f"[API Warning] Failed to load model: {e}", flush=True)
    return _classifier


@app.on_event("startup")
def startup_event():
    """Warmup model on server startup."""
    get_classifier()


class PredictRequest(BaseModel):
    url: str = Field(..., description="Target URL string to evaluate")
    include_regional: bool = Field(default=False, description="Whether to include regional feature signals")


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
    key_signals: Dict[str, Any]


@app.get("/health")
def health_check():
    """Health and model status probe."""
    clf = get_classifier()
    return {
        "status": "healthy",
        "model_loaded": clf is not None,
        "model_type": clf.model_type if clf else None,
        "calibrated_threshold": clf.optimal_threshold if clf else 0.5
    }


@app.post("/predict", response_model=PredictResponse)
def predict_url(payload: PredictRequest):
    """Analyze static URL string and return classification verdict."""
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

    # Inference
    probs = clf.predict_proba(feature_vector)[0]
    risk_score = float(probs[1])
    threshold = float(clf.optimal_threshold)
    is_phish = bool(risk_score >= threshold)
    verdict = "phishing" if is_phish else "legitimate"

    # Confidence calculation relative to threshold
    if is_phish:
        confidence = float(min(1.0, max(0.5, (risk_score - threshold) / (1.0 - threshold + 1e-6))))
    else:
        confidence = float(min(1.0, max(0.5, (threshold - risk_score) / (threshold + 1e-6))))

    elapsed_ms = (time.perf_counter() - start_time) * 1000.0

    # Key signals for client UI
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
        key_signals=key_signals
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host="127.0.0.1", port=8000, reload=True)
