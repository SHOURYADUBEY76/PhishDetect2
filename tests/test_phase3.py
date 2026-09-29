"""Automated verification suite for Phase 3 components and experimental pipelines.
Validates neural models, adversarial generators, latency profilers, and data integrity.
"""

import os
import sys
import pytest
import numpy as np
import pandas as pd

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models.bilstm import CharacterTokenizer, BiLSTMClassifier
from src.models.baseline import PhishingClassifier
from src.utils.metrics import measure_inference_latency, measure_detailed_latencies, measure_pipeline_latency
from src.utils.domain import extract_registered_domain
from experiments.E06_adversarial import perturb_url_adversarial, HOMOGLYPH_REPLACEMENTS


def test_character_tokenizer():
    tokenizer = CharacterTokenizer(max_length=50)
    urls = ["https://sbi-verification.xyz/login", "http://axis-bank.in"]
    seqs = tokenizer.texts_to_sequences(urls)
    
    assert seqs.shape == (2, 50)
    assert np.all(seqs >= 0)
    assert np.all(seqs < tokenizer.vocab_size)
    # Check that padding (0) exists at the tail
    assert seqs[0][-1] == 0


def test_bilstm_classifier():
    clf = BiLSTMClassifier(max_length=60, lstm_units=16, embedding_dim=16)
    train_urls = [
        "https://google.com/search?q=news",
        "https://amazon.com/products/view",
        "http://sbi-login-verify-urgent.xyz/auth",
        "https://hdfc-reward-points-claim.top/login"
    ]
    y_train = np.array([0, 0, 1, 1])
    clf.fit(train_urls, y_train, epochs=1, batch_size=2)
    
    probs = clf.predict_proba_urls(train_urls)
    assert probs.shape == (4, 2)
    assert np.all(probs >= 0.0) and np.all(probs <= 1.0)
    assert np.allclose(np.sum(probs, axis=1), 1.0, atol=1e-5)


def test_adversarial_perturbations():
    base_url = "https://sbi-portal.com/login"
    
    # Homoglyph
    perturbed_hg = perturb_url_adversarial(base_url, technique="homoglyph")
    assert perturbed_hg != base_url or any(c in base_url for c in HOMOGLYPH_REPLACEMENTS)
    
    # Token padding
    perturbed_pad = perturb_url_adversarial(base_url, technique="token_padding")
    assert any(tok in perturbed_pad for tok in ["secure-login", "auth-verify", "update-portal", "account-center", "portal-gateway"])
    
    # Subdomain trick
    perturbed_sub = perturb_url_adversarial(base_url, technique="subdomain_trick")
    assert "auth-service-" in perturbed_sub


def test_latency_and_size_metrics():
    clf = PhishingClassifier(model_type="logistic_regression", random_state=42)
    X = np.random.randn(20, 47)
    y = np.random.randint(0, 2, size=20)
    clf.fit(X, y)
    
    # Model size must be non-zero
    size_mb = clf.get_model_size_mb()
    assert size_mb > 0.0
    
    # Latency measurement
    lat = measure_inference_latency(clf, X, n_iterations=20, warmup_iterations=5)
    assert lat > 0.0
    
    detailed = measure_detailed_latencies(clf, X, n_iterations=20, warmup_iterations=5)
    assert "latency_mean_ms" in detailed
    assert "latency_p95_ms" in detailed
    assert detailed["latency_p95_ms"] >= detailed["latency_mean_ms"]


def test_processed_splits_zero_leakage():
    gtrain = pd.read_csv("data/processed/global_train.csv")
    gtest = pd.read_csv("data/processed/global_test.csv")
    itest = pd.read_csv("data/processed/indian_test.csv")
    itrain = pd.read_csv("data/processed/indian_train.csv")
    
    gtrain_doms = set(gtrain["registered_domain"].dropna().unique())
    gtest_doms = set(gtest["registered_domain"].dropna().unique())
    itest_doms = set(itest["registered_domain"].dropna().unique())
    itrain_doms = set(itrain["registered_domain"].dropna().unique())
    
    assert len(gtrain_doms.intersection(gtest_doms)) == 0, "Global train and test must have zero domain overlap"
    assert len(gtrain_doms.intersection(itest_doms)) == 0, "Global train and Indian test must have zero domain overlap"
    assert len(itrain_doms.intersection(itest_doms)) == 0, "Indian train and Indian test must have zero domain overlap"
