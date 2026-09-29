"""Unit and integration tests for Phase 1 components.
Verifies feature extraction, model prediction, threshold tuning, and FastAPI endpoint.
"""

import os
import sys
import unittest
import numpy as np

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.features.extractor import extract_features_dict, extract_features_vector, get_feature_names
from src.models.baseline import PhishingClassifier
from fastapi.testclient import TestClient
from backend.main import app


class TestPhase1Pipeline(unittest.TestCase):

    def test_feature_extraction(self):
        url = "https://secure-login.example.com/account/verify?user=test"
        feat_dict = extract_features_dict(url, include_regional=False)
        self.assertIn("url_length", feat_dict)
        self.assertIn("num_subdomains", feat_dict)
        self.assertIn("url_entropy", feat_dict)
        self.assertEqual(feat_dict["is_https"], 1.0)
        self.assertEqual(feat_dict["has_ip"], 0.0)

        vec = extract_features_vector(url, include_regional=False)
        self.assertEqual(len(vec), len(get_feature_names(include_regional=False)))
        self.assertFalse(np.isnan(vec).any(), "Feature vector contains NaN values")

    def test_model_loading_and_inference(self):
        model_path = "backend/model_baseline.joblib"
        self.assertTrue(os.path.exists(model_path), "Trained baseline model not found on disk")
        clf = PhishingClassifier.load(model_path)
        self.assertIsNotNone(clf.model)
        
        sample_vec = np.zeros((1, len(get_feature_names(include_regional=False))), dtype=np.float32)
        probs = clf.predict_proba(sample_vec)
        self.assertEqual(probs.shape, (1, 2))
        self.assertGreaterEqual(probs[0, 1], 0.0)
        self.assertLessEqual(probs[0, 1], 1.0)

    def test_fastapi_endpoints(self):
        client = TestClient(app)
        
        # Health endpoint
        res_health = client.get("/health")
        self.assertEqual(res_health.status_code, 200)
        data_health = res_health.json()
        self.assertEqual(data_health["status"], "healthy")
        self.assertTrue(data_health["model_loaded"])

        # Warmup call
        _ = client.post("/predict", json={"url": "https://example.com"})

        # Predict endpoint on benign URL
        res_benign = client.post("/predict", json={"url": "https://www.google.com"})
        self.assertEqual(res_benign.status_code, 200)
        data_benign = res_benign.json()
        self.assertIn("verdict", data_benign)
        self.assertIn("risk_score", data_benign)
        self.assertIn("latency_ms", data_benign)
        self.assertLess(data_benign["latency_ms"], 50.0, f"Inference latency ({data_benign['latency_ms']}ms) exceeded 50ms SLA")

        # Predict endpoint on phishing candidate
        res_phish = client.post("/predict", json={"url": "http://192.168.1.1/login/verify.php?token=fake"})
        self.assertEqual(res_phish.status_code, 200)
        data_phish = res_phish.json()
        self.assertIn("verdict", data_phish)
        self.assertTrue(data_phish["key_signals"]["has_ip"])


if __name__ == "__main__":
    unittest.main()
