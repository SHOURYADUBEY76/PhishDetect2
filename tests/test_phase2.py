"""Phase 2 verification test suite.
Validates region-aware features, explainer output, ablation results, and API endpoints.
"""

import os
import sys
import unittest

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.features.regional import extract_regional_features
from src.features.extractor import extract_features_dict
from src.explainer.explainer import PhishingExplainer
from fastapi.testclient import TestClient
from backend.main import app


class TestPhase2Pipeline(unittest.TestCase):

    def test_regional_feature_extraction(self):
        phish_url = "http://sbi-kyc-update-login.top/account/verify.php"
        reg_feats = extract_regional_features(phish_url)
        self.assertEqual(reg_feats["has_indian_brand_token"], 1.0)
        self.assertEqual(reg_feats["has_indian_lure"], 1.0)
        self.assertEqual(reg_feats["brand_tld_misalignment"], 1.0)

        legit_url = "https://onlinesbi.sbi/portal"
        reg_feats_legit = extract_regional_features(legit_url)
        self.assertEqual(reg_feats_legit["brand_tld_misalignment"], 0.0)

    def test_explainer_output(self):
        explainer = PhishingExplainer()
        phish_url = "http://sbi-kyc-update-login.top/account/verify.php"
        feats = extract_features_dict(phish_url, include_regional=True)
        reasons = explainer.explain(phish_url, feats, risk_score=0.98, is_phishing=True)
        self.assertGreaterEqual(len(reasons), 1)
        self.assertTrue(any("Impersonation" in r or "SBI" in r or "Lure" in r for r in reasons))

    def test_experiment_results_exist(self):
        # Verify the consolidated benchmark summary exists and is readable
        summary_path = "results/SUMMARY.md"
        self.assertTrue(os.path.exists(summary_path), f"Missing benchmark summary: {summary_path}")
        with open(summary_path, "r", encoding="utf-8") as f:
            content = f.read()
        self.assertIn("E03", content, "Summary should contain E03 ablation results")
        self.assertIn("F1", content, "Summary should contain F1 metrics")
        self.assertIn("99", content, "Summary should contain high-accuracy results")

    def test_api_explain_endpoint(self):
        client = TestClient(app)
        res = client.post("/explain", json={"url": "http://sbi-kyc-update-login.top/verify"})
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertIn("reasons", data)
        self.assertIn("risk_score", data)
        self.assertEqual(data["verdict"], "phishing")


if __name__ == "__main__":
    unittest.main()
