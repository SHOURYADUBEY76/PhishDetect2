"""Phase 1 Rigorous Engineering Validation Tests.
Validates HTTP 422 input rejection, canonicalization, true lexical masking,
brand matching boundary guarantees, LexiconConfig isolation, and model integrity.
"""

import os
import sys
import unittest

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fastapi.testclient import TestClient

from backend.main import app, compute_file_sha256, MODEL_REGION_AWARE, MODEL_BASELINE
from src.utils.domain import (
    validate_url_syntax,
    canonicalize_url,
    extract_registered_domain,
    has_idn_punycode
)
from src.features.lexical import extract_lexical_features
from src.features.structural import extract_structural_features
from src.features.regional import (
    extract_regional_features,
    LexiconConfig,
    DEFAULT_LEXICON_CONFIG
)


class TestPhase1Rigor(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    def test_validation_http_422_on_malformed_inputs(self):
        """Phase 1.1: Verify HTTP 422 on invalid schemes, ports, and malformed syntax."""
        malformed_cases = [
            "",
            "   ",
            "javascript:alert(1)",
            "data:text/html;base64,PHNjcmlwdD5hbGVydCgxKTwvc2NyaXB0Pg==",
            "ftp://example.com/file.txt",
            "http://foo.bar:999999999/test",
            "http://example.com:abc/path",
            "http://[invalid-ipv6/test",
            "http://[2001:db8::1:extra]/test",
        ]
        for bad_url in malformed_cases:
            res = self.client.post("/predict", json={"url": bad_url})
            self.assertEqual(
                res.status_code, 422,
                f"Expected HTTP 422 for malformed input {repr(bad_url)}, got {res.status_code}"
            )
            data = res.json()
            self.assertIn("detail", data)

    def test_canonicalization_and_scheme_assumption(self):
        """Phase 1.2: Verify canonicalization, trailing slash, and scheme_assumed flag."""
        # Bare input without scheme
        res1 = self.client.post("/predict", json={"url": "instagram.com"})
        self.assertEqual(res1.status_code, 200)
        d1 = res1.json()
        self.assertTrue(d1["scheme_assumed"])
        self.assertEqual(d1["canonical_url"], "https://instagram.com/")
        self.assertEqual(d1["registered_domain"], "instagram.com")

        # Explicit scheme with trailing slash
        res2 = self.client.post("/predict", json={"url": "https://www.google.com"})
        self.assertEqual(res2.status_code, 200)
        d2 = res2.json()
        self.assertFalse(d2["scheme_assumed"])
        self.assertEqual(d2["canonical_url"], "https://www.google.com/")
        self.assertEqual(d2["registered_domain"], "google.com")

    def test_true_lexical_masking(self):
        """Phase 1.3: Verify true lexical masking eliminates path and query leakage."""
        test_url = "https://example.com/very/long/suspicious/path/with/digits12345/and-symbols?token=abcdef"
        feats_full = extract_lexical_features(test_url, include_path_query=True)
        feats_masked = extract_lexical_features(test_url, include_path_query=False)

        # Masked URL should strictly evaluate scheme + authority ('https://example.com')
        expected_len = len("https://example.com")
        self.assertEqual(feats_masked["url_length"], expected_len)
        self.assertEqual(feats_masked["path_length"], 0.0)
        self.assertEqual(feats_masked["query_length"], 0.0)
        self.assertEqual(feats_masked["num_digits"], 0.0)  # The digits '12345' were in path
        self.assertEqual(feats_masked["num_slashes"], 2.0)  # Only slashes in 'https://'
        self.assertNotEqual(feats_full["url_length"], feats_masked["url_length"])
        self.assertNotEqual(feats_full["url_entropy"], feats_masked["url_entropy"])

    def test_brand_matching_boundary_guarantees(self):
        """Phase 1.4: Verify no substring collisions on English words."""
        # Non-target English words containing brand substrings ('pan' in 'company', 'axis' in 'coaxial')
        benign_false_targets = [
            "https://company.com/about",
            "https://coaxial-cables.net/store",
            "https://bobsleigh-racing.org/championship",
            "https://japan-travel.com/guide",
        ]
        for u in benign_false_targets:
            reg_feats = extract_regional_features(u)
            self.assertEqual(
                reg_feats["has_indian_brand_token"], 0.0,
                f"False positive brand match on substring in {u}"
            )
            self.assertEqual(reg_feats["brand_tld_misalignment"], 0.0)

        # True target should match
        true_target = "https://sbi-netbanking-portal.xyz/login"
        reg_true = extract_regional_features(true_target)
        self.assertEqual(reg_true["has_indian_brand_token"], 1.0)
        self.assertEqual(reg_true["brand_tld_misalignment"], 1.0)

    def test_idn_homoglyph_detection(self):
        """Phase 1.4: Verify Punycode and IDN homoglyphs are detected."""
        self.assertTrue(has_idn_punycode("https://xn--e1afmkfd.xn--p1ai/"))
        self.assertTrue(has_idn_punycode("https://gооgle.com"))  # Cyrillic 'о'
        self.assertFalse(has_idn_punycode("https://google.com"))

    def test_lexicon_config_isolation(self):
        """Phase 1.5: Verify zero-knowledge evaluation using custom LexiconConfig."""
        # Withhold 'sbi' and 'hdfc' from custom config
        custom_cfg = LexiconConfig(
            brands=["paytm", "phonepe"],
            alias_map={},
            official_domains={"paytm": {"paytm.com"}, "phonepe": {"phonepe.com"}},
            tlds={"in", "co.in"},
            lures=["kyc"]
        )

        test_url = "https://sbi-kyc-verify.xyz/auth"
        # Default config detects sbi
        feats_default = extract_regional_features(test_url, config=DEFAULT_LEXICON_CONFIG)
        self.assertEqual(feats_default["has_indian_brand_token"], 1.0)

        # Custom isolated config has zero knowledge of sbi
        feats_isolated = extract_regional_features(test_url, config=custom_cfg)
        self.assertEqual(feats_isolated["has_indian_brand_token"], 0.0)
        self.assertEqual(feats_isolated["brand_token_count"], 0.0)
        self.assertEqual(feats_isolated["brand_tld_misalignment"], 0.0)

    def test_health_probe_and_model_integrity(self):
        """Phase 1.6: Verify health endpoint reports valid cryptographic model hash."""
        res = self.client.get("/health")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertTrue(data["model_loaded"])
        self.assertIsNotNone(data["model_sha256"])
        self.assertEqual(len(data["model_sha256"]), 64)  # 64-char hex string


if __name__ == "__main__":
    unittest.main()
