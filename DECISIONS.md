# Architectural and Research Decision Log

This document records key technical, design, and experimental decisions made during the development of the Region-Aware Phishing URL Detection project.

---

## 1. Zero-Network Safe Ingestion Protocol
* **Decision**: All URL inspection and feature extraction is performed purely on static URL strings. No socket connections, HTTP requests, or live crawling of phishing candidates will be performed.
* **Rationale**: Eliminates malware delivery risk, prevents alerting malicious infrastructure, avoids dependence on transient phishing domains that have already been taken down, and ensures sub-millisecond feature extraction.

## 2. Leakage-Free Dataset Splitting (Domain Grouping)
* **Decision**: Train/validation/test splits are strictly partitioned by registered domain (`tldextract.registered_domain`). No registered domain present in training is ever permitted in the test or validation splits.
* **Rationale**: Random train/test splits suffer from domain overrepresentation leakage (e.g., hundreds of path variations on the same phishing host appearing in both train and test), artificially inflating benchmark accuracy. Group-splitting ensures true out-of-distribution evaluation.

## 3. Exclusion of Live Network / WHOIS Features from Core Inference
* **Decision**: WHOIS, DNS, and TLS queries are omitted from the real-time detection pipeline.
* **Rationale**: Live WHOIS/DNS queries introduce 500ms–2500ms latencies, encounter strict registrar rate limits, and frequently fail due to WHOIS privacy shielding. Omitting them keeps model inference under 5 milliseconds, enabling direct real-time deployment in Chrome extensions and high-throughput APIs.

## 4. Primary Tabular Classifier Selection
* **Decision**: LightGBM is chosen as the primary gradient boosted tree algorithm alongside Random Forest and Logistic Regression baselines, with XGBoost support maintained.
* **Rationale**: LightGBM provides optimal tree-splitting efficiency, lower memory footprint, and faster training loops across 5-fold seed evaluations while maintaining competitive ROC-AUC.

## 5. Label Normalization Across Heterogeneous Benchmarks
* **Decision**: All datasets are normalized to binary classification where `1 = Phishing (Malicious)` and `0 = Legitimate (Benign)`.
* **Rationale**: Benchmarks such as PhiUSIIL invert this convention ($1 = \text{legit}, 0 = \text{phishing}$). Standardizing early prevents silent evaluation errors.

## 6. Threshold Tuning on Validation Only
* **Decision**: Classification decision thresholds are tuned exclusively on validation splits using F1-optimization, never on test splits.
* **Rationale**: Prevents optimistic bias on test performance.
