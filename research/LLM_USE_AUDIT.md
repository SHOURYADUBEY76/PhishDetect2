# Comprehensive LLM-Use and Synthetic Data Audit

**Date:** October 1, 2026  
**Repository:** Region-aware Phishing Detection (`shourya2004raj-hub/Region-aware-Phishing-Detection`)  
**Scope:** Full codebase audit covering datasets, feature pipelines, inference engines, adversarial generators, and manuscript drafting.  
**Auditor:** Automated Rigor & Integrity Audit Suite (Independent Pipeline Verification)

---

## 1. Executive Summary

This audit was conducted to address mentor and peer-reviewer scrutiny regarding whether any research claims, benchmark results, or detection capabilities are artifacts of Large Language Models (LLMs), synthetic prompt generation, or automated LLM evaluators.

**Core Audit Verdict:**
- **Track A Confirmatory Findings (A1–A5): 100% Free of LLM and Synthetic Artifacts.** All primary evaluations use real-world web URLs crawled from production threat feeds (OpenPhish, PhishTank, PhreshPhish), public web rankings (Tranco Top 1M), and peer-reviewed external benchmarks (PhiUSIIL).
- **Feature Extraction: Deterministic & Rule-Based.** All 28 features in `src/features/feature_extractor.py` are computed using deterministic string parsing, regular expressions, and mathematical formulas (Shannon entropy, Levenshtein distance). Zero LLM embeddings or API calls exist.
- **Model Inference: Classical Machine Learning.** Primary classifiers are gradient-boosted trees (LightGBM) and deep character-level neural networks (BiLSTM). No LLM sits in the detection loop or serves as an ensemble judge.
- **Exploratory Synthetic Data (E08): Isolated & Non-Confirmatory.** Synthetic lures exist only in exploratory script `experiments/E08_ai_generated.py` and `data/raw/ai_phishing_standin.csv`. They were generated via deterministic regex templates (not live LLM calls) and are completely excluded from Track A confirmatory results and manuscript claims.
- **AI Coding Assistance:** Disclosed transparently under ACM/IEEE guidelines. AI assistance was utilized for test scaffolding, benchmark audits, and manuscript formatting, while all empirical evaluations and metric calculations were executed and verified against raw data on-disk.

---

## 2. Granular Stage-by-Stage Audit

| Pipeline Stage | Component / File | LLM Used? | Synthetic Data Used? | Evidence & Provenance |
| :--- | :--- | :---: | :---: | :--- |
| **Confirmatory Datasets** | `data/benchmark/a1_path_matched.csv` (N=1,460) | **NO** | **NO** | Real URLs from PhreshPhish and Tranco. Matched by path depth and character length. |
| **Confirmatory Datasets** | `data/benchmark/a2_syntactic_perturbations.csv` (N=250) | **NO** | **NO** | 50 real phishing URLs subjected to 4 deterministic syntactic transformations (homoglyphs, hyphens, subdomains, TLD swaps). |
| **Confirmatory Datasets** | `data/benchmark/a3_unseen_brands.csv` (N=658) | **NO** | **NO** | Real crawled URLs targeting verified Indian public/private entities not in training set. |
| **Confirmatory Datasets** | `data/benchmark/a4_cross_distribution.csv` (N=10,000) | **NO** | **NO** | Real URLs sampled from the external PhiUSIIL benchmark (Prasad & Chandra 2023). |
| **Confirmatory Datasets** | `data/benchmark/a5_temporal_test.csv` (N=9,339) | **NO** | **NO** | Real-world chronological holdout URLs from late-period threat feeds. |
| **Exploratory Datasets** | `data/raw/ai_phishing_standin.csv` (N=250) | **NO** (Stand-in) | **YES** | Handcrafted Python format-string templates (`NOVEL_UNSEEN_TEMPLATES` in `scripts/generate_ai_urls.py`). |
| **Feature Extraction** | `src/features/feature_extractor.py` (28 features) | **NO** | **NO** | Pure Python regex, `urllib.parse`, character counting, Shannon entropy. Zero LLM API calls. |
| **Classification Models** | `src/models/baseline.py` (LightGBM) | **NO** | **NO** | `lightgbm.LGBMClassifier` trained on numerical feature matrices. |
| **Deep Learning Models** | `src/models/bilstm.py` (BiLSTM + Lion) | **NO** | **NO** | Character-level embedding (vocab 128) + Bidirectional LSTM + Dense layers in Keras/TensorFlow. |
| **Serving Backend** | `backend/main.py` | **NO** | **NO** | FastAPI serving LightGBM pickle. No external API queries or prompt handlers. |
| **Explainability** | `src/explainer/explainer.py` | **NO** | **NO** | Rule-based string templates matching top LightGBM tree feature contributions. |
| **Research Writing** | `docs/PAPER_MANUSCRIPT.md`, `research/` reports | **Partial** | **NO** | Drafted with AI pair-programming assistance; grounded in verified code execution and data outputs. |

---

## 3. Detailed Investigation of LLM-Related Code

### 3.1 `scripts/generate_ai_urls.py` and `experiments/E08_ai_generated.py`
- **Purpose in Codebase:** Originally authored during exploratory capstone prototyping to explore how models respond to AI-themed or emerging lures.
- **Code Inspection:**
  - Lines 80–104 of `scripts/generate_ai_urls.py` include code that checks for `GEMINI_API_KEY` or `OPENAI_API_KEY` to query `gemini-1.5-flash`.
  - When no key is set (the standard offline execution environment), lines 105–129 generate deterministic template strings using `NOVEL_UNSEEN_TEMPLATES` (e.g., `http://pmsuryaghar-gov-subsidy-claim-{idx}.top/...`).
  - Output files are explicitly stamped with metadata: `provenance="synthetic_template_standin"`, `is_standin=True`.
- **Status in Manuscript & Track A:**
  - Experiment E08 is **strictly exploratory**.
  - It is **not** included in the confirmatory protocol (`research/EXPERIMENT_PROTOCOL.md` and `research/EXPERIMENT_PROTOCOL_AMENDMENT_1.md`).
  - It is **not** cited in `docs/PAPER_MANUSCRIPT.md` as supporting evidence for any core hypothesis (H1–H5).

### 3.2 Substring Keyword Hits in Production Datasets
During regex auditing across `data/benchmark/`, strings matching `gpt`, `openai`, `claude`, or `llm` were identified:
- `data/benchmark/a1_path_matched.csv`: `https://chatgpt-login-auth.xyz/...`, `https://openai.com/...`
- `data/benchmark/a4_cross_distribution.csv`: `http://openai.com.free-tokens.top/...`
- **Audit Verification:** These are real-world phishing lures targeting popular generative AI brands or the legitimate domain `openai.com` in Tranco. They are not generated by an LLM; they are targets of phishers captured in open feeds.

---

## 4. Addressing Mentor and Reviewer Attack Vectors

### Reviewer Challenge 1: *"Are the reported high AUCs or performance degradation figures synthetic artifacts of an LLM evaluator?"*
**Definitive Response:** No. Evaluation in this study uses zero LLM evaluators. Every metric (AUC-ROC, Brier score, ECE, Precision, Recall, F1) is calculated using Scikit-Learn (`sklearn.metrics.roc_auc_score`, `brier_score_loss`) against verified ground-truth labels from threat feeds (PhishTank/OpenPhish) and authoritative legitimate lists (Tranco Top 1M).

### Reviewer Challenge 2: *"Was the regional model trained on synthetic LLM-augmented data to give it an unfair advantage?"*
**Definitive Response:** No. The regional model and generic baseline are trained on the exact same real-world training splits (`data/processed/global_train.csv` and `data/processed/indian_train.csv`). The regional model differs solely by the addition of 4 regional host/TLD features (`is_indian_tld`, `has_indian_keyword`, `brand_indian_bank`, `brand_indian_gov`). No data augmentation or synthetic expansion was applied.

### Reviewer Challenge 3: *"Did an LLM generate the adversarial perturbations in A2/E06?"*
**Definitive Response:** No. The adversarial perturbations in A2 (`data/benchmark/a2_syntactic_perturbations.csv`) are strictly programmatic character-level mutations:
1. Homoglyphs: Cyrillic/Greek lookalikes (`а` for `a`, `о` for `o`).
2. Hyphenation: Insertion of hyphens into core domain tokens.
3. Subdomain padding: Prepending target brand names as subdomains.
4. TLD swap: Swapping ccTLDs (`.in` -> `.top`, `.com` -> `.xyz`).
Every perturbation is deterministic and reproducible via `src/features/adversarial.py`.

---

## 5. Formal Statement for Peer Review and Ethics Compliance

> **Declaration of LLM Usage and Data Integrity:**  
> The authors certify that all empirical benchmarks, model training procedures, feature extractions, and statistical evaluations reported in this study were conducted on authentic, real-world web data and standard open benchmarks (OpenPhish, PhishTank, PhreshPhish, Tranco, PhiUSIIL). No Large Language Model was employed to generate training data, label test samples, extract representations, or evaluate predictions for the confirmatory experiments (A1–A5). Generative AI assistance was strictly limited to code development, repository verification tooling, and editorial assistance in preparing manuscript drafts, adhering to ACM and IEEE guidelines on Generative AI in scholarly publishing.
