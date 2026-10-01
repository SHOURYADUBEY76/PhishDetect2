# AI Assistance and Generative Tooling Disclosure

**Project:** Region-Aware Phishing Detection  
**Adherence Standard:** ACM Publications Policy on Generative AI / IEEE Authorship Guidelines on AI-Generated Text  
**Date:** October 1, 2026  

---

## 1. Scope of AI Assistance

In compliance with scholarly publishing norms regarding transparency in computational research, the authors disclose the following roles of Artificial Intelligence and Large Language Models (LLMs) in this project:

### Permitted Scientific Engineering Use Cases
1. **Code Scaffolding and Quality Assurance:**  
   AI pair-programming assistants (Google Gemini / Antigravity) were utilized for writing unit test suites (`tests/test_benchmark_integrity.py`, `tests/test_model.py`), calculating bootstrap confidence intervals (`research/stats.py`), and verifying domain-level cluster independence.
2. **Repository Consistency and Audit Verification:**  
   LLM tools assisted in cross-referencing row counts, checking domain overlaps across training and test splits, and auditing manuscript claims against serialized result files (`results/*.json`).
3. **Editorial and Formatting Assistance:**  
   AI assistance was utilized for structuring LaTeX tables, standardizing markdown presentation in `docs/PAPER_MANUSCRIPT.md`, and organizing experiment logs.

---

## 2. Boundaries and Prohibitions (Zero Synthetic Research Artifacts)

The authors explicitly confirm the following boundaries:
- **No LLM Generated Scientific Data:** Zero synthetic or LLM-generated URLs were used in the confirmatory Track A benchmarks (A1–A5). All 22,707 evaluated URLs originate from authentic threat feeds (PhreshPhish, OpenPhish, PhishTank), legitimate domain rankings (Tranco Top 1M), or published benchmarks (PhiUSIIL).
- **No LLM Ground-Truth Labeling:** All ground-truth benign and phishing designations were assigned by authoritative external crawlers and verified security feeds, never by automated LLM prompting.
- **No LLM in Production Inference:** Neither the LightGBM classifier nor the BiLSTM neural network relies on external LLM APIs, embeddings, or prompts. The model runs fully local and offline with deterministic latency.
- **No LLM in Metric Evaluation:** All statistical claims, effect sizes, p-values, AUC-ROC scores, and calibration metrics were computed using standard Python scientific libraries (`scipy`, `scikit-learn`, `numpy`).

---

## 3. Contact and Reproducibility

Full reproduction scripts and raw benchmark files are permanently versioned in the Git repository. Reviewers may independently verify the authenticity of all dataset rows and execution traces by running:
```bash
python research/reproduce.py --all
```
