# Scientific Experiment Protocol: Track A Pre-Registration (`prereg-v2`)

**Protocol Version**: `prereg-v2`  
**Git Branch**: `research/rigor-v2`  
**Committed Direction**: Track A (Shortcut Audit, Representation Sensitivity & Distribution Shift Benchmark)  
**Scope**: Experiments A1–A5 (Experiment E06 retained strictly as an Appendix)

---

## 1. Research Overview & Objectives

Static URL-only phishing detectors have historically reported exceptional performance metrics ($F1 > 0.98$, $\text{AUC} > 0.99$) on standard benchmarks. Recent empirical audits, however, indicate that these high scores are substantially driven by **dataset-level lexical and structural shortcuts** (e.g. legitimate URLs being 100% bare hostnames without paths, while phishing URLs almost universally possess deep paths and parameters).

The objective of this research package is to conduct a scientifically defensible, reproducible audit quantifying:
1. **Shortcut Exploitation**: The extent to which detection performance relies on trivial structural artifacts rather than semantic deception.
2. **Representation Sensitivity**: How minor representation shifts (presence of `www.`, scheme assumptions, and trailing slashes) induce catastrophic false-positive cascades on major benign homepages.
3. **Generalization Failures**: Model degradation under zero-knowledge held-out brand shifts, cross-dataset transfer, and temporal distribution drift.

---

## 2. Experimental Regimes (A1 – A5)

### Experiment A1: Shortcut Sensitivity & True Lexical Masking
- **Hypothesis $H_1$**: Standard static classifiers rely heavily on path/query presence and length shortcuts. When evaluated on stratum-matched data (balancing path presence, path depth, length quintiles, and protocol) or when true lexical masking is applied (`include_path_query=False`), detection performance drops significantly.
- **Evaluation Set**: `data/benchmark/a1_path_matched.csv` (21,094 rows; 1:1 balance).
- **Models Evaluated**:
  - Baseline Shortcut-Only Logistic Regression / LightGBM (trained only on path presence, depth, length, HTTPS, subdomain presence).
  - Generic Lexical + Structural LightGBM (Unmasked).
  - Generic Lexical + Structural LightGBM with True Lexical Masking.
- **Statistical Test**: Paired cluster-bootstrap over registered domain clusters ($B=1,000$, $\alpha=0.05$).

### Experiment A2: Representation Sensitivity Diagnostic
- **Hypothesis $H_2$**: Static feature representations are hypersensitive to syntactic formatting choices. Specifically, trailing slash inclusion (`/`), scheme assumptions (`https` vs `http`), and subdomain presence (`www.` vs bare) induce non-uniform, bidirectional risk score shifts across popular benign websites.
- **Evaluation Set**: `data/benchmark/a2_representation.csv` (300 URLs across 50 top global and regional domains).
- **Diagnostics**:
  - Probability shift $\Delta p = p(\text{variant}) - p(\text{base})$.
  - Single-feature flip attributions across `subdomain_length`, `path_length`, `is_https`, and `num_slashes`.

### Experiment A3: Lexicon-Clean Unseen Brand Generalization
- **Hypothesis $H_3$**: Hand-crafted regional feature extractors fail to generalize to novel target brands when evaluated under true zero-knowledge conditions (where held-out brands are strictly excluded from the feature extractor's `LexiconConfig`).
- **Evaluation Set**: `data/benchmark/a3_unseen_brands.csv` (1,660 rows).
- **Protocol**: Feature extraction is executed using a custom `LexiconConfig` purged of all 10 held-out brands (`phonepe`, `aadhaar`, `uidai`, `irctc`, `axis`, `pnb`, `bob`, `kotak`, `epfo`, `paytm`).
- **Comparison**: Zero-knowledge regional model vs. generic structural/lexical baseline.

### Experiment A4: Cross-Distribution & Hosting-Region Analysis
- **Hypothesis $H_4$**: Models trained on one distribution (e.g. PhreshPhish) suffer significant performance drops when transferred to external distributions (e.g. PhiUSIIL).
- **Pre-Registered Descriptive Hosting-Region Analysis** [DESCRIPTIVE ONLY]:
  - Measure performance of generic detectors on `.in` and `.bharat`-hosted URLs versus generic TLDs.
  - Reported with domain-clustered 95% confidence intervals.
  - Strictly labeled as **hosting infrastructure** (not targeting), with no model-comparison or superiority claims.
- **Evaluation Set**: `data/benchmark/a4_cross_distribution.csv` (20,000 rows).

### Experiment A5: Temporal Generalization
- **Hypothesis $H_5$**: Static detectors degrade over time due to evolving phishing campaigns, domain churning, and hosting shifts.
- **Protocol**: Train strictly on historical data ($T < \text{2025-01-01}$); evaluate on strictly future holdout ($T \ge \text{2025-06-01}$).
- **Leakage Prevention**: Registered domains in the training partition are strictly purged from the test partition ($D_{\text{train}} \cap D_{\text{test}} = \emptyset$).
- **Evaluation Sets**: `data/benchmark/a5_temporal_train.csv` (25,000 rows), `data/benchmark/a5_temporal_test.csv` (9,339 rows).

### Appendix: Experiment E06 (Adversarial Perturbation)
- Retained strictly as an informative diagnostic appendix evaluating character-level perturbations (homoglyphs, hyphen insertion, subdomain padding).

---

## 3. Pre-Registered Feasibility Criteria & Threshold Sensitivity

The feasibility criteria for attempting regional model training (Track A+R) versus committing to Track A were established and frozen at Gate G1:

| Criterion | Metric | Pre-Registered Threshold | Observed on PhreshPhish | Status |
| :--- | :--- | :--- | :--- | :--- |
| **C1: Sample Size** | Independent Indian Phishing URLs | $N \ge 500$, $D \ge 100$ | $N = 89$, $D = 78$ | **FAILED** |
| | Path-Bearing Benign Indian URLs | $N \ge 500$, $D \ge 100$ | $N = 430$, $D = 230$ | **FAILED** |
| **C2: Temporal Span** | Calendar Months Covered | $\ge 3$ months per class | Phish: 12 mo, Benign: 10 mo | **PASSED** |
| **C3: Matching Survival** | Stratum-Matched Pairs | $\ge 250$ pairs, $D \ge 50$ | 50 pairs, $D_0=38, D_1=48$ | **FAILED** |

### Threshold Sensitivity Analysis
- **Halved Thresholds** ($N \ge 250$, Matched Pairs $\ge 125$): Decision remains **Track A ONLY** (89 phishing instances and 50 matched pairs fail both conditions).
- **Doubled Thresholds** ($N \ge 1000$, Matched Pairs $\ge 500$): Decision remains **Track A ONLY**.
- *Conclusion*: The decision to pursue Track A only is robust to any reasonable threshold choice.

---

## 4. Frozen Benchmark Hashes (SHA-256)

All benchmark partitions are cryptographically frozen in `research/split_manifest.json`:

| File | Rows | SHA-256 Hash |
| :--- | :--- | :--- |
| `data/benchmark/a1_path_matched.csv` | 21,094 | `ae74e098bb87c06bf16aa4702bbe01645a13d0cc36bde9d9a0291cd9d319b670` |
| `data/benchmark/a2_representation.csv` | 300 | `57cb32cf152988376d454c5613ed07309bcb4412829d4aac560fcd325a98e5dc` |
| `data/benchmark/a3_unseen_brands.csv` | 1,660 | `08e409c9979fbbad095b9fb85dd8397f06e4046cc87d125fd514c0a32093a636` |
| `data/benchmark/a4_cross_distribution.csv` | 20,000 | `0b763a117a89b4635379b905e2de1d68ca0349e83ba8550905b272c3e213890d` |
| `data/benchmark/a5_temporal_train.csv` | 25,000 | `e9f50f9112c3f118aba4c61663c9555f0f6c31684c9f9c5536f546908d916a75` |
| `data/benchmark/a5_temporal_test.csv` | 9,339 | `28a98158068a39d955886bf0c6534f1e4ab7ddc2934f501039cbb55e0a03b747` |

---

## 5. Statistical Testing Protocol

- **Domain-Clustered Resampling**: All confidence intervals and hypothesis tests are clustered at the **registered domain level** using `research/stats.py` (`paired_cluster_bootstrap`).
- **Resamples**: $B = 1,000$ bootstrap iterations.
- **Confidence Level**: 95% ($2.5\%$ to $97.5\%$ percentiles).
- **Hypothesis Decision Rule**: A performance difference $\Delta = M_A - M_B$ is statistically significant at $\alpha = 0.05$ if and only if the 95% bootstrap confidence interval strictly excludes $0.0$ and $p < 0.05$.
- **Random Seeds**: Fixed across all multi-seed runs: `[42, 43, 44, 45, 46]`.
