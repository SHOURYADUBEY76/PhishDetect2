# Confirmatory Evaluation Report: Track A Confirmatory Findings and Methodological Corrections

**Date**: 2026-10-01  
**Project**: Region-Aware Phishing Detection  
**Branch**: `research/rigor-v2`  
**Protocol Freeze Baseline**: `prereg-v2` (Git commit `b4a5d2a758223f1d7895b722ef6650087ef33a27`)  
**Protocol Amendments**: Adopted in `research/EXPERIMENT_PROTOCOL_AMENDMENT_1.md` (Git commit `204692d`)  
**Status**: Confirmatory Run Complete; Bounded Post-Run Reporting Correction Incorporated  

---

## Executive Summary & Provenance Verification

This report documents the finalized confirmatory experimental results for **Track A (A1–A5)** and the **E06 Appendix**, incorporating all post-run reporting corrections. The experiments evaluate lexical shortcut reliance, syntactic representation fragility, unseen brand generalization under zero-knowledge constraints, out-of-distribution transfer to external corpora, temporal decay across chronological feeds, and adversarial perturbation dynamics.

All confirmatory runs were executed strictly on frozen, non-synthetic benchmarks and verified data splits. Model artifacts, prediction matrices, and paired cluster bootstrap distributions are recorded in `research/results/` and verified against the protocol specifications. The Git tag `prereg-v2` remains completely preserved at commit `b4a5d2a`.

---

## 1. Metric Definitions & Mathematical Sign Conventions

To prevent ambiguity, every comparative performance metric ($\Delta\text{AUC}$) is explicitly defined with its exact mathematical sign convention, minuend, and subtrahend:

| Experiment | Metric Symbol | Formal Mathematical Definition | Minuend ($M_1$) | Subtrahend ($M_2$) | Sign Interpretation |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **A1** | $\Delta\text{AUC}_{A1}$ | $\text{AUC}_{\text{unmasked}} - \text{AUC}_{\text{masked}}$ | Full Lexical Model (Unmasked) | Masked Lexical Model (Host-Only) | **Positive ($>0$)**: Full URL features provide higher discrimination than host-only features. |
| **A3** | $\Delta\text{AUC}_{A3}$ | $\text{AUC}_{\text{regional,zk}} - \text{AUC}_{\text{generic}}$ | Zero-Knowledge Regional Model | Generic Baseline Model | **Negative ($<0$)**: Regional features degrade performance under zero-knowledge brand testing. |
| **A4 (Primary)** | $\Delta\text{AUC}_{A4,\text{pri}}$ | $\text{AUC}_{\text{in-dist}} - \text{AUC}_{\text{cross,pri}}$ | In-Distribution (`global_test`) | Primary External (`phiusiil_test`) | **Positive ($>0$)**: Out-of-distribution distribution shift degrades ranking discrimination. |
| **A4 (Disjoint)** | $\Delta\text{AUC}_{A4,\text{dis}}$ | $\text{AUC}_{\text{in-dist}} - \text{AUC}_{\text{cross,dis}}$ | In-Distribution (`global_test`) | Secondary Disjoint (`phiusiil_disjoint`) | **Positive ($>0$)**: Strict domain-disjoint shift degrades ranking discrimination. |
| **A5** | $\Delta\text{AUC}_{A5}$ | $\text{AUC}_{\text{in-time,val}} - \text{AUC}_{\text{out-of-time,test}}$ | In-Time Validation Split | Out-of-Time Future Test Split | **Positive ($>0$)**: Chronological feed aging degrades classification performance over time. |

---

## 2. Experiment A1: Shortcut Sensitivity & Lexical Masking

### A. Experimental Design & Hypotheses
* **Hypothesis $H_{1,0}$**: Lexical features derived from the URL path and query components do not provide incremental discriminative power over registered-domain and host-level features ($\Delta\text{AUC}_{A1} = 0$).
* **Hypothesis $H_{1,1}$**: Lexical path/query features provide statistically significant discriminative power ($\Delta\text{AUC}_{A1} \neq 0$).
* **Benchmark**: `data/benchmark/a1_path_matched.csv` ($N = 21,094$; 10,547 benign, 10,547 phishing across 10,870 unique registered domains). Evaluates unmasked, path-masked, and shortcut-only models.

### B. Empirical Results
* **Unmasked Generic LightGBM**: $\text{AUC} = 0.9598$, $\text{F1} = 0.8952$, $\text{Precision} = 0.9102$, $\text{Recall} = 0.8807$, $\text{FPR} = 8.69\%$ ($\tau = 0.5136$).
* **Masked Generic LightGBM** (Host-Only): $\text{AUC} = 0.9317$, $\text{F1} = 0.8193$, $\text{Precision} = 0.9323$, $\text{Recall} = 0.7307$, $\text{FPR} = 5.31\%$ ($\tau = 0.6500$).
* **Shortcut-Only LightGBM** (6 Structural Features: `path_depth`, `path_length`, `url_length`, `is_https`, `subdomain_length`, `num_slashes`): $\text{AUC} = 0.9088$, $\text{F1} = 0.8270$, $\text{Precision} = 0.8381$, $\text{Recall} = 0.8163$, $\text{FPR} = 15.77\%$ ($\tau = 0.5500$).

### C. Paired Domain-Clustered Bootstrap Analysis (1,000 Replicates)
* **Observed $\Delta\text{AUC}_{A1}$**: $+0.0281$ ($+2.81\ \text{AUC points}$).
* **Bootstrap Mean $\Delta\text{AUC}$**: $+0.0287$ ($\text{std} = 0.0047$).
* **95% Bootstrap Percentile Confidence Interval**: $[+0.0207, +0.0386]$ (Interval width: $0.0179$).
* **Inferential Conclusion**: Null hypothesis $H_{1,0}$ is rejected ($p < 0.001$, valid replicates: 1,000/1,000). Removing path and query features incurs an observed $2.81$ AUC point degradation.
* **Shortcut Baseline Finding**: The shortcut-only model achieves $\text{AUC} = 0.9088$ using solely 6 coarse length/depth features without inspecting lexical tokens, demonstrating that structural shortcuts provide massive baseline predictive power in URL datasets.

---

## 3. Experiment A2: Syntactic Representation Sensitivity

### A. Experimental Scope & Boundary Conditions
Experiment A2 evaluates representation sensitivity specifically on a controlled test sample of **50 high-traffic canonical benign homepages** across 6 distinct URL string representations:
1. `http_bare`: `http://example.com`
2. `https_bare` (Canonical Reference): `https://example.com`
3. `https_trailing_slash`: `https://example.com/`
4. `https_www_bare`: `https://www.example.com`
5. `https_www_slash`: `https://www.example.com/`
6. `raw_bare`: `example.com`

*Scope Limitation*: These findings characterize model sensitivity on the specified 50 test domains and lexical feature extractor. They are not asserted as universal mathematical properties of all arbitrary web endpoints.

### B. Empirical Results
* **Mean Base Phishing Probability** (`https_bare`): $0.0221$ (range: $[0.0068, 0.0371]$).
* **Mean Probability Shifts Relative to `https_bare`**:
  * $\Delta_{\text{http\_bare}}$: $+0.9199$ (Mean prob: $0.9420$, max shift: $+0.9638$).
  * $\Delta_{\text{raw\_bare}}$: $+0.9351$ (Mean prob: $0.9573$, max shift: $+0.9654$).
  * $\Delta_{\text{https\_www\_bare}}$: **$+0.6159$** (Mean prob: $0.6380$, max shift: $+0.8038$).
  * $\Delta_{\text{https\_trailing\_slash}}$: $+0.3258$ (Mean prob: $0.3479$, max shift: $+0.6075$).
  * $\Delta_{\text{https\_www\_slash}}$: $+0.1255$ (Mean prob: $0.1477$, max shift: $+0.3608$).
* **Benign-to-Phishing Prediction Flips** (at in-distribution optimal threshold $\tau = 0.5136$):
  * **`https_www_bare`**: **39 out of 50 domains (78.0%)** flip from benign to phishing simply by prefixing `www.` (e.g. `adobe.com` shifts from $0.0193 \to 0.7059$; `amazon.com` shifts from $0.0191 \to 0.7144$; `craigslist.org` shifts from $0.0371 \to 0.7148$).
  * **`https_trailing_slash`**: 5 out of 50 domains (10.0%) flip to phishing.
  * **`http_bare`**: 50 out of 50 domains (100.0%) flip to phishing.
* **Conclusion**: Verifies extreme syntactic brittleness. The classifier relies heavily on superficial token length, subdomain parsing, and protocol flags rather than invariant domain ownership, making canonicalization mandatory prior to model inference.

---

## 4. Experiment A3: Unseen Brand Generalization & Scope Calibration

### A. Experimental Protocol & Zero-Knowledge Architecture
* **Benchmark**: `data/benchmark/a3_unseen_brands.csv` ($N = 658$; 329 phishing, 329 benign across 579 unique registered domains).
* **Ground-Truth Entity Verification**: 6 Indian brand targets (`sbi`, `hdfc`, `icici`, `phonepe`, `paytm`, `uidai`) verified through token-boundary matching and threat feed target metadata. Ambiguous polysemic matches (`bob`, `axis`, `pan`) were permanently purged.
* **Zero-Knowledge Feature Configuration**: Held-out target brands were strictly excluded from the training splits and completely expunged from the regional feature extractor's `LexiconConfig`.
* **Hypotheses**:
  * $H_{3,0}$: Under zero-knowledge brand testing, the regional model's AUC equals the generic model's AUC ($\Delta\text{AUC}_{A3} = \text{AUC}_{\text{regional,zk}} - \text{AUC}_{\text{generic}} = 0$).
  * $H_{3,1}$: Regional features alter zero-knowledge discrimination ($\Delta\text{AUC}_{A3} \neq 0$).

### B. Empirical Results
* **Generic Baseline LightGBM**: $\text{AUC} = 0.9766$, $\text{F1} = 0.8354$, $\text{Precision} = 1.0000$, $\text{Recall} = 0.7173$, $\text{FPR} = 0.00\%$ ($\tau = 0.5136$).
* **Zero-Knowledge Regional Model**: $\text{AUC} = 0.9480$, $\text{F1} = 0.8542$, $\text{Precision} = 0.9960$, $\text{Recall} = 0.7477$, $\text{FPR} = 0.30\%$ ($\tau = 0.4864$).
* **Paired Domain-Clustered Bootstrap Analysis (1,000 Replicates)**:
  * Observed $\Delta\text{AUC}_{A3}$: **$-0.0286$** ($-2.86\ \text{AUC points}$).
  * Bootstrap Mean $\Delta\text{AUC}$: $-0.0290$ ($\text{std} = 0.0058$).
  * 95% Bootstrap Percentile Confidence Interval: $[-0.0427, -0.0186]$ (Interval width: $0.0241$).
  * Inferential Conclusion: Null hypothesis $H_{3,0}$ is rejected ($p < 0.001$). Zero-knowledge regional features produce a statistically significant degradation of $2.86$ AUC points relative to the generic baseline.

### C. Analysis of Shared Hosting Overlaps & Calibration of Claims
* **Documentation of Shared Domain Overlaps**: Exact cross-referencing between `data/benchmark/a3_unseen_brands.csv` and the training partition (`global_train`) reveals **12 overlapping registered domains**:
  1. `blogspot.com` (free blogging/hosting platform)
  2. `boi-infinity.online` (generic campaign infrastructure)
  3. `hostingersite.com` (shared cloud web host)
  4. `in.net` (shared SLD registry)
  5. `netlify.app` (serverless cloud application host)
  6. `rbiorg-online.in` (campaign infrastructure)
  7. `resortsdella.com` (shared domain)
  8. `sterling-hotel.com` (shared domain)
  9. `vercel.app` (serverless cloud deployment platform)
  10. `weebly.com` (free website builder platform)
  11. `wordpress.com` (multi-tenant CMS hosting platform)
  12. `zendesk.com` (customer support SaaS platform)
* **Methodological Implications**:
  1. Seven of the 12 overlapping domains are major multi-tenant hosting platforms. Because multi-tenant platforms host both legitimate sites and phishing lures across independent subdomains, generic structural features on these hosts already provide baseline signal to both models.
  2. In zero-knowledge settings where the brand dictionary is purged, regional features register zeroes or uninformative noise for the targeted brands, diluting tree split choices.
* **Calibrated Conclusion**: This finding is strictly calibrated to the tested benchmark ($N = 658$), the specific lexical feature set, and the GBDT model architecture. It **does not claim universal absence of regional-feature value** across all possible architectures (e.g. computer vision, HTML DOM parsing, or graph embeddings) or all possible future regional attack vectors.

---

## 5. Experiment A4: Cross-Distribution Transfer & Hosting Census

### A. Primary External vs. Secondary Disjoint Benchmarks
* **Primary Cross-Distribution Benchmark (`phiusiil_test`)**: Full external PhiUSIIL dataset ($N = 3,520$; 1,760 benign, 1,760 phishing across 2,838 unique registered domains). Evaluates natural out-of-distribution transfer with naturally occurring multi-tenant hosting platforms.
* **Secondary Domain-Disjoint Diagnostic (`phiusiil_disjoint`)**: Strict disjoint subset ($N = 2,888$; 1,701 benign, 1,187 phishing across 2,746 unique registered domains). Purges all 92 overlapping registered domains that appear in any training partition.

### B. Empirical Results: Discrimination (AUC) vs. Severe Operating-Threshold Transfer Failure (FPR)

| Evaluation Partition | $N$ (URLs) | Registered Domains | ROC-AUC | $\Delta\text{AUC}$ vs In-Dist | $\tau$ (Threshold) | Recall (TPR) | Precision | False Positive Rate (FPR) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **In-Distribution (`global_test`)** | 21,094 | 10,870 | **0.9284** | — | 0.5136 | 79.52% | 91.06% | **8.58%** |
| **Primary PhiUSIIL (`phiusiil_test`)** | 3,520 | 2,838 | **0.8661** | **+0.0623** | 0.5136 (Fixed) | 92.33% | 51.51% | **86.93%** |
| **Secondary Disjoint (`phiusiil_disjoint`)** | 2,888 | 2,746 | **0.8303** | **+0.0982** | 0.5136 (Fixed) | 90.06% | 41.95% | **86.95%** |

### C. Distinction Between Ranking Discrimination and Operating-Threshold Transfer Failure
* **Ranking Discrimination**: The model retains moderate ranking ability on external data ($\text{AUC} = 0.8661$ primary, $0.8303$ disjoint). The degradation ($\Delta\text{AUC}_{\text{pri}} = +0.0623$; $\Delta\text{AUC}_{\text{dis}} = +0.0982$) exceeds the post-preregistration operational criterion of $\Delta\text{AUC} > 0.05$.
* **Operating-Threshold Transfer Failure**: Under the in-distribution fixed operating threshold ($\tau = 0.5136$), the false-positive rate explodes to **86.93%** ($1,530 / 1,760$ benign URLs misclassified as phishing).
* **Clinical / Deployment Implication**: This failure is a severe **probability miscalibration and operating-threshold transfer problem**. Although the model can rank malicious URLs higher than benign ones within the external corpus, the baseline scores of benign external URLs are significantly shifted toward the phishing class. Deploying an uncalibrated threshold across independent feeds causes operational collapse via false-positive flooding.

### D. Dedicated Hosting Infrastructure Census (`.in` / `.co.in`)
* **Census Population**: Decoupled census of $N = 1,682$ URLs from the PhreshPhish stream hosted under Indian ccTLD infrastructure (`.in` and `.co.in`; 1,205 phishing URLs across 496 domains, 477 benign URLs across 263 domains).
* **Model Discrimination**: $\text{AUC} = 0.9379$, $\text{F1} = 0.9008$, $\text{Precision} = 0.9511$, $\text{Recall} = 0.8556$, $\text{FPR} = 11.11\%$ ($\tau = 0.5136$).
  * 95% Clustered Bootstrap CI on AUC: $[0.8978, 0.9630]$ ($\text{mean} = 0.9349, \text{std} = 0.0171, B_{\text{valid}} = 1,000$).
* **Candidate Feed-Target Breakdown**:
  * **Identified Foreign / Global Candidate Targets**: **713 URLs (59.17% of phishing; 42.39% of census)** across 118 candidate feed targets (`google`, `facebook`, `whatsapp`, `canva`, `roblox`, `netflix`, etc.).
  * **Ambiguous / Unspecified Candidate Targets**: **412 URLs (34.19% of phishing; 24.49% of census)** across generic feed labels (`other`: 370, `genericwebmailphishing`: 30, `webmail`: 12).
  * **Domestic Indian Candidate Targets**: **80 URLs (6.64% of phishing; 4.76% of census)** across 19 domestic entities (`governmentofindia`: 29, `della adventure`: 8, `sterling holidays`: 7, `meesho`: 6, `npci`: 6, `flipkart`: 5, etc.).
  * **Benign Reference**: **477 URLs (28.36% of census)**.
* **Methodological Clarification**: All entity classifications represent **candidate feed-target labels** assigned by upstream feeds, not independently verified physical victim identities. ccTLD registration reflects DNS delegation under the `.in` registry and does not infer physical server location, cloud infrastructure domicile, attacker intent, or victim physical location.

---

## 6. Experiment A5: Temporal Generalization

### A. Experimental Design
* **Splits**: Strict chronological splitting based on ingestion timestamps.
  * In-Time Training Partition (`data/benchmark/a5_temporal_train.csv`): $N = 25,000$ URLs (evaluated with domain-grouped validation split $N = 4,545$).
  * Out-of-Time Future Test Partition (`data/benchmark/a5_temporal_test.csv`): $N = 9,339$ URLs (5,201 benign, 4,138 phishing across 7,646 unique registered domains).
  * *Reconciliation of Sample Size*: Note that an earlier exploratory script (`experiments/E05_temporal.py`) evaluated an ad-hoc downsampled, balanced subset of $N = 5,789$. The confirmatory evaluation under Protocol Amendment 1 evaluated the full frozen benchmark of $N = 9,339$, which yielded the reported $\text{AUC} = 0.9160$ and $95\%\ \text{CI}: [0.8987, 0.9319]$.
* **Hypotheses**:
  * $H_{5,0}$: Model discrimination is invariant across temporal epochs ($\Delta\text{AUC}_{A5} = \text{AUC}_{\text{in-time,val}} - \text{AUC}_{\text{out-of-time,test}} = 0$).
  * $H_{5,1}$: Temporal distribution shift causes performance decay ($\Delta\text{AUC}_{A5} > 0$).

### B. Empirical Results
* **In-Time Validation Split**: $\text{AUC} = 0.9807$, $\text{F1} = 0.9228$, $\text{Precision} = 0.9423$, $\text{Recall} = 0.9041$, $\text{FPR} = 4.66\%$ ($\tau = 0.3955$).
* **Out-of-Time Future Test Split**: $\text{AUC} = 0.9160$, $\text{F1} = 0.8513$, $\text{Precision} = 0.8920$, $\text{Recall} = 0.8142$, $\text{FPR} = 7.84\%$ ($\tau = 0.3955$).
* **Temporal Drop**: $\Delta\text{AUC}_{A5} = 0.9807 - 0.9160 = \mathbf{+0.0647}$ ($+6.47\ \text{AUC points decay}$).
* **Domain-Clustered Bootstrap CI for Out-of-Time Test AUC**:
  * Point Estimate: $0.9160$
  * Bootstrap Mean: $0.9161$ ($\text{std} = 0.0089$)
  * 95% Bootstrap Percentile Confidence Interval: $[0.8987, 0.9319]$ (Interval width: $0.0332$).
* **Inferential Conclusion**: Null hypothesis $H_{5,0}$ is rejected ($p < 0.001$). Temporal decay reduces discriminative AUC by $6.47$ points, confirming that static lexical models degrade rapidly under real-world threat evolution.

---

## 7. Appendix E06: Adversarial Perturbation Diagnostic

### A. Experimental Scope & Evaluated Perturbations
Appendix E06 evaluates the effect of 3 specific synthetic heuristic perturbations applied to phishing URLs in `global_test` ($N = 21,094$ total; evaluated on 10,547 phishing instances):
1. **Homoglyph Substitution**: Replacing ASCII characters with visually identical Unicode homoglyphs (e.g. Cyrillic 'а' for Latin 'a').
2. **Token Padding**: Appending benign-mimicking tokens and keywords into path segments (e.g. `login-verify-account-update`).
3. **Subdomain Trick**: Embedding targeted brand names as nested subdomains on unrelated domains (e.g. `sbi.co.in.verification-portal.com`).

*Scope Limitation*: These findings characterize model response exclusively to the specified 3 heuristic perturbation rules and do not constitute a universal proof of security or vulnerability across all adversarial attack spaces.

### B. Empirical Results
* **Clean Baseline Performance**: $\text{AUC} = 0.9284$, $\text{F1} = 0.8490$, $\text{Precision} = 0.9106$, $\text{Recall} = 0.7952$, $\text{FPR} = 8.58\%$ ($\tau = 0.5136$).
* **Perturbation 1 (Homoglyph)**: $\text{AUC} = 0.9504$, $\text{F1} = 0.8870$, $\text{Precision} = 0.9167$, $\text{Recall} = 0.8592$, $\text{FPR} = 8.58\%$.
* **Perturbation 2 (Token Padding)**: $\text{AUC} = 0.9887$, $\text{F1} = 0.9365$, $\text{Precision} = 0.9240$, $\text{Recall} = 0.9494$, $\text{FPR} = 8.58\%$.
* **Perturbation 3 (Subdomain Trick)**: $\text{AUC} = 0.9905$, $\text{F1} = 0.9560$, $\text{Precision} = 0.9267$, $\text{Recall} = 0.9872$, $\text{FPR} = 8.58\%$.

### C. Finding: The Lexical Bias Paradox
Rather than evading detection, all three heuristic perturbations **paradoxically increased phishing recall** (from $79.52\%$ clean up to $98.72\%$ on subdomain tricks). Because the underlying tree-based classifier strongly weights token length, path depth, and subdomain complexity as indicators of maliciousness, synthetic attempts to disguise URLs by padding keywords or stacking subdomains inadvertently amplified the very shortcut features that trigger phishing classifications.

---

## 8. Statistical Verification & Bootstrap Methodology Standards

All reported numerical values match the raw outputs stored in `research/results/` (`a1_shortcut_results.json`, `a2_representation_results.json`, `a3_brand_results.json`, `a4_cross_distribution_results.json`, `a5_temporal_results.json`, `e06_adversarial_results.json`, `summary_confirmatory_report.json`).

* **Statistical Reporting Standards**:
  1. No empirical bootstrap p-value is reported as literal `0.0000`. Where no bootstrap replicate crossed the null value across 1,000 iterations, the result is reported as $p < 0.001$.
  2. In all domain-clustered bootstrap routines, resampling registered-domain clusters with replacement and evaluating the resulting pool of URLs estimates **URL-weighted performance under domain-clustered sampling variance**. It does not estimate equal-domain-weighted performance.
  3. Replicate validity was strictly audited: all 1,000 bootstrap replicates were valid ($B_{\text{valid}} = 1,000$, $B_{\text{invalid}} = 0$) across all paired and single-distribution tests.

---

## 9. Comprehensive Confirmatory Synthesis Table

| Exp | Description | Primary Test Set ($N$) | Model / Condition A | Model / Condition B | Metric A | Metric B | $\Delta\text{AUC}$ [95% CI] | $p$-value | Decision / Outcome |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **A1** | Shortcut Sensitivity | `a1_path_matched` (21,094) | Unmasked LightGBM | Masked LightGBM | 0.9598 | 0.9317 | **+0.0281** [+0.0207, +0.0386] | $< 0.001$ | $H_{1,0}$ Rejected; path features aid AUC (+2.81 pts); Shortcut-only AUC = 0.9088 |
| **A2** | Syntactic Fragility | 50 Benign Domains | `https_bare` | `https_www_bare` | 0.0221 prob | 0.6380 prob | **+0.6159** mean prob shift | — | Extreme fragility; 39/50 (78%) benign homepages flip to phishing |
| **A3** | Unseen Brands | `a3_unseen_brands` (658) | ZK Regional Model | Generic LightGBM | 0.9480 | 0.9766 | **-0.0286** [-0.0427, -0.0186] | $< 0.001$ | $H_{3,0}$ Rejected; regional features degrade ZK AUC (-2.86 pts) |
| **A4** | Cross-Dist Transfer | `phiusiil_test` (3,520) | `global_test` (In-Dist) | PhiUSIIL (Cross-Dist) | 0.9284 | 0.8661 | **+0.0623 drop** ($> 0.05$ op. criterion) | — | Severe threshold failure: FPR explodes to 86.93% at fixed $\tau$ |
| **A4** | Disjoint Diagnostic | `phiusiil_disjoint` (2,888) | `global_test` (In-Dist) | Disjoint PhiUSIIL | 0.9284 | 0.8303 | **+0.0982 drop** ($> 0.05$ op. criterion) | — | Zero-shot domain transfer drops AUC to 0.8303; FPR = 86.95% |
| **A4** | Hosting Census | `.in` Census (1,682) | LightGBM on `.in` | — | 0.9379 | — | 95% CI: [0.8978, 0.9630] | — | 59.17% foreign candidate targets, 34.19% ambiguous, 6.64% domestic Indian |
| **A5** | Temporal Generalization | `a5_temporal_test` (9,339) | In-Time Validation | Out-of-Time Test | 0.9807 | 0.9160 | **+0.0647 temporal drop** | $< 0.001$ | $H_{5,0}$ Rejected; 95% CI on test AUC: [0.8987, 0.9319] |
| **E06** | Adversarial Diagnostic | `global_test` Perturbed | Clean Baseline | Perturbed Phish | 0.7952 rec | 0.8592–0.9872 rec | **Paradoxical Recall Increase** | — | Lexical bias paradox: heuristic perturbations amplify phishing recall |

---

## 10. Phase 4 Deliverables: Reproduction CLI & Gate G3 Paper Package

### A. Reproduction CLI Specification (`research/reproduce.py`)
To ensure full external reproducibility and auditability, a standalone CLI tool has been deployed:

```bash
# 1. Inspect and display calibrated empirical summary of Phase 3 results:
python -m research.reproduce inspect

# 2. Verify all benchmark datasets, registered domain counts, and split manifests:
python -m research.reproduce verify

# 3. Re-run an individual experiment (e.g. A1, A3, or A4):
python -m research.reproduce run --experiment a1

# 4. Re-run the full confirmatory suite (A1-A5, E06 appendix):
python -m research.reproduce run --all
```

### B. Revised Paper Package Ready for Gate G3
* **Paper Title**: *The Lexical Illusion: Shortcut Reliance, Syntactic Fragility, and Calibration Failures in Machine Learning Phishing Detection*
* **Core Scientific Abstract**:
  > Machine learning classifiers operating on uniform resource locators (URLs) frequently report near-perfect detection accuracies exceeding 98%. In this work, we conduct a pre-registered confirmatory evaluation dissecting the failure modes of tree-based lexical phishing detectors across six critical axes: structural shortcuts, syntactic representations, zero-knowledge brand generalization, external cross-distribution transfer, temporal decay, and adversarial perturbations. Using domain-clustered paired bootstrap statistics, we demonstrate that: (1) structural length and depth shortcuts alone achieve an ROC-AUC of 0.9088 without inspecting lexical content; (2) classifiers exhibit extreme syntactic fragility, where adding `www.` to bare HTTPS domains causes 78% of top benign homepages to flip to phishing predictions; (3) regional lexical features fail under zero-knowledge brand holdout evaluation ($\Delta\text{AUC} = -0.0286$, $p < 0.001$); (4) transferring classifiers to external corpora causes catastrophic operating-threshold transfer failure, where the false-positive rate explodes to 86.93% under uncalibrated decision thresholds; (5) chronological feed aging degrades discriminative AUC by 6.47 points; and (6) heuristic perturbations paradoxically increase phishing recall from 79.52% to 98.72% by inadvertently amplifying length and depth shortcut cues. Our findings demonstrate that URL-based phishing classifiers do not learn robust invariant representations of deception, but rather brittle operating shortcuts that require rigorous pre-inference canonicalization, cluster-aware evaluation, and mandatory threshold recalibration.

* **Target Paper Outline**:
  1. **Introduction**: The promise vs. reality of client-side and feed-level URL phishing detection.
  2. **Threat Model & Dataset Provenance**: Rigorous construction of leak-free splits, domain-clustered grouping, and the PhreshPhish/PhiUSIIL benchmark ecosystem.
  3. **Empirical Shortcut Reliance (A1)**: Dissecting path/query lexical utility vs. pure structural length features.
  4. **Syntactic Representation Sensitivity (A2)**: Demonstrating 78% benign false-positive flips under trivial URL prefix shifts.
  5. **The Zero-Knowledge Brand Dilemma (A3)**: Empirical evaluation of regional feature degradation and multi-tenant hosting overlaps.
  6. **Cross-Distribution Transfer & Calibration Collapse (A4)**: Distinguishing ranking discrimination from the 86.93% FPR operating failure; census of `.in` infrastructure.
  7. **Temporal Decay & Threat Drift (A5)**: Quantifying the longitudinal half-life of static lexical models.
  8. **Adversarial Perturbation Paradox (Appendix E06)**: Why heuristic evasion attacks backfire on tree-based models.
  9. **Defensive Recommendations**: Mandatory URL canonicalization, threshold recalibration protocols, and domain-disjoint evaluation standards.
  10. **Conclusion**: Moving past the lexical illusion toward robust multi-modal phishing defenses.
