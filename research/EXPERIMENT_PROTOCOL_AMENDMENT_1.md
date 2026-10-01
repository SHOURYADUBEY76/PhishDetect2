# Protocol Amendment 1: Methodological Hardening of Track A Protocols

**Date**: 2026-10-01  
**Status**: Adopted prior to Phase 3 confirmatory runs  
**Baseline Frozen Protocol**: `research/EXPERIMENT_PROTOCOL.md` (tagged `prereg-v2` at commit `b4a5d2a`)  
**Scope**: Methodological clarifications and benchmark corrections for Experiments A3, A4, Hosting Analysis, and Clustered Bootstrapping.

---

## 1. Experiment A3: Brand Generalization & Statistical Inference

### A. Hypothesis Refinement (Two-Tailed Null Hypothesis)
* **Null Hypothesis ($H_{3,0}$)**: Under zero-knowledge brand evaluation (where target brands are strictly excluded from training data and completely purged from the feature extractor's `LexiconConfig`), the regional-feature-augmented model's ROC-AUC does not differ from the generic lexical/structural baseline model:
  $$\Delta \text{AUC}_{A3} = \text{AUC}_{\text{regional,zk}} - \text{AUC}_{\text{generic}} = 0$$
* **Alternative Hypothesis ($H_{3,1}$)**: Under zero-knowledge brand evaluation, the regional-feature-augmented model's ROC-AUC differs from the generic baseline model ($\Delta \text{AUC}_{A3} \neq 0$).
* **Statistical Conclusion & Equivalence Rule**:
  * We report the empirical point estimate $\hat{\Delta} \text{AUC}$, its 95% domain-clustered bootstrap percentile confidence interval $[\Delta_{(0.025)}, \Delta_{(0.975)}]$, and the absolute interval width $W = \Delta_{(0.975)} - \Delta_{(0.025)}$.
  * The difference is statistically significant at $\alpha = 0.05$ if and only if $0.0 \notin [\Delta_{(0.025)}, \Delta_{(0.975)}]$.
  * **No Equivalence Claim**: Failure to reject the null hypothesis ($0.0 \in \text{CI}_{95\%}$) does **not** establish equivalence between models. In the absence of a pre-registered equivalence margin $[-\delta, +\delta]$, results spanning zero are reported strictly as failing to detect a statistically significant difference at $\alpha = 0.05$, with the maximum difference bounded by the confidence interval.

### B. Attribution Rules and Handling of Ambiguous / Sparse Brands
* **Ground-Truth Target Attribution**:
  * *Polysemic Tokens* (`bob`, `axis`, `pan`): Cannot qualify through substring matches alone. Attribution requires either:
    1. Unambiguous compound naming (e.g. `bankofbaroda`, `axisbank`), OR
    2. Upstream ground-truth threat feed metadata explicitly identifying the entity in its `target` field.
    *All other standalone or substring matches on polysemic tokens are classified as `AMBIGUOUS` and strictly excluded.*
  * *Non-Polysemic Brands* (`sbi`, `hdfc`, `icici`, `phonepe`, `paytm`, `uidai`, `irctc`, `jio`): Matched with strict token boundaries (`(?:^|[^a-zA-Z0-9])<brand>(?:[^a-zA-Z0-9]|$)`) on host or path, confirmed not to be official registered domains.
* **Sparse-Brand Reporting Rule**:
  * Disaggregated individual-brand metrics are reported if and only if the brand stratum contains $\ge 20$ instances and $\ge 5$ distinct registered-domain clusters.
  * Brands with fewer than 20 instances or 5 domain clusters are flagged with `is_sparse_stratum = 1` and excluded from individual reporting, but remain fully included in the pooled aggregate benchmark.
* **Benchmark Replacement**: The previous preliminary benchmark (`data/benchmark/a3_unseen_brands.csv`, which contained 83.9% false-positive substring noise on "bob" and "axis") is permanently discarded and replaced by the validated non-synthetic benchmark ($N = 658$; 329 phishing, 329 benign).

---

## 2. Experiment A4: External Cross-Distribution Benchmarks

### A. Primary vs. Secondary Benchmark Architecture
* **Primary Cross-Distribution Benchmark (`data/benchmark/a4_cross_distribution.csv`)**:
  * Full external PhiUSIIL test set ($N = 3,520$; 1,760 benign, 1,760 phishing across 2,838 unique registered domains).
  * **Target Estimand**: Out-of-distribution performance under natural cross-corpus deployment, where shared hosting and infrastructure platforms (e.g. `workers.dev`, `vercel.app`, `amazonaws.com`) naturally recur across independent corpora.
  * Retains 100% of the external dataset without filtering or class distortion.
* **Secondary Domain-Disjoint Diagnostic (`data/benchmark/a4_domain_disjoint.csv`)**:
  * Strict domain-disjoint partition ($N = 2,888$; 1,701 benign, 1,187 phishing across 2,746 unique registered domains).
  * Purges all 92 overlapping registered domains that appear in any training partition (`global_train`, `a5_temporal_train`, `indian_train`).
  * **Target Estimand**: Strict zero-shot domain generalization ($D_{\text{train}} \cap D_{\text{test}} = \emptyset$).
  * **Documented Data Loss**: Excludes 632 rows ($17.95\%$ of `phiusiil_test`), of which $90.66\%$ ($573/632$) are phishing URLs hosted on shared infrastructure, shifting class balance to $41.10\%$ phishing.

### B. Post-Preregistration Operational Effect Size Criterion ($\Delta \text{AUC} > 0.05$)
* The criterion $\Delta \text{AUC} = \text{AUC}_{\text{in-distribution}} - \text{AUC}_{\text{cross-distribution}} > 0.05$ is designated as a **post-preregistration researcher-selected operational criterion**, not a universal standard or empirical law.
* It serves as a practical benchmark for qualitative discussion of operational degradation, evaluated alongside the primary pre-registered inferential test (whether the 95% bootstrap CI excludes $0.0$).

---

## 3. Dedicated Hosting Infrastructure Census

### A. Population and Accounting
* Decoupled into `data/benchmark/hosting_infrastructure_census.csv` ($N = 1,682$ URLs from the streamed PhreshPhish census).
* **Composition**: 1,205 phishing URLs (496 registered domains), 477 benign URLs (263 registered domains), 0 cross-class domain overlap.
* **Mutually Exclusive Phishing Target Accounting**:
  1. *Identified Foreign / Global Targets*: **713 URLs (59.17%)** across 118 international feed-target classifications (`google`, `facebook`, `whatsapp`, `canva`, `roblox`, etc.).
  2. *Ambiguous / Unspecified Targets*: **412 URLs (34.19%)** across generic feed categories (`other`: 370, `genericwebmailphishing`: 30, etc.). Explicitly not classified as foreign.
  3. *Domestic Indian Targets*: **80 URLs (6.64%)** across 19 domestic candidate entities (`governmentofindia`: 29, `della adventure`: 8, `sterling holidays`: 7, `meesho`: 6, `npci`: 6, `flipkart`: 5, etc.).
* **Descriptive Framing**: Findings are framed strictly as descriptive evidence of URLs hosted under Indian ccTLDs (`.in` and `.co.in`) based on candidate feed-target metadata. No claims of server physical location, server compromise, or verified victim identities are asserted.

---

## 4. Clustered Bootstrap Resampling Specification

* **Estimand**: Resampling registered-domain clusters with replacement and retaining all URLs within sampled clusters estimates **URL-weighted performance under domain-clustered sampling variance**. It is not equal-domain-weighted performance.
* **Stratified Resampling**: Benign domain clusters ($D_0$) and phishing domain clusters ($D_1$) are resampled independently, ensuring both classes are represented in every bootstrap replicate.
* **Handling of Invalid Replicates**:
  * Replicates yielding undefined metrics (e.g. division by zero in precision if $\text{TP} + \text{FP} = 0$) are recorded.
  * The count $B_{\text{invalid}}$ is explicitly tracked and reported.
  * Percentile confidence intervals are computed over valid replicates ($B_{\text{valid}} = B - B_{\text{invalid}}$) rather than silently replacing undefined values.
* **Operational Thresholds**: Decision thresholds ($\tau$) are fixed a priori on validation splits and held constant across all bootstrap iterations. Paired differences $\Delta^{(b)} = M_1^{(b)} - M_2^{(b)}$ are computed on identical resampled datasets.
