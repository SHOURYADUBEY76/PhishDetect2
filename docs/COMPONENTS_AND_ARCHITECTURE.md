# System Documentation: Architecture & Components

## 1. System Overview & Philosophy

**Project Title:** Region-Aware Phishing URL Detection: Robust Generalization Across Geographic, Temporal, and Adversarial Distribution Shifts  
**Target Domain:** Final-Year Computer Science & Engineering Capstone Project / Production Prototype  

### 1.1 The Core Problem
Conventional machine learning phishing detectors experience severe performance degradation when exposed to distribution shifts:
1. **Geographic Shift**: Models trained on Western benchmarks misclassify legitimate non-Western institutional ccTLDs (e.g. `.gov.in`, `.nic.in`, `.co.in`) as obfuscation, while failing to detect regional brand lures.
2. **Temporal Drift**: Attackers constantly register novel domains and update infrastructure over time, eroding static model accuracy.
3. **Adversarial Evasion**: Attackers use homoglyph substitution (Cyrillic lookalikes), subdomain injection, and brand token padding to bypass lexical filters.
4. **AI-Generated Lures**: Modern attackers leverage LLMs to generate highly convincing, zero-shot targeted phishing URLs.

### 1.2 Architectural Principles
* **100% Static Offline String Parsing**: The detector treats URLs strictly as static strings. It **never** visits the URL, resolves DNS, queries WHOIS, or downloads web content. This guarantees zero execution risk and sub-millisecond performance ($1.5\text{--}3.5\text{ ms}$).
* **Leakage-Free Domain Isolation**: Group splitting on `tldextract` registered domains ensures that no second-level domain in the training set ever leaks into the test set.
* **Region-Aware Signals**: Combines statistical lexical features and hierarchical structural features with localized threat indicators specifically tuned for the Indian digital ecosystem.
* **Transparent Attribution**: Accompanied by a sub-millisecond, rule-based explanation layer that produces plain-English bullet points for non-technical users.

---

## 2. End-to-End System Architecture

```
                                  STATIC URL STRING
                                         │
                                         ▼
                 ┌───────────────────────────────────────────────┐
                 │       URLFeatureExtractor (src/features/)     │
                 │  ├── Lexical (24 features)                    │
                 │  ├── Structural (15 features)                 │
                 │  └── Regional (8 features)                    │
                 └───────────────────────┬───────────────────────┘
                                         │  (Numeric Feature Vector: 47 Dimensions)
                                         ▼
                 ┌───────────────────────────────────────────────┐
                 │     Trained Classifier (src/models/baseline)   │
                 │      LightGBM GBDT (0.50 MB, Latency ~2.7 ms) │
                 │      Calibrated Threshold: P(Phish | x) >= θ  │
                 └───────────────┬───────────────────────────────┘
                                 │
                 ┌───────────────┴───────────────────────────────┐
                 ▼                                               ▼
┌─────────────────────────────────┐             ┌─────────────────────────────────┐
│  Explanation Layer              │             │  FastAPI Backend Engine         │
│  (src/explainer/explainer.py)   │             │  (backend/main.py)              │
│  - Threat factor attribution    │             │  - GET /health                  │
│  - Plain-English warnings       │             │  - POST /predict                │
│  - Official domain matching     │             │  - POST /explain                │
└────────────────┬────────────────┘             └────────────────┬────────────────┘
                 │                                               │
                 └───────────────────────┬───────────────────────┘
                                         │
                         ┌───────────────┴───────────────┐
                         ▼                               ▼
         ┌───────────────────────────────┐ ┌───────────────────────────────┐
         │ React 19 + Vite Dashboard     │ │ Chrome Extension (MV3)        │
         │ (web/)                        │ │ (extension/)                  │
         │ - Real-time Risk Meter        │ │ - DOM Link Badges             │
         │ - Threat Explanation Cards    │ │ - Click Intercept Modal       │
         │ - Model Ablation Showcase     │ │ - Popup Risk Inspector        │
         └───────────────────────────────┘ └───────────────────────────────┘
```

---

## 3. Detailed Component Breakdown

### Component 1: Data Acquisition & Preprocessing Pipeline
**Location:** [`scripts/`](file:///c:/Users/HP/OneDrive/Documents/Desktop/Phishing/scripts/), [`src/utils/domain.py`](file:///c:/Users/HP/OneDrive/Documents/Desktop/Phishing/src/utils/domain.py), [`data/`](file:///c:/Users/HP/OneDrive/Documents/Desktop/Phishing/data/)

#### What It Does
Acquires, cleans, balances, and partitions phishing and legitimate URLs from multiple global and regional sources without data leakage.

#### Files & Responsibilities
1. **[`scripts/download_data.py`](file:///c:/Users/HP/OneDrive/Documents/Desktop/Phishing/scripts/download_data.py)**:
   * **PhreshPhish Ingestion**: Downloads Parquet partitions from Hugging Face (`phreshphish/phreshphish`) with individual per-file caching (`data/raw/phreshphish_cache/`) and retry logic ($N=47,703$).
   * **Tranco Legit Ingestion**: Evaluates top 200,000 domains via the Tranco list, partitioning into 25,000 global clean root domains (`tranco_global_legit.csv`) and 2,204 Indian legitimate institutional domains (`tranco_indian_legit.csv`).
   * **Indian Phishing Feeds**: Streams active lures from OpenPhish and Phishing.Database targeting Indian banks, telecom, and citizen services (`indian_phishing_raw.csv`).
   * **Balanced Root Domain Integration**: Merges non-overlapping Tranco global root domains into `global_train` to eliminate the root-versus-path structural artifact.
   * **Group Partitioning**: Groups URLs by `registered_domain` so that domains appearing in training sets never appear in validation or test sets.
2. **[`scripts/download_phiusiil.py`](file:///c:/Users/HP/OneDrive/Documents/Desktop/Phishing/scripts/download_phiusiil.py)**:
   * Fetches UCI PhiUSIIL Phishing Dataset (UCI ID 967).
   * Inverts labels from the original dataset format to the project standard: **$1 = \text{phishing}$**, **$0 = \text{legitimate}$**.
   * Constructs domain-isolated splits: `phiusiil_train.csv` (9,440) and `phiusiil_test.csv` (3,520).
3. **[`scripts/generate_ai_urls.py`](file:///c:/Users/HP/OneDrive/Documents/Desktop/Phishing/scripts/generate_ai_urls.py)**:
   * Generates zero-shot phishing lures targeting novel 2024–2026 digital initiatives (*PM Surya Ghar Solar Subsidy*, *Digi Yatra Airport Biometrics*, *Ayushman Bharat Golden Card*, *Kisan Credit*, *RuPay Rewards*, and *Neobanks*).
   * Implements explicit provenance metadata (`provenance: llm_gemini_api` vs `provenance: synthetic_template_standin`, `is_standin: bool`).
   * Audits registered domains to ensure strictly 0 overlap with training data.
4. **[`src/utils/domain.py`](file:///c:/Users/HP/OneDrive/Documents/Desktop/Phishing/src/utils/domain.py)**:
   * Core URL parsing utilities: `normalize_url`, `extract_registered_domain`, `extract_tld`, `extract_subdomain` using public suffix rules.

---

### Component 2: Multi-Tier Feature Extraction Engine
**Location:** [`src/features/`](file:///c:/Users/HP/OneDrive/Documents/Desktop/Phishing/src/features/)

#### What It Does
Transforms raw static URL strings into a fixed 47-dimensional numeric vector in under 1 millisecond. No network calls or socket connections are initiated.

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           FEATURE HIERARCHY (47 FEATURES)                       │
├──────────────────────────┬──────────────────────────┬───────────────────────────┤
│    LEXICAL (24)          │    STRUCTURAL (15)       │    REGIONAL (8)           │
├──────────────────────────┼──────────────────────────┼───────────────────────────┤
│ • URL / Host / Path Len  │ • Protocol (HTTPS/HTTP)  │ • is_indian_tld           │
│ • Query String Length    │ • Raw IPv4 / Hex IP      │ • has_indian_brand_token  │
│ • Digit Count & Ratio    │ • Subdomain Count & Depth│ • brand_token_count       │
│ • Character Counts (.,-,_)│• Excessive Subdomain Flag│ • has_indian_lure         │
│ • Delimiters (?,=,@,%,&) │ • Non-standard Port      │ • lure_token_count        │
│ • Special Char Ratio     │ • Path Depth             │ • brand_tld_misalignment  │
│ • Shannon Entropy:       │ • Query Parameter Count  │ • max_brand_similarity   │
│   - URL Entropy          │ • Suspicious Keywords    │ • has_homoglyph_or_punycode│
│   - Hostname Entropy     │ • Double Slash Redirect  │                           │
│   - Path Entropy         │ • Longest Host Token Len │                           │
│                          │ • TLD & Subdomain Length │                           │
└──────────────────────────┴──────────────────────────┴───────────────────────────┘
```

#### Detailed Logic
1. **[`lexical.py`](file:///c:/Users/HP/OneDrive/Documents/Desktop/Phishing/src/features/lexical.py)**:
   * Calculates Shannon entropy: $H(X) = -\sum P(x) \log_2 P(x)$ to detect randomly generated machine strings and DGA domains.
   * Tracks character distributions, symbol density, and digit ratios across hostname and path.
2. **[`structural.py`](file:///c:/Users/HP/OneDrive/Documents/Desktop/Phishing/src/features/structural.py)**:
   * Analyzes URL hierarchy: checks protocol security (`is_https`, `is_http`), raw IP hostnames, non-standard ports, redirection markers (`//` in path), and suspicious keyword counts (`login`, `verify`, `banking`, `secure`).
3. **[`regional.py`](file:///c:/Users/HP/OneDrive/Documents/Desktop/Phishing/src/features/regional.py)**:
   * **Brand Token Matching**: Scans for 48 high-value Indian brands across banking, payments, telecom, and citizen services, resolving aliases to canonical entities (e.g. `onlinesbi` $\to$ `sbi`, `uidai` $\to$ `aadhaar`).
   * **Homoglyph Normalization**: Normalizes Cyrillic lookalike characters (`HOMOGLYPH_MAP`) before brand matching to catch visual spoofing attacks (e.g. `ісісіbаnk.com`).
   * **Brand-TLD Misalignment**: Fires if a recognized brand token is detected in the URL, but the registered domain does not match the official registry set (`OFFICIAL_DOMAINS`).
   * **Action Lures**: Scans for regional urgency keywords (`kyc`, `pan-update`, `yono`, `biometric`, `cashback`, `subsidy`).
   * **Fuzzy Similarity**: Computes SequenceMatcher ratio against target brand names to catch typosquatting.
4. **[`extractor.py`](file:///c:/Users/HP/OneDrive/Documents/Desktop/Phishing/src/features/extractor.py)**:
   * Unified entry point providing `extract_features_dict` and `extract_features_vector`.
   * Supports `include_path_query=False` to execute source-bias audits.

---

### Component 3: Machine Learning & Deep Learning Core
**Location:** [`src/models/`](file:///c:/Users/HP/OneDrive/Documents/Desktop/Phishing/src/models/), [`backend/*.joblib`](file:///c:/Users/HP/OneDrive/Documents/Desktop/Phishing/backend/)

#### What It Does
Trains, calibrates, serializes, and serves high-efficiency tabular gradient-boosted trees and character-level deep neural networks.

#### Models Implemented
1. **[`baseline.py`](file:///c:/Users/HP/OneDrive/Documents/Desktop/Phishing/src/models/baseline.py) (`PhishingClassifier`)**:
   * Unified wrapper supporting **Logistic Regression**, **Random Forest**, and **LightGBM**.
   * **Threshold Calibration**: Automatically tunes decision threshold $\theta$ strictly on validation data (`find_optimal_threshold`) to maximize $F_1$-score without test-set leakage.
   * **Memory Measurement**: Serializes estimators into in-memory `io.BytesIO()` streams to report accurate non-zero model size in MB (LightGBM: $0.50\text{ MB}$, Logistic Regression: $0.001\text{ MB}$, Random Forest: $19.33\text{ MB}$).
   * **Persistence**: Saves and loads full bundles (model, scaler, optimal threshold, model type) via `joblib`.
2. **[`bilstm.py`](file:///c:/Users/HP/OneDrive/Documents/Desktop/Phishing/src/models/bilstm.py) (`BiLSTMClassifier`)**:
   * **CharacterTokenizer**: Vocabulary indexer mapping raw printable ASCII characters (indices 1..95, `<UNK>`: 96, padding: 0) with a max sequence length of 150.
   * **Neural Architecture**:
     * `Embedding(input_dim=97, output_dim=32)`
     * `Bidirectional(LSTM(units=64))`
     * `Dropout(0.3)`
     * `Dense(32, activation="relu")`
     * `Dense(1, activation="sigmoid")`
   * Used in Experiment E04 for cross-dataset deep learning comparisons.
3. **Serialized Production Artifacts**:
   * `backend/model_region_aware.joblib`: Production 47-feature LightGBM model ($0.50\text{ MB}$).
   * `backend/model_baseline.joblib`: Standard 39-feature LightGBM model ($0.50\text{ MB}$).

---

### Component 4: Transparent Explainability Layer
**Location:** [`src/explainer/explainer.py`](file:///c:/Users/HP/OneDrive/Documents/Desktop/Phishing/src/explainer/explainer.py)

#### What It Does
Translates numeric feature contributions into clear, plain-English justification bullet points. Operates deterministically in under $0.5\text{ ms}$ without calling heavy, slow, or non-deterministic external LLMs.

#### Decision Rules
* **Brand Impersonation**: Triggers when `brand_tld_misalignment == 1.0`, explicitly naming the impersonated brand and highlighting that the registered domain is unauthorized.
* **Credential Lures**: Identifies specific urgent triggers (e.g. `kyc`, `yono`, `unblock`, `subsidy`).
* **Homoglyph Detection**: Explains that lookalike non-ASCII characters or Punycode are spoofing a trusted name.
* **Subdomain Masking**: Flags nesting ($\ge 3$ levels) designed to obscure destination domains.
* **Transport Security**: Explains unencrypted HTTP or non-standard ports.
* **Official Registry Verification**: When matching `OFFICIAL_DOMAINS` (or `.gov.in`/`.nic.in`), explains that the domain is officially verified, establishing trust.

---

### Component 5: Experimental Evaluation Suite (E01–E09)
**Location:** [`experiments/`](file:///c:/Users/HP/OneDrive/Documents/Desktop/Phishing/experiments/), [`results/`](file:///c:/Users/HP/OneDrive/Documents/Desktop/Phishing/results/)

#### What It Does
Stress-tests detection models against distinct distribution shifts across 5 random seeds ($\text{Seeds} = [42, 43, 44, 45, 46]$) with registered domain isolation.

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                        EXPERIMENTAL SUITE ROADMAP (E01 - E09)                   │
├──────┬───────────────────────────────┬──────────────────────────────────────────┤
│ Exp  │ Target Shift / Hypothesis     │ Key Finding                              │
├──────┼───────────────────────────────┼──────────────────────────────────────────┤
│ E01  │ Standard Global Benchmark     │ LightGBM achieves 84.22% F1 (0.50 MB).   │
│      │ Source Bias Audit (No PQ)     │ Removing Path/Query drops F1 by only 0.3%│
│ E02  │ Regional Transfer Gap         │ Global models suffer 7.84% FNR on Indian │
│      │                               │ phishing lures without regional signals. │
│ E03  │ 3-Way Feature Ablation        │ Region-aware signals slash FNR from      │
│      │ (Lexical -> Struct -> Region) │ 5.52% down to 1.91% (F1 = 99.02%).       │
│ E04  │ Cross-Dataset Transfer        │ PhreshPhish <-> PhiUSIIL domain shift;   │
│      │ (LightGBM vs BiLSTM)          │ BiLSTM transfers better on character seqs│
│      │                               │ but runs 30x slower (55ms vs 1.6ms).     │
│ E05  │ Temporal Concept Drift        │ Models generalize chronologically        │
│      │ (Early 2024 -> Late 2025)     │ across historical timestamps (87.92% F1).│
│ E06  │ Adversarial Evasion           │ Clean model resists mutations (99.16%);  │
│      │ (Homoglyphs, Padding, Trees)  │ Adversarial retraining hits 99.97% F1.   │
│ E07  │ Zero-Shot Withheld Brands     │ 12 brands withheld during training:      │
│      │ (N = 600 balanced test)       │ 99.90% F1, proving no brand memorization.│
│ E08  │ Emerging AI-Generated URLs    │ Zero-shot detection of novel synthetic   │
│      │ (N = 500 balanced test)       │ lures reaches 100% recall.               │
│ E09  │ Multi-Shift Compound Stress   │ Simultaneous regional + temporal +       │
│      │ (Compound multi-shift test)   │ adversarial stress: Region-Aware = 99.19%│
└──────┴───────────────────────────────┴──────────────────────────────────────────┘
```

Outputs are saved as standalone CSV files (`results/EXX.csv`) and synthesized into [`results/SUMMARY.md`](file:///c:/Users/HP/OneDrive/Documents/Desktop/Phishing/results/SUMMARY.md).

---

### Component 6: REST API Backend
**Location:** [`backend/main.py`](file:///c:/Users/HP/OneDrive/Documents/Desktop/Phishing/backend/main.py)

#### What It Does
Serves production model predictions, risk confidence scores, and plain-English explanations over high-performance HTTP REST endpoints.

#### Key Endpoints
* `GET /health`: Probes server health, loaded model type, and calibrated threshold.
* `POST /predict`: Analyzes a static URL, returning binary verdict, risk score, confidence rating, calibrated threshold, latency, and key signals.
* `POST /explain`: Returns the full threat analysis including risk category (`LOW`, `MEDIUM`, `HIGH`, `CRITICAL`) and bulleted human-readable explanations.
* **Modern Lifespan Architecture**: Uses FastAPI `asynccontextmanager` lifespan handlers to warm up models during initialization.
* **CORS Middleware**: Allows requests from localhost web apps and Chrome extension origins (`chrome-extension://*`).

---

### Component 7: React 19 Web Dashboard
**Location:** [`web/`](file:///c:/Users/HP/OneDrive/Documents/Desktop/Phishing/web/)

#### What It Does
Provides an interactive graphical dashboard for inspecting URLs, reviewing model risk assessments, exploring preloaded test cases, and viewing feature ablation metrics.

#### Technical Stack & Highlights
* **React 19 + Vite**: High-speed, modern component architecture (production bundle compiles in $<3\text{ seconds}$).
* **Tailwind CSS**: Modern responsive cybersecurity dashboard design.
* **Lucide Icons**: Visual status cues (shields, alerts, checkmarks, gauges).
* **Interactive Elements**:
  * **Real-Time Risk Meter**: Dynamic semi-circular gauge displaying risk percentages with color transitions (Emerald Safe $\to$ Amber Warning $\to$ Crimson Threat).
  * **Interactive Case Selector**: Instant 1-click loading of official banking, government, adversarial spoofing, and AI-generated phishing test URLs.
  * **Explanation Cards**: Plain-English threat justification bullet points and official domain comparisons.
  * **Ablation Comparison Modal**: Displays experimental metrics across Lexical, Structural, and Region-Aware configurations.

---

### Component 8: Chrome Browser Extension (Manifest V3)
**Location:** [`extension/`](file:///c:/Users/HP/OneDrive/Documents/Desktop/Phishing/extension/)

#### What It Does
Provides real-time browser protection by inspecting links on active webpages, adding risk badges, and intercepting navigation to phishing sites.

#### Key Files
* `manifest.json`: Manifest V3 specification with host permissions and declarative content scripts.
* `background.js`: Service worker managing extension lifecycle, badge counts, and communication with the backend API.
* `content.js`: Injected DOM script that:
  1. Scans anchor elements (`<a>`) on web pages.
  2. Injects small, unobtrusive shield badges next to links (Green = Safe, Red = Phishing).
  3. Intercepts clicks on high-risk links, displaying an accessible modal with threat bullet points and an option to abort or proceed.
* `popup.html` & `popup.js`: Extension popup for scanning the current tab's active URL or manually inspecting suspicious links.

---

### Component 9: Verification & Testing Suite
**Location:** [`tests/`](file:///c:/Users/HP/OneDrive/Documents/Desktop/Phishing/tests/), [`scripts/demo_offline.py`](file:///c:/Users/HP/OneDrive/Documents/Desktop/Phishing/scripts/demo_offline.py)

#### What It Does
Maintains continuous code quality, numerical stability, and regression protection across all project modules.

#### Verification Assets
1. **Automated Test Suite (`pytest tests/ -v`)**:
   * **[`tests/test_phase1.py`](file:///c:/Users/HP/OneDrive/Documents/Desktop/Phishing/tests/test_phase1.py)**: Tests feature extraction vector sizes, model initialization, threshold tuning, and FastAPI `/predict` and `/health` routes.
   * **[`tests/test_phase2.py`](file:///c:/Users/HP/OneDrive/Documents/Desktop/Phishing/tests/test_phase2.py)**: Tests regional feature extraction, explainer output structure, and API `/explain` endpoints.
   * **[`tests/test_phase3.py`](file:///c:/Users/HP/OneDrive/Documents/Desktop/Phishing/tests/test_phase3.py)**: Tests `CharacterTokenizer` sequence bounds, `BiLSTMClassifier` forward inference, adversarial perturbation generators, latency measurement routines, and strict domain isolation across all processed splits.
   * **Test Pass Rate**: **12 / 12 tests passing** (100%).
2. **Offline Interactive Inspector (`scripts/demo_offline.py`)**:
   * Demonstrates zero-network static inference from terminal.
   * Evaluates curated showcase batteries with exact microsecond latency breakdowns.
   * Includes interactive CLI prompt (`--interactive`) allowing users to paste any custom URL for immediate offline evaluation.

---

## 4. How the Components Work Together: Step-by-Step Walkthrough

### Scenario A: End-to-End User URL Inspection
```
1. User enters: "http://sbi-kyc-verification-update-urgent.xyz/login.php" in Web UI or Browser
2. Frontend/Extension sends POST request to backend API (or demo_offline.py executes locally).
3. Backend normalizes URL and calls URLFeatureExtractor:
   - Lexical: High entropy (4.18), high length (54 chars), 4 slashes.
   - Structural: is_http=1.0, is_https=0.0, suspicious_keywords=2 ("kyc", "login").
   - Regional: has_indian_brand_token=1.0 ("sbi"), has_indian_lure=1.0 ("kyc"),
               brand_tld_misalignment=1.0 ("sbi-kyc-verification-update-urgent.xyz" != official SBI).
4. Feature vector (47 float values) is fed into model_region_aware.joblib.
5. LightGBM outputs: P(Phishing | x) = 0.9986.
6. Since 0.9986 >= 0.45 (optimal threshold), verdict = "PHISHING", risk_level = "CRITICAL".
7. Explainer analyzes feature triggers:
   - "Brand Impersonation: Displays 'SBI' lure, but registered domain is not an official authorized portal."
   - "Credential Lure: Contains high-risk lure keyword ('kyc') frequently used in regional financial phishing."
   - "Insecure Protocol: Transmits over unencrypted HTTP without TLS verification."
8. API packages response with sub-millisecond execution time (~3.2 ms).
9. Web Dashboard renders Crimson Risk Meter (99.9%) and threat explanation cards.
10. Extension intercepts user navigation, displaying modal warning to protect the user.
```

### Scenario B: Official Institutional Domain Verification
```
1. User enters: "https://www.onlinesbi.sbi"
2. Extractor checks registered domain: "onlinesbi.sbi".
3. Regional module queries OFFICIAL_DOMAINS registry: "onlinesbi.sbi" matches official SBI portal.
4. brand_tld_misalignment = 0.0, has_ip = 0.0, is_https = 1.0.
5. Official safelist dampens threat score: risk_score = 0.02.
6. Verdict = "LEGITIMATE", risk_level = "LOW".
7. Explainer outputs:
   - "Official Domain: Verified as an authentic registered portal for State Bank of India."
   - "Secure Transport: Valid standard HTTPS protocol configuration."
   - "Clean Architecture: Direct domain routing with standard hierarchical depth."
8. Dashboard renders Emerald Shield with 2% threat rating.
```

---

## 5. Technology Stack Summary

| Layer | Technologies Used |
|---|---|
| **Core Language** | Python 3.12+ |
| **Machine Learning** | LightGBM, Scikit-learn, Joblib |
| **Deep Learning** | TensorFlow 2.16+, Keras 3 |
| **Data Processing** | Pandas, NumPy, PyArrow (Parquet), Tldextract |
| **Backend REST API** | FastAPI, Uvicorn, Pydantic |
| **Frontend Web App** | React 19, Vite, Tailwind CSS, Lucide Icons |
| **Browser Extension** | Chrome Manifest V3, Vanilla JavaScript, CSS3 |
| **Testing & Quality** | Pytest, Unittest |
