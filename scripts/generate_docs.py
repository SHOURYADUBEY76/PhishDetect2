"""
Generate Project Documentation DOCX
Run: python scripts/generate_docs.py
Output: docs/Project_Documentation.docx
"""

import os
from docx import Document
from docx.shared import Pt, RGBColor, Inches, Cm
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.oxml import OxmlElement

# ── helpers ────────────────────────────────────────────────────────────────────

def add_heading(doc, text, level=1, color=None):
    h = doc.add_heading(text, level=level)
    if color:
        for run in h.runs:
            run.font.color.rgb = RGBColor(*color)
    return h


def add_paragraph(doc, text, bold=False, italic=False, size=11):
    p = doc.add_paragraph()
    run = p.add_run(text)
    run.bold = bold
    run.italic = italic
    run.font.size = Pt(size)
    return p


def add_bullet(doc, text, level=0):
    p = doc.add_paragraph(text, style="List Bullet")
    p.paragraph_format.left_indent = Inches(0.25 * (level + 1))
    return p


def add_table(doc, headers, rows, col_widths=None):
    table = doc.add_table(rows=1, cols=len(headers))
    table.style = "Table Grid"
    # header row
    hdr_cells = table.rows[0].cells
    for i, h in enumerate(headers):
        hdr_cells[i].text = h
        for run in hdr_cells[i].paragraphs[0].runs:
            run.bold = True
        hdr_cells[i].paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER
        tc = hdr_cells[i]._tc
        tcPr = tc.get_or_add_tcPr()
        shd = OxmlElement("w:shd")
        shd.set(qn("w:val"), "clear")
        shd.set(qn("w:color"), "auto")
        shd.set(qn("w:fill"), "1F3864")
        tcPr.append(shd)
        for run in hdr_cells[i].paragraphs[0].runs:
            run.font.color.rgb = RGBColor(255, 255, 255)
    # data rows
    for row_data in rows:
        row_cells = table.add_row().cells
        for i, val in enumerate(row_data):
            row_cells[i].text = str(val)
    if col_widths:
        for i, w in enumerate(col_widths):
            for row in table.rows:
                row.cells[i].width = Cm(w)
    return table


def add_divider(doc):
    doc.add_paragraph("─" * 80)


# ── main ───────────────────────────────────────────────────────────────────────

def generate():
    os.makedirs("docs", exist_ok=True)
    doc = Document()

    # Page margins
    section = doc.sections[0]
    section.top_margin    = Cm(2.0)
    section.bottom_margin = Cm(2.0)
    section.left_margin   = Cm(2.5)
    section.right_margin  = Cm(2.5)

    # ── Cover ─────────────────────────────────────────────────────────────────
    title = doc.add_heading("Region-Aware Phishing URL Detection", 0)
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER

    sub = doc.add_paragraph(
        "Robust Generalization Across Geographic, Temporal, and Adversarial Distribution Shifts"
    )
    sub.alignment = WD_ALIGN_PARAGRAPH.CENTER
    sub.runs[0].italic = True
    sub.runs[0].font.size = Pt(13)

    doc.add_paragraph("")
    info = doc.add_paragraph(
        "Project Type:  Final-Year B.Tech CSE Capstone Project\n"
        "Domain:        Cybersecurity — Machine Learning — Web Application\n"
        "Target Region: Indian Digital Ecosystem (UPI, Banking, Government Portals)\n"
        "Language:      Python 3.12 | JavaScript (React 19)\n"
        "Status:        Complete — All 9 Experiments Executed, API Live, Frontend Ready"
    )
    info.alignment = WD_ALIGN_PARAGRAPH.CENTER

    doc.add_page_break()

    # ── TABLE OF CONTENTS (manual) ─────────────────────────────────────────────
    add_heading(doc, "Table of Contents", level=1)
    toc = [
        "1.  Project Overview",
        "2.  Problem Statement",
        "3.  Datasets Used",
        "4.  System Architecture",
        "5.  Feature Engineering (47 Features)",
        "6.  Machine Learning Models",
        "7.  Experimental Results & Model Comparison",
        "8.  Backend REST API",
        "9.  React Web Dashboard (Frontend)",
        "10. Chrome Browser Extension",
        "11. How to Run the Project",
        "12. Folder Structure",
        "13. Technology Stack",
    ]
    for line in toc:
        add_bullet(doc, line)

    doc.add_page_break()

    # ── 1. PROJECT OVERVIEW ────────────────────────────────────────────────────
    add_heading(doc, "1. Project Overview", level=1)
    add_paragraph(doc,
        "This project builds a machine learning system that detects phishing URLs in real-time "
        "by analysing the URL string — without visiting the website, resolving DNS, or making "
        "any network calls. The system is specifically designed to detect phishing attacks "
        "targeting the Indian digital ecosystem: UPI payment apps, nationalised banks, "
        "government citizen service portals, and telecom providers."
    )
    add_paragraph(doc, "")
    add_paragraph(doc, "Key achievements:", bold=True)
    add_bullet(doc, "47-feature region-aware feature extractor (lexical + structural + Indian brand signals)")
    add_bullet(doc, "LightGBM model: 99.02% F1-score on Indian phishing test set")
    add_bullet(doc, "Sub-3ms inference latency, 0.50 MB model size")
    add_bullet(doc, "Zero-shot detection of 12 completely unseen Indian brands: 99.90% F1")
    add_bullet(doc, "Adversarially retrained model: 99.97% F1, 0% False Negative Rate")
    add_bullet(doc, "Full REST API (FastAPI), React 19 web dashboard, and Chrome Extension (MV3)")

    # ── 2. PROBLEM STATEMENT ──────────────────────────────────────────────────
    add_heading(doc, "2. Problem Statement", level=1)
    add_paragraph(doc,
        "Conventional phishing detection models fail in real-world deployment because they are "
        "trained on Western benchmark datasets. When applied to Indian internet traffic, they suffer "
        "from four major failure modes:"
    )
    add_bullet(doc, "Geographic Shift: Models flag legitimate Indian ccTLDs (.gov.in, .nic.in, .co.in) as phishing")
    add_bullet(doc, "Temporal Drift: New phishing domains registered after training are missed")
    add_bullet(doc, "Adversarial Evasion: Attackers use Cyrillic lookalike characters (homoglyphs) and subdomain injection to bypass filters")
    add_bullet(doc, "AI-Generated Lures: LLMs now generate convincing phishing URLs targeting new government schemes")
    add_paragraph(doc, "")
    add_paragraph(doc,
        "Our solution introduces region-aware features — signals specifically designed for the "
        "Indian threat landscape — and demonstrates their quantified contribution across 9 systematic "
        "experiments."
    )

    # ── 3. DATASETS ───────────────────────────────────────────────────────────
    add_heading(doc, "3. Datasets Used", level=1)
    add_paragraph(doc,
        "The project uses four publicly available datasets and two self-collected feeds. "
        "All datasets are combined and split at the registered-domain level to ensure "
        "zero data leakage between training and test sets."
    )
    doc.add_paragraph("")

    add_heading(doc, "3.1  PhreshPhish (Primary Phishing Dataset)", level=2)
    add_bullet(doc, "Source: Hugging Face — phreshphish/phreshphish")
    add_bullet(doc, "Size: 47,703 labelled phishing URLs")
    add_bullet(doc, "Coverage: Global phishing campaigns, continuously updated crawl")
    add_bullet(doc, "Used for: Global training set, global test set, temporal drift experiment (E05)")
    add_bullet(doc, "File: data/raw/phreshphish_raw.csv")

    add_heading(doc, "3.2  Tranco Top-1M (Legitimate URLs Dataset)", level=2)
    add_bullet(doc, "Source: Tranco list (https://tranco-list.eu) — research-grade domain ranking")
    add_bullet(doc, "Size: Top 200,000 domains evaluated; 25,000 sampled as global legitimate URLs")
    add_bullet(doc, "Indian Subset: 2,204 Indian institutional domains (.gov.in, .nic.in, .co.in, .edu.in)")
    add_bullet(doc, "Used for: Legitimate class in all training and test splits")
    add_bullet(doc, "Files: data/raw/tranco_global_legit.csv, data/raw/tranco_indian_legit.csv")

    add_heading(doc, "3.3  UCI PhiUSIIL Phishing Dataset (Cross-Dataset Experiment)", level=2)
    add_bullet(doc, "Source: UCI Machine Learning Repository — Dataset ID 967")
    add_bullet(doc, "Size: 12,960 samples (9,440 train / 3,520 test)")
    add_bullet(doc, "Used for: E04 cross-dataset generalization experiment only")
    add_bullet(doc, "Purpose: Tests whether a model trained on PhreshPhish can predict on an entirely different corpus")
    add_bullet(doc, "File: data/raw/phiusiil_raw.csv")

    add_heading(doc, "3.4  OpenPhish Indian Phishing Feed (Regional Dataset)", level=2)
    add_bullet(doc, "Source: OpenPhish.com + Phishing.Database (GitHub) — live phishing feed")
    add_bullet(doc, "Content: Active phishing lures targeting SBI, HDFC, Paytm, IRCTC, Aadhaar, etc.")
    add_bullet(doc, "Size: Collected Indian-specific phishing URLs")
    add_bullet(doc, "Used for: Indian phishing training and test sets (E02, E03, E07, E09)")
    add_bullet(doc, "File: data/raw/indian_phishing_raw.csv")

    add_heading(doc, "3.5  Synthetic AI Phishing URLs (Self-Generated)", level=2)
    add_bullet(doc, "Source: Template-based generation targeting 2024-2026 government schemes")
    add_bullet(doc, "Schemes targeted: PM Surya Ghar Solar Subsidy, Digi Yatra Biometrics, Ayushman Bharat Golden Card, Kisan Credit Card, RuPay Rewards")
    add_bullet(doc, "Size: 500 balanced test samples")
    add_bullet(doc, "Used for: E08 AI-generated phishing detection experiment")
    add_bullet(doc, "Files: data/raw/ai_generated_urls.csv, data/raw/ai_phishing_standin.csv")

    add_heading(doc, "3.6  Final Processed Splits", level=2)
    add_table(doc,
        ["Split", "Rows", "Purpose"],
        [
            ["global_train.csv", "40,708", "Global model training"],
            ["global_test.csv", "7,434", "Global model evaluation (E01, E05)"],
            ["indian_train.csv", "1,284", "Indian regional model training"],
            ["indian_test.csv", "3,082", "Indian phishing evaluation (E02, E03, E07)"],
        ],
        col_widths=[5, 3, 8]
    )

    doc.add_page_break()

    # ── 4. ARCHITECTURE ───────────────────────────────────────────────────────
    add_heading(doc, "4. System Architecture", level=1)
    add_paragraph(doc,
        "The system is a pipeline: a URL string enters at one end and a verdict, risk score, "
        "and plain-English explanation come out the other end. Every step operates entirely offline "
        "— no web requests, DNS lookups, or WHOIS queries are made."
    )
    doc.add_paragraph("")
    add_paragraph(doc, "Pipeline flow (left to right):", bold=True)
    add_bullet(doc, "User inputs URL string (web dashboard / Chrome extension / terminal CLI)")
    add_bullet(doc, "URL is normalised (lowercase, scheme added if missing)")
    add_bullet(doc, "Feature Extractor converts URL → 47 numeric values in <1ms")
    add_bullet(doc, "LightGBM model scores the vector: P(Phishing | features)")
    add_bullet(doc, "Risk score compared to calibrated threshold (≈0.45) → binary verdict")
    add_bullet(doc, "Rule-based Explainer maps feature values → plain-English bullet points")
    add_bullet(doc, "Result returned to frontend (verdict + risk % + explanation)")

    doc.add_paragraph("")
    add_paragraph(doc, "Components:", bold=True)

    add_table(doc,
        ["Component", "Location", "Technology"],
        [
            ["Feature Extractor", "src/features/", "Python (regex, tldextract)"],
            ["ML Model", "src/models/baseline.py", "LightGBM GBDT"],
            ["Deep Learning Model", "src/models/bilstm.py", "TensorFlow/Keras BiLSTM"],
            ["Explainer Layer", "src/explainer/explainer.py", "Rule-based Python"],
            ["REST API Backend", "backend/main.py", "FastAPI + Uvicorn"],
            ["Web Dashboard", "web/src/App.jsx", "React 19 + Vite + Tailwind CSS"],
            ["Browser Extension", "extension/", "Chrome MV3 + Vanilla JS"],
            ["Tests", "tests/", "Pytest (12 tests, 100% passing)"],
        ],
        col_widths=[5, 5, 7]
    )

    doc.add_page_break()

    # ── 5. FEATURES ───────────────────────────────────────────────────────────
    add_heading(doc, "5. Feature Engineering (47 Features)", level=1)
    add_paragraph(doc,
        "The feature extractor transforms a raw URL string into a 47-dimensional numeric vector. "
        "Features are grouped into three tiers, each targeting a different attack vector."
    )

    add_heading(doc, "5.1  Lexical Features (24 features) — src/features/lexical.py", level=2)
    add_paragraph(doc, "Analyse character-level statistics of the URL string.")
    feats_lex = [
        ("url_length", "Total character length of the full URL"),
        ("hostname_length", "Length of the hostname portion"),
        ("path_length", "Length of the URL path"),
        ("query_length", "Length of query string"),
        ("digit_count", "Count of digit characters in URL"),
        ("digit_ratio", "Ratio of digits to total URL length"),
        ("dot_count", "Number of dots in the URL"),
        ("hyphen_count", "Number of hyphens in the URL"),
        ("underscore_count", "Number of underscores in the URL"),
        ("slash_count", "Number of forward slashes"),
        ("question_count", "Number of '?' characters"),
        ("equals_count", "Number of '=' characters (query params)"),
        ("at_count", "Number of '@' characters (credential injection)"),
        ("percent_count", "Number of '%' characters (URL encoding)"),
        ("ampersand_count", "Number of '&' characters"),
        ("special_char_ratio", "Ratio of special chars to total length"),
        ("url_entropy", "Shannon entropy of the full URL string"),
        ("hostname_entropy", "Shannon entropy of hostname only"),
        ("path_entropy", "Shannon entropy of path only"),
        ("longest_hostname_token", "Length of longest token when hostname is split by '.'"),
        ("tld_length", "Length of the top-level domain"),
        ("hostname_digit_count", "Count of digits in hostname"),
        ("hostname_digit_ratio", "Ratio of digits in hostname"),
        ("vowel_consonant_ratio", "Ratio of vowels to consonants (DGA detection)"),
    ]
    add_table(doc, ["Feature Name", "Description"], feats_lex, col_widths=[6, 11])

    doc.add_paragraph("")
    add_heading(doc, "5.2  Structural Features (15 features) — src/features/structural.py", level=2)
    add_paragraph(doc, "Analyse the URL hierarchy, protocol, and infrastructure signals.")
    feats_struct = [
        ("is_https", "1.0 if protocol is HTTPS, else 0.0"),
        ("is_http", "1.0 if protocol is HTTP (insecure)"),
        ("has_ip", "1.0 if hostname is a raw IPv4 address"),
        ("has_hex_ip", "1.0 if IP is hex-encoded"),
        ("num_subdomains", "Count of subdomain levels"),
        ("subdomain_length", "Total character length of all subdomains"),
        ("has_excessive_subdomains", "1.0 if subdomain depth >= 3"),
        ("has_port", "1.0 if a non-standard port is in the URL"),
        ("path_depth", "Depth of the URL path (number of '/')"),
        ("query_param_count", "Number of query parameters"),
        ("suspicious_keyword_count", "Count of keywords: login, verify, banking, secure, update, confirm"),
        ("has_double_slash_redirect", "1.0 if '//' appears in path (redirect marker)"),
        ("has_at_symbol", "1.0 if '@' in URL (credential prefix attack)"),
        ("longest_path_token", "Length of longest token in path"),
        ("tld_in_subdomain", "1.0 if TLD string appears as subdomain token"),
    ]
    add_table(doc, ["Feature Name", "Description"], feats_struct, col_widths=[6, 11])

    doc.add_paragraph("")
    add_heading(doc, "5.3  Regional Features (8 features) — src/features/regional.py", level=2)
    add_paragraph(doc,
        "These are the novel contribution of this project. They detect India-specific phishing "
        "patterns that generic models miss entirely."
    )
    feats_reg = [
        ("is_indian_tld", "1.0 if TLD is .in, .co.in, .gov.in, .nic.in, .bharat, etc."),
        ("has_indian_brand_token", "1.0 if a known Indian brand appears in the URL (SBI, HDFC, Paytm, PhonePe, IRCTC, Aadhaar...)"),
        ("brand_token_count", "Count of distinct matched Indian brand entities"),
        ("has_indian_lure", "1.0 if URL contains regional urgency keywords (kyc, yono, subsidy, cashback, biometric, pan-update...)"),
        ("lure_token_count", "Count of distinct matched lure keywords"),
        ("brand_tld_misalignment", "1.0 if a brand token is detected BUT registered domain is NOT the official portal (KEY PHISHING SIGNAL)"),
        ("max_brand_similarity", "Fuzzy similarity score to nearest Indian brand (detects typosquatting)"),
        ("has_homoglyph_or_punycode", "1.0 if Cyrillic lookalike characters or punycode (xn--) detected in hostname"),
    ]
    add_table(doc, ["Feature Name", "Description"], feats_reg, col_widths=[6, 11])

    doc.add_paragraph("")
    add_paragraph(doc,
        "Indian brands covered (48 entries): SBI, HDFC, ICICI, Axis Bank, PNB, Bank of Baroda, "
        "Canara Bank, Kotak, Yes Bank, IndusInd, Union Bank, RBI, Paytm, PhonePe, Google Pay, "
        "BharatPe, BHIM, UPI, MobiKwik, FreeCharge, CRED, Aadhaar, UIDAI, PAN, Income Tax, "
        "NSDL, UTIITSL, EPFO, IRCTC, India Post, DigiLocker, Parivahan, Voter ID, eChallan, "
        "Jio, Airtel, BSNL, FASTag."
    )

    doc.add_page_break()

    # ── 6. MODELS ─────────────────────────────────────────────────────────────
    add_heading(doc, "6. Machine Learning Models", level=1)

    add_heading(doc, "6.1  LightGBM (Primary Production Model)", level=2)
    add_bullet(doc, "Algorithm: Gradient Boosted Decision Trees (GBDT)")
    add_bullet(doc, "Library: LightGBM (Microsoft)")
    add_bullet(doc, "Input: 47-feature numeric vector")
    add_bullet(doc, "Output: Probability P(Phishing | URL) between 0.0 and 1.0")
    add_bullet(doc, "Threshold: Automatically calibrated on validation set to maximise F1-score (≈0.45)")
    add_bullet(doc, "Model Size: 0.50 MB (serialized with joblib)")
    add_bullet(doc, "Inference Latency: ~2.7 ms per URL (single-sample)")
    add_bullet(doc, "Saved to: backend/model_region_aware.joblib")
    add_bullet(doc, "Code: src/models/baseline.py → class PhishingClassifier")

    add_heading(doc, "6.2  Random Forest (Comparison Model)", level=2)
    add_bullet(doc, "Algorithm: Ensemble of decision trees with bagging")
    add_bullet(doc, "Model Size: 19.33 MB (much larger than LightGBM)")
    add_bullet(doc, "Inference Latency: ~40 ms per URL")
    add_bullet(doc, "Used for: Comparison in E01, E02, E05 experiments")
    add_bullet(doc, "Finding: LightGBM outperforms RF on all metrics while being 38x smaller")

    add_heading(doc, "6.3  Logistic Regression (Comparison Model)", level=2)
    add_bullet(doc, "Algorithm: Linear classifier with sigmoid output")
    add_bullet(doc, "Model Size: 0.001 MB (tiny)")
    add_bullet(doc, "Inference Latency: 0.30 ms (fastest)")
    add_bullet(doc, "Finding: F1 is 8–15% lower than LightGBM — linear model cannot capture URL complexity")

    add_heading(doc, "6.4  BiLSTM Neural Network (Deep Learning Comparison)", level=2)
    add_bullet(doc, "Algorithm: Bidirectional Long Short-Term Memory neural network")
    add_bullet(doc, "Input: Raw URL characters tokenized to ASCII indices (max 150 chars)")
    add_bullet(doc, "Architecture: Embedding(97, 32) → BiLSTM(64) → Dropout(0.3) → Dense(32) → Dense(1)")
    add_bullet(doc, "Library: TensorFlow 2.16 / Keras 3")
    add_bullet(doc, "Model Size: 0.22 MB")
    add_bullet(doc, "Inference Latency: 54–55 ms (30x slower than LightGBM)")
    add_bullet(doc, "Used for: E04 cross-dataset comparison only")
    add_bullet(doc, "Finding: BiLSTM transfers slightly better across datasets but is too slow for production")
    add_bullet(doc, "Code: src/models/bilstm.py → class BiLSTMClassifier")

    doc.add_page_break()

    # ── 7. EXPERIMENTAL RESULTS & MODEL COMPARISON ─────────────────────────────
    add_heading(doc, "7. Experimental Results & Model Comparison", level=1)
    add_paragraph(doc,
        "To determine the most effective architecture for real-time phishing URL detection, "
        "four distinct models (LightGBM, Random Forest, Logistic Regression, and BiLSTM) "
        "were evaluated across standard global benchmarks, regional distribution shifts, and "
        "adversarial stress tests using 5-fold cross-validation (Seeds: 42, 43, 44, 45, 46) "
        "with strict registered-domain isolation."
    )
    doc.add_paragraph("")

    add_heading(doc, "7.1  Master 4-Model Benchmark Comparison", level=2)
    add_paragraph(doc,
        "The following consolidated table compares the performance, efficiency, and resource "
        "consumption of all four candidate models:"
    )

    model_metrics = [
        ["LightGBM (Region-Aware)", "47 (Full Region)", "99.96%", "98.09%", "99.02%", "0.9996", "2.73 ms", "0.50 MB"],
        ["LightGBM (Baseline)", "39 (Conventional)", "91.17%", "78.32%", "84.22%", "0.9283", "2.64 ms", "0.50 MB"],
        ["Random Forest", "39 (Conventional)", "89.48%", "77.08%", "82.81%", "0.9044", "40.88 ms", "19.33 MB"],
        ["Logistic Regression", "39 (Conventional)", "75.33%", "75.80%", "75.56%", "0.8615", "0.28 ms", "0.001 MB"],
        ["BiLSTM (Deep Learning)", "Character Seq.", "98.02%", "47.75%", "64.19%", "0.7766", "54.67 ms", "0.22 MB"],
    ]
    add_table(doc,
        ["Model Architecture", "Feature Set", "Precision", "Recall", "F1-Score", "ROC-AUC", "Latency", "Size"],
        model_metrics,
        col_widths=[3.2, 2.6, 1.7, 1.7, 1.7, 1.8, 1.8, 1.5]
    )

    doc.add_paragraph("")
    add_heading(doc, "7.2  Summary of Key Findings Across Experiments (E01 - E09)", level=2)
    add_paragraph(doc,
        "The models were subjected to nine targeted experimental evaluations to stress-test "
        "generalization across distribution shifts:"
    )

    exp_summary_rows = [
        ["E01", "Global Benchmark", "LightGBM achieved 84.22% F1 on PhreshPhish. Removing path/query caused only a 0.35% drop, proving reliance on hostname structure rather than crawl artifacts."],
        ["E02", "Regional Transfer Gap", "Global models suffered a 7.84% False Negative Rate on Indian phishing lures. Integrating root-domain balancing eliminated structural bias."],
        ["E03", "Feature Ablation", "Adding 8 region-aware features reduced False Negative Rate from 5.52% to 1.91% (a 65% relative reduction in missed attacks), reaching 99.02% F1."],
        ["E04", "Cross-Dataset Shift", "Tested domain adaptation across PhreshPhish and UCI PhiUSIIL corpora. BiLSTM transferred slightly better on character sequences (67.04% F1) but ran 30x slower."],
        ["E05", "Temporal Drift", "Evaluated models trained on early 2024 URLs against late 2025 emerging domains; LightGBM maintained 87.92% F1, proving temporal stability."],
        ["E06", "Adversarial Robustness", "Adversarially retrained LightGBM achieved 99.97% F1 and 0.00% False Negative Rate against homoglyph, subdomain injection, and padding attacks."],
        ["E07", "Unseen Brand Generalization", "Evaluated on 12 completely withheld Indian brands (N=600); achieved 99.90% F1, proving the model learns systemic threat signals rather than memorizing brand names."],
        ["E08", "Emerging AI Phishing", "Zero-shot evaluation on synthetic lures mimicking 2024-2026 government initiatives (PM Surya Ghar, Digi Yatra, etc.) achieved 100% detection with zero false positives."],
        ["E09", "Compound Multi-Shift Test", "Under simultaneous geographic, temporal, and adversarial stress, Region-Aware LightGBM achieved 99.19% F1, outperforming conventional models."],
    ]
    add_table(doc,
        ["Exp", "Evaluation Task", "Key Experimental Takeaway"],
        exp_summary_rows,
        col_widths=[1.5, 3.8, 10.7]
    )

    doc.add_paragraph("")
    add_heading(doc, "7.3  Final Best Model Selection", level=2)
    add_paragraph(doc,
        "WINNER: LightGBM with 47 Region-Aware Features (backend/model_region_aware.joblib)",
        bold=True
    )
    add_paragraph(doc,
        "Based on the empirical evidence across all 9 experiments, LightGBM (Region-Aware) was selected "
        "as the final production model for the following definitive reasons:"
    )
    add_bullet(doc, "Top Predictive Accuracy: Highest overall F1-score (99.02%) and ROC-AUC (0.9996) on regional phishing attacks, slashing missed attacks by 65% compared to the baseline.")
    add_bullet(doc, "Ultra-Low Real-Time Latency: Achieves ~2.7 ms single-sample inference time, easily satisfying the strict real-time response requirement (<25 ms) needed for browser extension link interception.")
    add_bullet(doc, "Minimal Memory Footprint: Serialized model bundle is only 0.50 MB, making it 38x smaller than Random Forest (19.33 MB) and highly portable for client-side or edge deployment.")
    add_bullet(doc, "Adversarial & Temporal Resilience: Maintains >99.1% F1 against active evasion attempts (homoglyphs, subdomain injection) and generalises across chronological shifts without retraining.")
    add_bullet(doc, "Zero-Shot Brand Protection: Proved capable of identifying attacks targeting brands completely absent from training data (99.90% F1), confirming robust structural understanding.")

    doc.add_page_break()

    # ── 8. BACKEND API ────────────────────────────────────────────────────────
    add_heading(doc, "8. Backend REST API", level=1)
    add_paragraph(doc,
        "The backend is a FastAPI application that serves the trained model over HTTP. "
        "It loads the model once on startup and keeps it in memory for fast inference."
    )
    add_paragraph(doc, "File: backend/main.py", italic=True)
    add_paragraph(doc, "Start command:", bold=True)
    p = doc.add_paragraph()
    p.add_run("python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000").font.name = "Courier New"

    doc.add_paragraph("")
    add_heading(doc, "API Endpoints:", level=2)
    add_table(doc,
        ["Endpoint", "Method", "Description"],
        [
            ["GET /health", "GET", "Returns server status, model name, and calibrated threshold"],
            ["POST /predict", "POST", "Accepts {url, include_regional} → returns verdict, risk_score, confidence, latency_ms, reasons, key_signals"],
            ["POST /explain", "POST", "Same as /predict but focused on reasons and feature signals for explainability"],
        ],
        col_widths=[4, 2, 11]
    )

    doc.add_paragraph("")
    add_heading(doc, "Special Logic in the API:", level=2)
    add_bullet(doc, "Official Domain Safelist: URLs matching OFFICIAL_DOMAINS registry or ending in .gov.in/.nic.in are capped at 2% risk (safelisted as legitimate)")
    add_bullet(doc, "Homoglyph Override: If has_homoglyph_or_punycode=1.0 and not safelisted, risk is floored at 98% regardless of model score")
    add_bullet(doc, "CORS enabled for localhost (React dashboard) and chrome-extension:// origins")

    doc.add_page_break()

    # ── 9. FRONTEND ───────────────────────────────────────────────────────────
    add_heading(doc, "9. React Web Dashboard (Frontend)", level=1)
    add_paragraph(doc, "File: web/src/App.jsx  |  Stack: React 19 + Vite + Tailwind CSS + Lucide Icons", italic=True)
    add_paragraph(doc,
        "The web dashboard provides a graphical interface for testing URLs. "
        "It connects to the backend API (localhost:8000) and displays results visually."
    )
    add_heading(doc, "How to start:", level=2)
    p = doc.add_paragraph()
    p.add_run("cmd /c \"cd web && npm run dev\"").font.name = "Courier New"
    add_paragraph(doc, "Then open: http://localhost:5173 in any browser", italic=True)

    add_heading(doc, "Features of the dashboard:", level=2)
    add_bullet(doc, "URL input box — paste any URL and click 'Analyze URL'")
    add_bullet(doc, "Verdict banner — PHISHING DETECTED (red) or VERIFIED LEGITIMATE (green)")
    add_bullet(doc, "Threat probability percentage (e.g. 99.86%)")
    add_bullet(doc, "Inference latency, model confidence, protocol displayed")
    add_bullet(doc, "Plain-English reason cards explaining WHY it is phishing")
    add_bullet(doc, "Feature signals grid — brand lure, TLD misalignment, Shannon entropy, subdomains")
    add_bullet(doc, "5 Quick Test buttons — SBI KYC scam, Official SBI, Paytm scam, IRCTC, Aadhaar link scam")
    add_bullet(doc, "Ablation benchmark table showing E1/E2/E3 results")
    add_bullet(doc, "Backend health indicator (green dot = API online)")

    # ── 10. CHROME EXTENSION ──────────────────────────────────────────────────
    add_heading(doc, "10. Chrome Browser Extension", level=1)
    add_paragraph(doc, "Location: extension/  |  Standard: Chrome Manifest V3", italic=True)
    add_paragraph(doc,
        "The Chrome extension provides real-time phishing protection while browsing. "
        "It automatically scans links on any webpage and adds visual risk badges."
    )
    add_heading(doc, "Files:", level=2)
    add_table(doc,
        ["File", "Purpose"],
        [
            ["manifest.json", "Chrome MV3 specification — declares permissions and content scripts"],
            ["background.js", "Service worker — manages badge counts and API communication lifecycle"],
            ["content.js", "Injected into every webpage — scans anchor tags, adds green/red shield badges, intercepts clicks on high-risk links with a warning modal"],
            ["popup.html + popup.js", "Extension popup — scan the currently active tab URL, show risk score and reasons"],
        ],
        col_widths=[4, 13]
    )
    add_heading(doc, "How to install (for demo):", level=2)
    add_bullet(doc, "Open Chrome → Settings → Extensions → Enable Developer Mode")
    add_bullet(doc, "Click 'Load unpacked' → select the extension/ folder")
    add_bullet(doc, "Extension appears in toolbar — click the icon to inspect any URL")

    doc.add_page_break()

    # ── 11. HOW TO RUN ────────────────────────────────────────────────────────
    add_heading(doc, "11. How to Run the Project", level=1)

    add_heading(doc, "Option A — Terminal (No browser, fastest)", level=2)
    add_paragraph(doc, "Single URL test:", bold=True)
    p = doc.add_paragraph()
    p.add_run("python scripts/demo_offline.py --url \"http://sbi-kyc-verify.xyz/login\"").font.name = "Courier New"
    add_paragraph(doc, "Interactive mode (type any URL):", bold=True)
    p = doc.add_paragraph()
    p.add_run("python scripts/demo_offline.py --interactive").font.name = "Courier New"
    add_paragraph(doc, "Showcase (10 pre-loaded test cases):", bold=True)
    p = doc.add_paragraph()
    p.add_run("python scripts/demo_offline.py").font.name = "Courier New"

    add_heading(doc, "Option B — Web Dashboard (Visual, recommended for demos)", level=2)
    add_paragraph(doc, "Step 1 — Start the backend (keep this terminal open):", bold=True)
    p = doc.add_paragraph()
    p.add_run("python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000").font.name = "Courier New"
    add_paragraph(doc, "Step 2 — Start the frontend (in a new terminal):", bold=True)
    p = doc.add_paragraph()
    p.add_run("cmd /c \"cd web && npm run dev\"").font.name = "Courier New"
    add_paragraph(doc, "Step 3 — Open browser:", bold=True)
    p = doc.add_paragraph()
    p.add_run("http://localhost:5173").font.name = "Courier New"

    add_heading(doc, "Option C — Run all automated tests", level=2)
    p = doc.add_paragraph()
    p.add_run("python -m pytest tests/ -v").font.name = "Courier New"
    add_paragraph(doc, "Expected: 12 passed, 1 warning")

    add_heading(doc, "Prerequisites:", level=2)
    add_bullet(doc, "Python 3.12 installed")
    add_bullet(doc, "All Python dependencies: pip install -r requirements.txt")
    add_bullet(doc, "Node.js installed (for frontend only)")
    add_bullet(doc, "Frontend dependencies: cmd /c \"cd web && npm install\"")

    # ── 12. FOLDER STRUCTURE ──────────────────────────────────────────────────
    doc.add_page_break()
    add_heading(doc, "12. Folder Structure", level=1)
    structure = """Phishing/
├── backend/
│   ├── main.py                    ← FastAPI REST server
│   ├── model_region_aware.joblib  ← Production LightGBM (47 features)
│   └── model_baseline.joblib      ← Baseline LightGBM (39 features)
│
├── data/
│   ├── raw/                       ← Source CSV datasets
│   │   ├── phreshphish_raw.csv    ← 47,703 phishing URLs
│   │   ├── tranco_global_legit.csv
│   │   ├── tranco_indian_legit.csv
│   │   ├── indian_phishing_raw.csv
│   │   ├── phiusiil_raw.csv
│   │   └── ai_generated_urls.csv
│   └── processed/                 ← Train/test splits (domain-isolated)
│       ├── global_train.csv
│       ├── global_test.csv
│       ├── indian_train.csv
│       └── indian_test.csv
│
├── experiments/                   ← 9 experiment scripts (E01–E09)
├── extension/                     ← Chrome MV3 browser extension
├── results/
│   └── SUMMARY.md                 ← All experiment results in one table
├── scripts/
│   ├── demo_offline.py            ← Offline URL inspection CLI
│   ├── download_data.py           ← Dataset download pipeline
│   └── generate_ai_urls.py        ← Synthetic phishing URL generator
├── src/
│   ├── features/
│   │   ├── extractor.py           ← Main feature extraction entry point
│   │   ├── lexical.py             ← 24 lexical features
│   │   ├── structural.py          ← 15 structural features
│   │   └── regional.py            ← 8 region-aware Indian features
│   ├── models/
│   │   ├── baseline.py            ← LightGBM/RF/LR wrapper
│   │   └── bilstm.py              ← BiLSTM neural network
│   ├── explainer/
│   │   └── explainer.py           ← Plain-English reason generator
│   └── utils/
│       ├── data_loader.py         ← Dataset loading and caching
│       ├── domain.py              ← URL parsing utilities
│       └── metrics.py             ← Latency and evaluation metrics
├── tests/
│   ├── test_phase1.py             ← Feature + model + API tests
│   ├── test_phase2.py             ← Regional + explainer tests
│   └── test_phase3.py             ← BiLSTM + adversarial + latency tests
├── web/
│   └── src/
│       ├── App.jsx                ← Main React dashboard component
│       └── App.css                ← Dashboard styling
├── README.md
└── requirements.txt"""
    p = doc.add_paragraph()
    p.add_run(structure).font.name = "Courier New"
    p.runs[0].font.size = Pt(8)

    # ── 13. TECH STACK ────────────────────────────────────────────────────────
    doc.add_page_break()
    add_heading(doc, "13. Technology Stack", level=1)
    add_table(doc,
        ["Layer", "Technology", "Purpose"],
        [
            ["Core Language", "Python 3.12", "All ML, API, and data processing"],
            ["Machine Learning", "LightGBM", "Primary gradient-boosted classification model"],
            ["ML Framework", "Scikit-learn", "Logistic Regression, Random Forest, train/test split, metrics"],
            ["Deep Learning", "TensorFlow 2.16 / Keras 3", "BiLSTM character-level neural network"],
            ["Data Processing", "Pandas + NumPy", "DataFrame manipulation, array operations"],
            ["URL Parsing", "tldextract + urllib", "Reliable registered domain and TLD extraction using public suffix rules"],
            ["Data Storage", "PyArrow (Parquet)", "Fast feature cache storage and retrieval"],
            ["REST API", "FastAPI + Uvicorn", "Async HTTP server with automatic Swagger docs"],
            ["API Validation", "Pydantic", "Request/response schema validation"],
            ["Frontend", "React 19 + Vite", "Web dashboard UI"],
            ["CSS Framework", "Tailwind CSS", "Responsive dark cybersecurity design"],
            ["Icons", "Lucide React", "Shield, alert, and metric icons"],
            ["Browser Extension", "Chrome MV3 + Vanilla JS", "Real-time link scanning and intercept modal"],
            ["Testing", "Pytest", "12 automated tests across 3 phases"],
            ["Model Persistence", "Joblib", "Serialization of LightGBM model + scaler + threshold"],
        ],
        col_widths=[4, 5, 8]
    )

    doc.add_paragraph("")
    doc.add_paragraph("")
    end = doc.add_paragraph("— End of Project Documentation —")
    end.alignment = WD_ALIGN_PARAGRAPH.CENTER
    end.runs[0].bold = True

    out_path = "docs/Project_Documentation.docx"
    doc.save(out_path)
    print(f"\n[OK] Saved: {out_path}")
    print(f"     Pages:  ~30 pages")
    print(f"     Open this file in Microsoft Word or Google Docs.\n")


if __name__ == "__main__":
    generate()
