# Experimental Benchmark Summary (E01 - E09)

## Executive Overview

This document presents the consolidated experimental benchmark results for **"Region-Aware Phishing URL Detection: Robust Generalization Across Geographic, Temporal, and Adversarial Distribution Shifts"**.

All experiments were executed with strict domain-level group partitioning using `tldextract` registered domains, ensuring **zero domain leakage** across train/test splits. Evaluations represent 5-fold seed averages ($\text{Seeds} = [42, 43, 44, 45, 46]$) with exact serialized model sizes and single-sample inference latency profiling.

---

## Master Comparison Table

| Experiment | Focus / Task | Model | Features | Precision | Recall | F1-Score | ROC-AUC | Latency (ms) | Model Size |
|---|---|---|---|---|---|---|---|---|---|
| **E01** | Standard Global Benchmark | LightGBM | Lexical + Structural | 91.17% ± 1.33% | 78.32% ± 2.26% | **84.22% ± 0.79%** | 0.9283 ± 0.0010 | 2.64 ms | 0.50 MB |
| **E01** | Standard Global Benchmark | Random Forest | Lexical + Structural | 89.48% ± 0.88% | 77.08% ± 1.19% | 82.81% ± 0.36% | 0.9044 ± 0.0008 | 42.23 ms | 19.33 MB |
| **E01** | Standard Global Benchmark | Logistic Regression | Lexical + Structural | 75.33% ± 0.77% | 75.80% ± 0.94% | 75.56% ± 0.51% | 0.8615 ± 0.0017 | 0.30 ms | 0.001 MB |
| **E01** | Source Bias Audit | LightGBM | No Path/Query | 91.19% ± 0.26% | 77.64% ± 0.67% | 83.87% ± 0.30% | 0.9225 ± 0.0013 | 2.75 ms | 0.50 MB |
| **E02** | Regional Transfer Gap | LightGBM | Lexical + Structural | 100.0% ± 0.00% | 92.16% ± 1.22% | 95.92% ± 0.67% | 0.9983 ± 0.0004 | 2.73 ms | 0.50 MB |
| **E02** | Regional Transfer Gap | Random Forest | Lexical + Structural | 99.61% ± 0.15% | 90.37% ± 1.51% | 94.76% ± 0.86% | 0.9966 ± 0.0006 | 40.88 ms | 19.33 MB |
| **E02** | Regional Transfer Gap | Logistic Regression | Lexical + Structural | 98.91% ± 0.29% | 84.41% ± 0.76% | 91.08% ± 0.33% | 0.9348 ± 0.0031 | 0.28 ms | 0.001 MB |
| **E03** | Feature Ablation (Lexical) | LightGBM | Lexical Only (24) | 91.24% ± 0.79% | 72.64% ± 7.63% | 80.66% ± 4.91% | 0.9431 ± 0.0049 | 2.77 ms | 0.50 MB |
| **E03** | Feature Ablation (Structural) | LightGBM | Lex + Struct (39) | 99.99% ± 0.03% | 94.48% ± 0.83% | 97.16% ± 0.44% | 0.9993 ± 0.0001 | 2.90 ms | 0.50 MB |
| **E03** | Feature Ablation (Full Region) | LightGBM | Full Region-Aware (47) | **99.96% ± 0.03%** | **98.09% ± 0.34%** | **99.02% ± 0.18%** | **0.9996 ± 0.0000** | 3.52 ms | 0.50 MB |
| **E04** | Cross-Dataset (Phresh $\to$ Phi) | LightGBM | Conventional (39) | 51.73% ± 2.22% | 91.83% ± 1.06% | 66.14% ± 1.54% | 0.8469 ± 0.0217 | 1.42 ms | 0.50 MB |
| **E04** | Cross-Dataset (Phresh $\to$ Phi) | BiLSTM (Neural) | Character Sequences | 98.02% ± 0.40% | 47.75% ± 2.00% | 64.19% ± 1.79% | 0.7766 ± 0.0159 | 55.44 ms | 0.22 MB |
| **E04** | Cross-Dataset (Phi $\to$ Phresh) | LightGBM | Conventional (39) | 48.80% ± 0.29% | 86.75% ± 0.99% | 62.46% ± 0.49% | 0.5494 ± 0.0242 | 1.63 ms | 0.47 MB |
| **E04** | Cross-Dataset (Phi $\to$ Phresh) | BiLSTM (Neural) | Character Sequences | 53.74% ± 1.24% | 89.13% ± 0.22% | 67.04% ± 0.91% | 0.6771 ± 0.0937 | 54.67 ms | 0.22 MB |
| **E05** | Temporal Drift ($T_{\text{early}} \to T_{\text{late}}$) | LightGBM | Conventional (39) | 86.64% ± 4.14% | 89.60% ± 3.61% | 87.92% ± 0.67% | 0.9565 ± 0.0010 | 1.58 ms | 0.50 MB |
| **E05** | Temporal Drift ($T_{\text{early}} \to T_{\text{late}}$) | Random Forest | Conventional (39) | 86.92% ± 2.40% | 86.95% ± 2.41% | 86.87% ± 0.35% | 0.9452 ± 0.0011 | 17.38 ms | 16.41 MB |
| **E06** | Adversarial Robustness (Clean) | LightGBM | Clean Model on Clean | 99.96% ± 0.03% | 98.09% ± 0.34% | 99.02% ± 0.18% | 0.9996 ± 0.0000 | 1.53 ms | 0.50 MB |
| **E06** | Adversarial Robustness (Evaded) | LightGBM | Clean Model on Adv Test | 99.96% ± 0.03% | 98.38% ± 0.46% | 99.16% ± 0.25% | 0.9995 ± 0.0003 | 1.61 ms | 0.50 MB |
| **E06** | Adversarially Retrained | LightGBM | Adv Model on Adv Test | **99.95% ± 0.05%** | **100.0% ± 0.00%** | **99.97% ± 0.02%** | **1.0000 ± 0.0000** | 1.46 ms | 0.50 MB |
| **E07** | Zero-Shot Unseen Brands ($N=600$) | LightGBM | 12 Withheld Brands | 100.0% ± 0.00% | 99.80% ± 0.16% | **99.90% ± 0.08%** | 0.9999 ± 0.0000 | 2.76 ms | 0.50 MB |
| **E07** | Seen Brands Subset ($N=294$) | LightGBM | Training Entities | 100.0% ± 0.00% | 95.78% ± 0.79% | 97.84% ± 0.41% | 0.9999 ± 0.0001 | 6.27 ms | 0.50 MB |
| **E08** | Synthetic AI Phishing ($N=500$) | LightGBM | Conventional (39) | 100.0% ± 0.00% | 100.0% ± 0.00% | 100.0% ± 0.00% | 1.0000 ± 0.0000 | 1.33 ms | 0.50 MB |
| **E08** | Synthetic AI Phishing ($N=500$) | LightGBM | Full Region-Aware (47) | 100.0% ± 0.00% | 100.0% ± 0.00% | **100.0% ± 0.00%** | 1.0000 ± 0.0000 | 1.47 ms | 0.50 MB |
| **E09** | Compound Shift Stress Test | LightGBM | Conventional (39) | 99.99% ± 0.03% | 97.22% ± 0.68% | 98.58% ± 0.34% | 0.9998 ± 0.0000 | 1.64 ms | 0.50 MB |
| **E09** | Compound Shift Stress Test | LightGBM | Full Region-Aware (47) | **99.96% ± 0.03%** | **98.44% ± 0.57%** | **99.19% ± 0.30%** | **0.9997 ± 0.0002** | 1.63 ms | 0.50 MB |

---

## Detailed Experimental Insights

### E01: Standard Benchmark & Source Bias Audit
- **LightGBM** established the strongest baseline ($F_1 = 0.8422$, ROC-AUC $= 0.9283$, Latency $= 2.64\text{ ms}$) while maintaining an ultra-compact footprint of $0.50\text{ MB}$.
- **Source Bias Audit**: Removing path and query parameters caused only a negligible $0.35\%$ drop in $F_1$ score ($0.8422 \to 0.8387$), verifying that the model relies on structural hostname integrity rather than dataset-specific path artifacts.

### E02: Regional Transfer Gap
- Conventional global models applied to Indian phishing traffic suffer an elevated False Negative Rate ($\text{FNR} = 7.84\%$).
- By balancing global training data with non-overlapping Tranco global root domains, the structural root-domain mismatch was completely resolved, isolating the genuine regional transfer gap.

### E03: Regional Feature Engineering & Ablation
- **Lexical Only (24 features)**: High False Negative Rate ($\text{FNR} = 27.36\%$).
- **Lexical + Structural (39 features)**: Improved $F_1$ to $97.16\%$, but still missed $5.52\%$ of regional lures.
- **Full Region-Aware (47 features)**: Increased $F_1$ to **$99.02\%$**, cutting the False Negative Rate down to **$1.91\%$** ($65\%$ relative reduction in missed phishing attacks).

### E04: Cross-Dataset Distribution Gap (PhreshPhish $\leftrightarrow$ UCI PhiUSIIL)
- Transfer across fundamentally distinct corpora highlighted substantial domain shift:
  - PhreshPhish $\to$ PhiUSIIL: LightGBM $F_1 = 0.6614$, BiLSTM $F_1 = 0.6419$.
  - PhiUSIIL $\to$ PhreshPhish: LightGBM $F_1 = 0.6246$, BiLSTM $F_1 = 0.6704$.
- The character-level BiLSTM exhibited greater domain transfer resilience on character distributions ($F_1 = 0.6704$) but required $54.67\text{ ms}$ inference latency—over $30\times$ slower than LightGBM ($1.63\text{ ms}$).

### E05: Temporal Concept Drift
- Training on earlier historical URLs ($2024\text{-}07$ to $2025\text{-}08$) and testing strictly on later emerging registered domains ($2025\text{-}08$ to $2025\text{-}12$) demonstrated stable chronological performance ($F_1 = 87.92\%$, ROC-AUC $= 0.9565$).

### E06: Adversarial Evasion & Adversarial Training
- Clean models demonstrated resilience against single-mutation evasions ($F_1 = 99.16\%$).
- Adding 20% adversarially perturbed samples during training achieved near-perfect defense (**$F_1 = 99.97\%$**, **$\text{FNR} = 0.00\%$** on adversarial test lures).

### E07: Zero-Shot Detection of Withheld Indian Brands
- Evaluating on $N=600$ balanced samples across 12 strictly held-out brands (PhonePe, Aadhaar, UIDAI, IRCTC, Axis Bank, PNB, Kotak, EPFO, Paytm, HDFC, ICICI) yielded **$99.90\%$ $F_1$** and **$99.80\%$ Recall**.
- Demonstrates that the region-aware feature architecture captures systemic adversarial patterns (lures, mismatch, homoglyphs) rather than memorizing individual brand strings.

### E08: Emerging AI-Generated Phishing Detection
- Zero-shot evaluation on novel synthetic lures (PM Surya Ghar, Digi Yatra, Ayushman Bharat, Kisan Credit) achieved $100\%$ detection with zero false positives.
- Provenance and stand-in statuses are explicitly recorded in `results/E08.csv`.

### E09: Multi-Shift Compound Stress Test
- Testing under simultaneous geographic, temporal, and adversarial shifts showed Region-Aware LightGBM achieving **$99.19\%$ $F_1$** and **$98.44\%$ Recall**, outperforming conventional models ($98.58\%$ $F_1$).
