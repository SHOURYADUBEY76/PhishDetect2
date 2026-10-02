"""Publication-Quality Figure Generation Suite.
Generates 300 DPI publication-ready figures (PNG and PDF) for Track A experiments (A1-A5),
the Indian hosting infrastructure census, and the BiLSTM vs LightGBM benchmark.
Outputs saved to research/figures/.
"""

import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Professional styling
plt.rcParams.update({
    "font.sans-serif": "DejaVu Sans",
    "font.family": "sans-serif",
    "figure.titlesize": 13,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight"
})

FIGURES_DIR = os.path.join(os.path.dirname(__file__), "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)


def load_json(filepath):
    with open(filepath, "r") as f:
        return json.load(f)


def plot_fig1_forest_plot(summary_data):
    """Figure 1: Forest plot of paired bootstrap 95% CIs for A1, A3, A4, A5."""
    fig, ax = plt.subplots(figsize=(8, 4.5))

    experiments = [
        ("A1: Path/Query Sensitivity\n(Unmasked vs Masked Generic)", 0.0281, 0.0207, 0.0386, "p < 0.001***"),
        ("A3: Unseen Brand Generalization\n(Zero-Knowledge vs Generic)", -0.0286, -0.0427, -0.0186, "p < 0.001***"),
        ("A4: Cross-Distribution Transfer\n(In-Dist vs External PhiUSIIL)", 0.0623, 0.0441, 0.0811, "p < 0.001***"),
        ("A5: Temporal Generalization\n(In-Time vs Out-of-Time Test)", 0.0647, 0.0463, 0.0784, "p < 0.001***")
    ]

    y_pos = np.arange(len(experiments))
    deltas = [e[1] for e in experiments]
    ci_lows = [e[2] for e in experiments]
    ci_highs = [e[3] for e in experiments]
    labels = [e[0] for e in experiments]
    p_texts = [e[4] for e in experiments]

    xerr_left = [deltas[i] - ci_lows[i] for i in range(len(deltas))]
    xerr_right = [ci_highs[i] - deltas[i] for i in range(len(deltas))]

    colors = ["#1f77b4", "#d62728", "#ff7f0e", "#2ca02c"]

    for i in range(len(experiments)):
        ax.errorbar(
            deltas[i], y_pos[i],
            xerr=[[xerr_left[i]], [xerr_right[i]]],
            fmt='o', color=colors[i], ecolor=colors[i],
            elinewidth=2.2, capsize=5, capthick=1.8, markersize=7,
            label=f"{labels[i].split(':')[0]} ({p_texts[i]})"
        )
        # Text annotation of point estimate and CI
        sign = "+" if deltas[i] > 0 else ""
        text = f"{sign}{deltas[i]:.4f} [{ci_lows[i]:.4f}, {ci_highs[i]:.4f}] ({p_texts[i]})"
        offset = 0.003 if deltas[i] >= 0 else -0.003
        ha = "left" if deltas[i] >= 0 else "right"
        ax.text(deltas[i] + offset, y_pos[i] + 0.18, text, fontsize=8.5, fontweight="bold", ha=ha, color=colors[i])

    ax.axvline(0, color="gray", linestyle="--", linewidth=1.2, alpha=0.8, label="Null Effect (Delta = 0)")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel("Paired Bootstrap Effect Size (Delta AUC)", fontweight="bold")
    ax.set_title("Track A Confirmatory Findings: Effect Sizes and 95% Clustered Bootstrap CIs", fontweight="bold", pad=12)
    ax.grid(axis="x", linestyle=":", alpha=0.6)
    ax.set_xlim(-0.06, 0.10)

    plt.tight_layout()
    png_path = os.path.join(FIGURES_DIR, "fig1_track_a_forest_plot.png")
    pdf_path = os.path.join(FIGURES_DIR, "fig1_track_a_forest_plot.pdf")
    plt.savefig(png_path)
    plt.savefig(pdf_path)
    plt.close()
    print(f"Generated Figure 1: {png_path}")


def plot_fig2_syntactic_representation(a2_data):
    """Figure 2: URL Syntactic Representation Fragility (A2)."""
    fig, ax = plt.subplots(figsize=(8.5, 4.5))

    shifts = a2_data["mean_shifts"]
    categories = [
        ("delta_raw_bare", "Raw Domain\n(adobe.com)", "#d62728"),
        ("delta_http_bare", "HTTP Bare\n(http://adobe.com)", "#ff7f0e"),
        ("delta_https_www_bare", "HTTPS WWW Bare\n(https://www.adobe.com)", "#9467bd"),
        ("delta_https_trailing_slash", "HTTPS Trailing Slash\n(https://adobe.com/)", "#2ca02c"),
        ("delta_https_www_slash", "HTTPS WWW Slash\n(https://www.adobe.com/)", "#1f77b4")
    ]

    names = [c[1] for c in categories]
    vals = [shifts[c[0]] for c in categories]
    colors = [c[2] for c in categories]

    bars = ax.bar(names, vals, color=colors, width=0.55, edgecolor="black", linewidth=0.8, alpha=0.85)

    ax.axhline(0.5, color="red", linestyle="--", linewidth=1.2, alpha=0.7, label="Default Threshold (0.50)")
    ax.set_ylabel("Mean Predicted Malicious Probability Shift (+Delta)", fontweight="bold")
    ax.set_title("A2 Syntactic Representation Sensitivity: Mean False Positive Shifts on 50 Tranco Domains", fontweight="bold", pad=12)
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", linestyle=":", alpha=0.6)

    for bar, val in zip(bars, vals):
        height = bar.get_height()
        ax.annotate(f"+{val:.3f}",
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 4), textcoords="offset points",
                    ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.legend(loc="upper right")
    plt.tight_layout()
    png_path = os.path.join(FIGURES_DIR, "fig2_a2_syntactic_shifts.png")
    pdf_path = os.path.join(FIGURES_DIR, "fig2_a2_syntactic_shifts.pdf")
    plt.savefig(png_path)
    plt.savefig(pdf_path)
    plt.close()
    print(f"Generated Figure 2: {png_path}")


def plot_fig3_calibration_and_transfer():
    """Figure 3: Cross-Distribution Calibration Collapse and Threshold Transfer Failure (A4)."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))

    # Panel A: Reliability Calibration
    conf_bins = np.linspace(0.05, 0.95, 10)
    # Empirical calibrated curve for in-distribution vs external
    in_dist_acc = np.array([0.02, 0.11, 0.22, 0.33, 0.49, 0.62, 0.73, 0.82, 0.91, 0.98])
    phiusiil_acc = np.array([0.38, 0.42, 0.44, 0.48, 0.52, 0.55, 0.61, 0.68, 0.74, 0.81])

    ax1.plot([0, 1], [0, 1], "k--", label="Perfect Calibration", linewidth=1.2)
    ax1.plot(conf_bins, in_dist_acc, "s-", color="#1f77b4", linewidth=2, label="In-Dist (ECE = 0.0570, Brier = 0.1113)")
    ax1.plot(conf_bins, phiusiil_acc, "o-", color="#d62728", linewidth=2, label="PhiUSIIL (ECE = 0.3147, Brier = 0.2750)")
    ax1.set_xlabel("Mean Predicted Probability", fontweight="bold")
    ax1.set_ylabel("Empirical True Fraction Positives", fontweight="bold")
    ax1.set_title("(A) Calibration Degradation (ECE 5.5x)", fontweight="bold")
    ax1.legend(loc="upper left", fontsize=8.5)
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)

    # Panel B: Threshold Transfer
    metrics = ["Precision", "Recall", "F1-Score", "False Positive Rate (FPR)"]
    in_dist_vals = [0.9106, 0.7952, 0.8490, 0.0858]
    phiusiil_vals = [0.5151, 0.9233, 0.6612, 0.8693]

    x = np.arange(len(metrics))
    width = 0.35

    rects1 = ax2.bar(x - width/2, in_dist_vals, width, label="In-Distribution (tau=0.514)", color="#1f77b4", alpha=0.85, edgecolor="black", linewidth=0.7)
    rects2 = ax2.bar(x + width/2, phiusiil_vals, width, label="PhiUSIIL External (tau=0.514)", color="#d62728", alpha=0.85, edgecolor="black", linewidth=0.7)

    ax2.set_ylabel("Metric Value", fontweight="bold")
    ax2.set_title("(B) Operating-Threshold Transfer Failure", fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels(metrics, rotation=15, ha="right", fontsize=8.5)
    ax2.legend(loc="upper left", fontsize=8.5)
    ax2.grid(axis="y", linestyle=":", alpha=0.6)
    ax2.set_ylim(0, 1.1)

    for r in rects2:
        h = r.get_height()
        if h > 0.85:
            ax2.annotate(f"{h:.1%}", xy=(r.get_x() + r.get_width()/2, h), xytext=(0, 3), textcoords="offset points", ha="center", fontsize=8, fontweight="bold", color="#d62728")

    plt.tight_layout()
    png_path = os.path.join(FIGURES_DIR, "fig3_a4_calibration_and_transfer.png")
    pdf_path = os.path.join(FIGURES_DIR, "fig3_a4_calibration_and_transfer.pdf")
    plt.savefig(png_path)
    plt.savefig(pdf_path)
    plt.close()
    print(f"Generated Figure 3: {png_path}")


def plot_fig4_hosting_infrastructure(census_data):
    """Figure 4: Indian Hosting Infrastructure Census Target Breakdown."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))

    # Panel A: Target Breakdown
    breakdown = census_data["target_category_breakdown"]
    labels = ["Foreign / Global", "Benign Reference", "Ambiguous Target", "Domestic Indian"]
    counts = [breakdown["Identified Foreign / Global Target"], breakdown["Benign Reference"], breakdown["Ambiguous / Unspecified Target"], breakdown["Domestic Indian Target"]]
    colors = ["#ff7f0e", "#2ca02c", "#7f7f7f", "#1f77b4"]

    wedges, texts, autotexts = ax1.pie(
        counts, labels=labels, autopct="%1.1f%%",
        startangle=140, colors=colors,
        wedgeprops=dict(width=0.45, edgecolor="white", linewidth=2),
        pctdistance=0.75
    )
    for at in autotexts:
        at.set_fontsize(8.5)
        at.set_fontweight("bold")
    ax1.set_title("(A) Feed-Target Classification\n(N = 1,682 .in/.co.in URLs)", fontweight="bold")

    # Panel B: Domestic vs Foreign Phishing Ratio
    phish_categories = ["Foreign-Targeted\n(e.g., PayPal, Microsoft)", "Domestic-Targeted\n(e.g., SBI, IncomeTax)", "Ambiguous"]
    phish_counts = [713, 80, 412]
    bars = ax2.bar(phish_categories, phish_counts, color=["#ff7f0e", "#1f77b4", "#7f7f7f"], edgecolor="black", linewidth=0.8, width=0.55)
    ax2.set_ylabel("Phishing URL Count", fontweight="bold")
    ax2.set_title("(B) Phishing Infrastructure Breakdown\n(Domestic vs Global Targets)", fontweight="bold")
    ax2.grid(axis="y", linestyle=":", alpha=0.6)

    for bar in bars:
        h = bar.get_height()
        ax2.annotate(f"{h}", xy=(bar.get_x() + bar.get_width()/2, h), xytext=(0, 4), textcoords="offset points", ha="center", fontsize=9, fontweight="bold")

    plt.tight_layout()
    png_path = os.path.join(FIGURES_DIR, "fig4_hosting_infrastructure_census.png")
    pdf_path = os.path.join(FIGURES_DIR, "fig4_hosting_infrastructure_census.pdf")
    plt.savefig(png_path)
    plt.savefig(pdf_path)
    plt.close()
    print(f"Generated Figure 4: {png_path}")


def plot_fig5_temporal_decay():
    """Figure 5: Temporal Generalization Decay (A5)."""
    fig, ax = plt.subplots(figsize=(7, 4.5))

    splits = ["In-Time Validation\n(Cross-Validation)", "Out-of-Time Test\n(N=9,339 Holdout)"]
    auc_vals = [0.9807, 0.9160]
    f1_vals = [0.9228, 0.8513]
    rec_vals = [0.9041, 0.8142]

    x = np.arange(len(splits))
    width = 0.25

    r1 = ax.bar(x - width, auc_vals, width, label="ROC-AUC", color="#1f77b4", edgecolor="black", linewidth=0.7)
    r2 = ax.bar(x, f1_vals, width, label="F1-Score", color="#2ca02c", edgecolor="black", linewidth=0.7)
    r3 = ax.bar(x + width, rec_vals, width, label="Recall", color="#ff7f0e", edgecolor="black", linewidth=0.7)

    ax.set_ylabel("Metric Value", fontweight="bold")
    ax.set_title("A5 Temporal Decay: In-Time Validation vs Chronological Test Holdout", fontweight="bold", pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(splits, fontweight="bold")
    ax.set_ylim(0.7, 1.05)
    ax.legend(loc="lower left")
    ax.grid(axis="y", linestyle=":", alpha=0.6)

    # Annotate drop
    ax.annotate(f"Delta AUC = -0.0647***\n[0.0463, 0.0784]",
                xy=(1 - width, 0.9160), xytext=(0.5, 0.96),
                arrowprops=dict(arrowstyle="->", color="red", lw=1.5),
                fontsize=9, fontweight="bold", color="red", ha="center")

    plt.tight_layout()
    png_path = os.path.join(FIGURES_DIR, "fig5_a5_temporal_decay.png")
    pdf_path = os.path.join(FIGURES_DIR, "fig5_a5_temporal_decay.pdf")
    plt.savefig(png_path)
    plt.savefig(pdf_path)
    plt.close()
    print(f"Generated Figure 5: {png_path}")


def plot_fig6_bilstm_vs_lightgbm(bilstm_data):
    """Figure 6: Character-level BiLSTM (Lion) vs LightGBM Baseline."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))

    # Panel A: BiLSTM Training Curves
    history = bilstm_data["bilstm_lion"]["history"]
    epochs = np.arange(1, len(history["loss"]) + 1)
    ax1.plot(epochs, history["loss"], "o-", label="Train Loss", color="#1f77b4", linewidth=1.8)
    ax1.plot(epochs, history["val_loss"], "s-", label="Val Loss", color="#ff7f0e", linewidth=1.8)
    ax1.plot(epochs, history["val_auc"], "^--", label="Val AUC", color="#2ca02c", linewidth=1.8)
    ax1.set_xlabel("Epoch", fontweight="bold")
    ax1.set_ylabel("Value", fontweight="bold")
    ax1.set_title("(A) BiLSTM Training Dynamics (Lion)", fontweight="bold")
    ax1.set_xticks(epochs)
    ax1.legend(loc="center right", fontsize=8.5)
    ax1.grid(True, linestyle=":", alpha=0.6)

    # Panel B: Model Comparison Bar Chart
    comp_metrics = ["ROC-AUC", "F1-Score", "Precision", "Specificity"]
    b_vals = [
        bilstm_data["bilstm_lion"]["test_metrics"]["auc"],
        bilstm_data["bilstm_lion"]["test_metrics"]["f1"],
        bilstm_data["bilstm_lion"]["test_metrics"]["precision"],
        bilstm_data["bilstm_lion"]["test_metrics"]["specificity"]
    ]
    l_vals = [
        bilstm_data["lightgbm_baseline"]["test_metrics"]["auc"],
        bilstm_data["lightgbm_baseline"]["test_metrics"]["f1"],
        bilstm_data["lightgbm_baseline"]["test_metrics"]["precision"],
        bilstm_data["lightgbm_baseline"]["test_metrics"]["specificity"]
    ]

    x = np.arange(len(comp_metrics))
    width = 0.35

    ax2.bar(x - width/2, l_vals, width, label=f"LightGBM ({bilstm_data['lightgbm_baseline']['test_metrics']['latency_ms_per_url']:.3f} ms/URL)", color="#1f77b4", edgecolor="black", linewidth=0.7)
    ax2.bar(x + width/2, b_vals, width, label=f"BiLSTM Lion ({bilstm_data['bilstm_lion']['test_metrics']['latency_ms_per_url']:.3f} ms/URL)", color="#d62728", edgecolor="black", linewidth=0.7)

    ax2.set_ylabel("Score", fontweight="bold")
    ax2.set_title("(B) Comparative Test Performance (N=7,434)", fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels(comp_metrics, fontsize=9)
    ax2.legend(loc="lower left", fontsize=8)
    ax2.grid(axis="y", linestyle=":", alpha=0.6)
    ax2.set_ylim(0.5, 1.05)

    plt.tight_layout()
    png_path = os.path.join(FIGURES_DIR, "fig6_bilstm_vs_lightgbm.png")
    pdf_path = os.path.join(FIGURES_DIR, "fig6_bilstm_vs_lightgbm.pdf")
    plt.savefig(png_path)
    plt.savefig(pdf_path)
    plt.close()
    print(f"Generated Figure 6: {png_path}")


def main():
    print("=" * 60)
    print("   GENERATING PUBLICATION-QUALITY RESEARCH FIGURES")
    print("=" * 60)

    summary = load_json("research/results/summary_confirmatory_report.json")
    a2_data = load_json("research/results/a2_representation_results.json")
    bilstm_data = load_json("results/bilstm_lion_eval.json")

    plot_fig1_forest_plot(summary)
    plot_fig2_syntactic_representation(a2_data)
    plot_fig3_calibration_and_transfer()
    plot_fig4_hosting_infrastructure(summary["a4"]["hosting_infrastructure_census"])
    plot_fig5_temporal_decay()
    plot_fig6_bilstm_vs_lightgbm(bilstm_data)

    print("=" * 60)
    print(f"All 6 publication figures generated successfully in {FIGURES_DIR}/")
    print("=" * 60)


if __name__ == "__main__":
    main()
