"""Experiment E6: Adversarial Robustness & Adversarial Training Evaluation.
Evaluates model vulnerability under realistic evasion attacks:
  1. Homoglyph substitution (Cyrillic lookalikes)
  2. Subdomain injection (prefixing target brands as deep subdomains)
  3. Token padding (inserting deceptive security tokens)
Compares Clean Training vs. Adversarially Trained models across 5 seeds.
Outputs results/E06.csv.
"""

import os
import sys
import random
from urllib.parse import urlparse
import pandas as pd
import numpy as np

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models.baseline import PhishingClassifier
from src.utils.data_loader import load_processed_split, split_train_val_grouped, get_feature_matrix


SEEDS = [42, 43, 44, 45, 46]

# Cyrillic lookalikes
HOMOGLYPH_REPLACEMENTS = {
    'a': 'а', 'c': 'с', 'e': 'е', 'o': 'о', 'p': 'р', 's': 'ѕ', 'i': 'і'
}

PADDING_TOKENS = ["secure-login", "auth-verify", "update-portal", "account-center", "portal-gateway"]
DECEPTIVE_TLDS = ["top", "xyz", "site", "cloud", "buzz", "online"]


def perturb_url_adversarial(url: str, technique: str = "random") -> str:
    """Apply realistic evasion perturbations to a static URL string."""
    u_str = str(url).strip()
    parsed = urlparse(u_str if "://" in u_str else f"http://{u_str}")
    hostname = (parsed.netloc or "").split(":")[0]
    path = parsed.path
    query = f"?{parsed.query}" if parsed.query else ""
    scheme = parsed.scheme or "http"

    if technique == "random":
        technique = random.choice(["homoglyph", "token_padding", "subdomain_trick"])

    if technique == "homoglyph":
        # Substitute 1-2 characters in hostname with lookalike
        chars = list(hostname)
        candidates = [i for i, c in enumerate(chars) if c in HOMOGLYPH_REPLACEMENTS]
        if candidates:
            idx = random.choice(candidates)
            chars[idx] = HOMOGLYPH_REPLACEMENTS[chars[idx]]
        new_host = "".join(chars)
        return f"{scheme}://{new_host}{path}{query}"

    elif technique == "token_padding":
        # Inject padding tokens into host or path
        pad = random.choice(PADDING_TOKENS)
        tokens = hostname.split(".")
        if len(tokens) >= 2:
            tokens[0] = f"{tokens[0]}-{pad}"
            new_host = ".".join(tokens)
        else:
            new_host = f"{hostname}-{pad}"
        return f"{scheme}://{new_host}{path}{query}"

    elif technique == "subdomain_trick":
        # Prepend hostname as subdomain of an attacker root
        tld = random.choice(DECEPTIVE_TLDS)
        clean_host = hostname.replace(".", "-")
        new_host = f"{hostname}.auth-service-{random.randint(10, 99)}.{tld}"
        return f"{scheme}://{new_host}{path}{query}"

    return u_str


def generate_adversarial_dataset(df_clean: pd.DataFrame, perturbation_rate: float = 1.0) -> pd.DataFrame:
    """Generate perturbed test set from clean phishing samples."""
    df_adv = df_clean.copy()
    perturbed_urls = []
    for u, label in zip(df_adv["url"], df_adv["label"]):
        if label == 1:
            # Apply adversarial evasion technique
            perturbed_urls.append(perturb_url_adversarial(u))
        else:
            # Keep legitimate URLs unaltered
            perturbed_urls.append(u)
    df_adv["url"] = perturbed_urls
    return df_adv


def run_experiment_e6():
    os.makedirs("results", exist_ok=True)
    print("\n=======================================================", flush=True)
    print("   EXPERIMENT E6: ADVERSARIAL EVASION & TRAINING", flush=True)
    print("=======================================================", flush=True)

    df_itest = load_processed_split("indian_test")
    df_itrain = load_processed_split("indian_train")
    df_gtrain = load_processed_split("global_train")
    df_clean_train = pd.concat([df_gtrain, df_itrain], ignore_index=True)

    # 1. Clean Test Set & Perturbed Test Set
    df_adv_test = generate_adversarial_dataset(df_itest)
    print(f"Generated Adversarial Test Set ({len(df_adv_test)} samples)", flush=True)

    # 2. Adversarially Augmented Training Set (20% adversarial perturbation on training phishing)
    df_adv_augmented_train = df_clean_train.copy()
    adv_train_samples = df_clean_train[df_clean_train["label"] == 1].sample(frac=0.2, random_state=42).copy()
    adv_train_samples["url"] = adv_train_samples["url"].apply(perturb_url_adversarial)
    df_adv_train_full = pd.concat([df_clean_train, adv_train_samples], ignore_index=True).sample(frac=1.0, random_state=42).reset_index(drop=True)
    print(f"Adversarially Augmented Train Set: {len(df_adv_train_full)} samples", flush=True)

    results_records = []

    evaluation_scenarios = [
        ("Clean Model on Clean Test", df_clean_train, df_itest, False, "clean_test"),
        ("Clean Model on Adversarial Test", df_clean_train, df_adv_test, False, "adv_test_clean_model"),
        ("Adv-Trained Model on Adversarial Test", df_adv_train_full, df_adv_test, True, "adv_test_adv_model")
    ]

    for scenario_name, train_df, test_df, is_adv_trained, cache_key in evaluation_scenarios:
        print(f"\n--- Scenario: {scenario_name} ---", flush=True)
        seed_metrics = []

        for seed in SEEDS:
            df_tr, df_vl = split_train_val_grouped(train_df, val_domain_ratio=0.15, random_state=seed)

            X_tr, y_tr, _ = get_feature_matrix(df_tr, f"e6_{cache_key}_tr_{seed}", include_regional=True, include_path_query=True)
            X_vl, y_vl, _ = get_feature_matrix(df_vl, f"e6_{cache_key}_vl_{seed}", include_regional=True, include_path_query=True)
            X_te, y_te, _ = get_feature_matrix(test_df, f"e6_{cache_key}_te", include_regional=True, include_path_query=True)

            clf = PhishingClassifier(model_type="lightgbm", random_state=seed)
            clf.fit(X_tr, y_tr, X_val=X_vl, y_val=y_vl)
            seed_metrics.append(clf.evaluate(X_te, y_te))

        avg_res = {
            "experiment": "E06_adversarial_robustness",
            "scenario": scenario_name,
            "adversarially_trained": is_adv_trained,
            "precision_mean": float(np.mean([m["precision"] for m in seed_metrics])),
            "precision_std": float(np.std([m["precision"] for m in seed_metrics])),
            "recall_mean": float(np.mean([m["recall"] for m in seed_metrics])),
            "recall_std": float(np.std([m["recall"] for m in seed_metrics])),
            "f1_mean": float(np.mean([m["f1"] for m in seed_metrics])),
            "f1_std": float(np.std([m["f1"] for m in seed_metrics])),
            "roc_auc_mean": float(np.mean([m["roc_auc"] for m in seed_metrics])),
            "roc_auc_std": float(np.std([m["roc_auc"] for m in seed_metrics])),
            "fpr_mean": float(np.mean([m["fpr"] for m in seed_metrics])),
            "fpr_std": float(np.std([m["fpr"] for m in seed_metrics])),
            "fnr_mean": float(np.mean([m["fnr"] for m in seed_metrics])),
            "fnr_std": float(np.std([m["fnr"] for m in seed_metrics])),
            "latency_ms": float(np.mean([m["latency_ms"] for m in seed_metrics])),
            "model_size_mb": float(np.mean([m["model_size_mb"] for m in seed_metrics])),
        }
        results_records.append(avg_res)
        print(f"  Result: F1={avg_res['f1_mean']:.4f}±{avg_res['f1_std']:.4f} | Recall={avg_res['recall_mean']:.4f}±{avg_res['recall_std']:.4f} | FNR={avg_res['fnr_mean']:.4f}±{avg_res['fnr_std']:.4f}", flush=True)

    res_df = pd.DataFrame(results_records)
    res_df.to_csv("results/E06.csv", index=False)
    print("\nSaved Experiment E6 results to results/E06.csv", flush=True)
    return res_df


if __name__ == "__main__":
    run_experiment_e6()
