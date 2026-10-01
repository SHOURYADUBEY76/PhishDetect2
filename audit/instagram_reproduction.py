"""Instagram and Benign Homepage Sensitivity Reproduction Script.
Audits model behavior on representation shifts: trailing slashes, protocol defaults,
and single-feature flips on Instagram and 20 independent benign homepages.
Saves results to audit/results/instagram_reproduction.json and .txt.
"""

import os
import sys
import json
from urllib.parse import urlparse
import numpy as np
import pandas as pd
import joblib

# Ensure root directory is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.features.extractor import extract_features_dict, get_feature_names
from src.utils.domain import normalize_url


INDEPENDENT_BENIGN_HOMEPAGES = [
    "wikipedia.org", "github.com", "apple.com", "amazon.com", "microsoft.com",
    "linkedin.com", "reddit.com", "netflix.com", "spotify.com", "nytimes.com",
    "bbc.com", "cnn.com", "stackoverflow.com", "medium.com", "dropbox.com",
    "quora.com", "salesforce.com", "adobe.com", "zoom.us", "x.com"
]


def run_instagram_audit():
    model_path = "backend/model_region_aware.joblib"
    d = joblib.load(model_path)
    clf = d["model"]
    threshold = float(d.get("optimal_threshold", 0.5))
    fnames = get_feature_names(include_regional=True)

    instagram_variants = [
        "https://www.instagram.com",
        "http://www.instagram.com",
        "instagram.com",
        "https://www.instagram.com/"
    ]

    variant_results = {}
    feature_dicts = {}

    for raw_u in instagram_variants:
        norm_u = normalize_url(raw_u)
        p = urlparse(norm_u)
        feats = extract_features_dict(norm_u, include_regional=True)
        feature_dicts[raw_u] = feats
        vec = np.array([[feats[f] for f in fnames]])
        prob = float(clf.predict_proba(vec)[0, 1])
        verdict = "PHISHING" if prob >= threshold else "LEGITIMATE"

        # Single feature flip analysis
        flips = {}
        for feat in ["path_length", "is_https", "is_http", "num_slashes", "path_depth"]:
            v_flip = vec.copy()
            idx = fnames.index(feat)
            orig_val = v_flip[0, idx]
            if feat in ["is_https", "is_http"]:
                v_flip[0, idx] = 1.0 - orig_val
            elif feat == "path_length":
                v_flip[0, idx] = 1.0 if orig_val == 0.0 else 0.0
            elif feat == "num_slashes":
                v_flip[0, idx] = orig_val + 1.0 if orig_val <= 2.0 else 2.0
            elif feat == "path_depth":
                v_flip[0, idx] = 1.0 if orig_val == 0.0 else 0.0

            prob_flip = float(clf.predict_proba(v_flip)[0, 1])
            flips[feat] = {
                "orig_val": orig_val,
                "flipped_val": v_flip[0, idx],
                "new_prob": prob_flip,
                "delta": round(prob_flip - prob, 4)
            }

        variant_results[raw_u] = {
            "raw_input": raw_u,
            "normalized_url": norm_u,
            "parsed_components": {
                "scheme": p.scheme,
                "netloc": p.netloc,
                "path": p.path,
                "query": p.query
            },
            "probability": round(prob, 4),
            "calibrated_threshold": round(threshold, 4),
            "verdict": verdict,
            "single_feature_flips": flips
        }

    # Identify features that differ across the 4 variants
    differing_features = {}
    for f in fnames:
        vals = [feature_dicts[u][f] for u in instagram_variants]
        if len(set(vals)) > 1:
            differing_features[f] = {u: feature_dicts[u][f] for u in instagram_variants}

    # 20 Independent Benign Homepages Evaluation
    benign_results = []
    for domain in INDEPENDENT_BENIGN_HOMEPAGES:
        # Test bare, https, https with slash, http
        dom_variants = [f"https://{domain}", f"https://{domain}/", domain]
        dom_evals = {}
        for u in dom_variants:
            norm = normalize_url(u)
            fts = extract_features_dict(norm, include_regional=True)
            v = np.array([[fts[f] for f in fnames]])
            pr = float(clf.predict_proba(v)[0, 1])
            vd = "PHISHING" if pr >= threshold else "LEGITIMATE"
            dom_evals[u] = {"norm": norm, "prob": round(pr, 4), "verdict": vd}
        
        benign_results.append({
            "domain": domain,
            "evaluations": dom_evals,
            "slash_flips_verdict": dom_evals[f"https://{domain}"]["verdict"] != dom_evals[f"https://{domain}/"]["verdict"],
            "prob_drop_with_slash": round(dom_evals[f"https://{domain}"]["prob"] - dom_evals[f"https://{domain}/"]["prob"], 4)
        })

    report = {
        "instagram_variants": variant_results,
        "differing_features_across_instagram_variants": differing_features,
        "twenty_independent_benign_homepages": benign_results,
        "summary": {
            "instagram_bare_https_verdict": variant_results["https://www.instagram.com"]["verdict"],
            "instagram_with_slash_verdict": variant_results["https://www.instagram.com/"]["verdict"],
            "instagram_raw_verdict": variant_results["instagram.com"]["verdict"],
            "slash_flip_count_in_20_homepages": sum(1 for b in benign_results if b["slash_flips_verdict"]),
            "mean_slash_prob_drop_in_20_homepages": round(float(np.mean([b["prob_drop_with_slash"] for b in benign_results])), 4)
        }
    }

    os.makedirs("audit/results", exist_ok=True)
    json_path = "audit/results/instagram_reproduction.json"
    txt_path = "audit/results/instagram_reproduction.txt"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("=== INSTAGRAM & BENIGN HOMEPAGE SENSITIVITY AUDIT ===\n\n")
        f.write("1. Instagram URL Variants:\n")
        for u, r in variant_results.items():
            f.write(f"  URL: {u:30s} -> Norm: {r['normalized_url']:30s}\n")
            f.write(f"    Parsed: scheme={r['parsed_components']['scheme']}, netloc={r['parsed_components']['netloc']}, path='{r['parsed_components']['path']}'\n")
            f.write(f"    Prob: {r['probability']:.4f} | Threshold: {r['calibrated_threshold']:.4f} | Verdict: {r['verdict']}\n")
            f.write("    Single Feature Flips:\n")
            for feat, fl in r["single_feature_flips"].items():
                f.write(f"      {feat:15s}: {fl['orig_val']} -> {fl['flipped_val']} => Prob: {fl['new_prob']:.4f} (delta: {fl['delta']:+.4f})\n")
            f.write("\n")

        f.write("\n2. Differing Features across Instagram Variants:\n")
        for f_name, f_vals in differing_features.items():
            f.write(f"  {f_name}:\n")
            for u, val in f_vals.items():
                f.write(f"    {u:30s}: {val}\n")

        f.write("\n3. 20 Independent Benign Homepages Evaluation:\n")
        for b in benign_results:
            d_name = b["domain"]
            h_prob = b["evaluations"][f"https://{d_name}"]["prob"]
            h_verd = b["evaluations"][f"https://{d_name}"]["verdict"]
            s_prob = b["evaluations"][f"https://{d_name}/"]["prob"]
            s_verd = b["evaluations"][f"https://{d_name}/"]["verdict"]
            r_prob = b["evaluations"][d_name]["prob"]
            r_verd = b["evaluations"][d_name]["verdict"]
            delta = b["prob_drop_with_slash"]
            f.write(f"  {d_name:20s}: https={h_prob:.4f} ({h_verd}) | https+/={s_prob:.4f} ({s_verd}) | raw={r_prob:.4f} ({r_verd}) | Delta={delta:+.4f}\n")

        f.write("\n4. Summary Statistics:\n")
        f.write(f"  Slash flips verdict count in 20 homepages: {report['summary']['slash_flip_count_in_20_homepages']} of 20\n")
        f.write(f"  Mean probability drop caused solely by adding trailing slash: {report['summary']['mean_slash_prob_drop_in_20_homepages']:+.4f}\n")

    print(f"Reproduction finished. Reports saved to:\n  - {json_path}\n  - {txt_path}")


if __name__ == "__main__":
    run_instagram_audit()
