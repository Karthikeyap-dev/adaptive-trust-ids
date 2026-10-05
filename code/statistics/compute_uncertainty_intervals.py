"""
Uncertainty/CI reporting - addresses the reviewer's request that headline
point-value metrics be reported with proper uncertainty, not bare percentages.

Two distinct statistical situations, handled correctly and differently:

1. Single-run proportions (baseline accuracy, per-category recall on a
   single held-out test set): correct interval is a Wilson score interval
   - more robust than a naive normal approximation, especially near 0/1.

2. Multi-seed repeated-measures results (NSL-KDD 10-seed ablation,
   CICIDS2017 10-seed stability): recomputes a proper t-distribution-based
   95% CI from the actual raw per-seed values already saved on disk.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = PROJECT_ROOT / "outputs"


def wilson_ci(k, n, confidence=0.95):
    if n == 0:
        return None, None, None
    z = stats.norm.ppf(1 - (1 - confidence) / 2)
    p_hat = k / n
    denom = 1 + z**2 / n
    center = (p_hat + z**2 / (2 * n)) / denom
    half_width = (z / denom) * np.sqrt(p_hat * (1 - p_hat) / n + z**2 / (4 * n**2))
    return p_hat, max(0, center - half_width), min(1, center + half_width)


def t_distribution_ci(values, confidence=0.95):
    values = np.asarray(values)
    n = len(values)
    mean = values.mean()
    se = values.std(ddof=1) / np.sqrt(n)
    t_val = stats.t.ppf(1 - (1 - confidence) / 2, df=n - 1)
    return mean, mean - t_val * se, mean + t_val * se


def process_baseline_accuracy_ci(result_filename, dataset_label):
    path = OUT_DIR / result_filename
    if not path.exists():
        print(f"  [skip] {dataset_label} - {result_filename} not found")
        return None
    results = json.load(open(path))
    excluded_keys = {"accuracy", "macro avg", "weighted avg"}
    n_total = sum(v["support"] for k, v in results["per_category_report"].items()
                  if k not in excluded_keys and isinstance(v, dict) and "support" in v)
    acc = results["overall_accuracy"]
    k = round(acc * n_total)
    p_hat, lo, hi = wilson_ci(k, n_total)
    print(f"  {dataset_label}: accuracy = {p_hat:.4f} (95% CI [{lo:.4f}, {hi:.4f}], n={n_total})")

    per_cat_ci = {}
    excluded_keys = {"accuracy", "macro avg", "weighted avg"}
    for cat, vals in results["per_category_report"].items():
        if cat in excluded_keys or not isinstance(vals, dict) or "support" not in vals:
            continue
        support = vals["support"]
        recall_k = round(vals["recall"] * support)
        _, r_lo, r_hi = wilson_ci(recall_k, support)
        per_cat_ci[cat] = {
            "support": support, "recall": vals["recall"], "recall_ci_95": [r_lo, r_hi],
            "f1": vals["f1-score"],
            "note": "F1 CI not computed directly (not a simple binomial proportion); recall CI shown as a proxy.",
        }
        print(f"    {cat:16s}  recall={vals['recall']:.3f}  95% CI [{r_lo:.3f}, {r_hi:.3f}]  (support={support})")

    return {"dataset": dataset_label, "accuracy": p_hat, "accuracy_ci_95": [lo, hi], "n": n_total,
            "per_category": per_cat_ci}


def process_multi_seed_ci(csv_filename, label, metric_cols):
    path = OUT_DIR / csv_filename
    if not path.exists():
        print(f"  [skip] {label} - {csv_filename} not found")
        return None
    df = pd.read_csv(path)
    print(f"\n  {label} (n={len(df)} seeds):")
    result = {}
    for col in metric_cols:
        if col not in df.columns:
            continue
        mean, lo, hi = t_distribution_ci(df[col].values)
        result[col] = {"mean": mean, "ci_95": [lo, hi], "std": float(df[col].std())}
        print(f"    {col:35s}  {mean:.4f}  (95% CI [{lo:.4f}, {hi:.4f}], SD={df[col].std():.4f})")
    return result


def main():
    print("=" * 70)
    print("BASELINE ACCURACY - Wilson score 95% confidence intervals")
    print("=" * 70)
    all_results = {}
    for fname, label in [("baseline_results.json", "NSL-KDD"),
                          ("unsw_baseline_results.json", "UNSW-NB15"),
                          ("cicids_baseline_results.json", "CICIDS2017"),
                          ("xgb_baseline_results.json", "NSL-KDD (XGBoost)")]:
        r = process_baseline_accuracy_ci(fname, label)
        if r:
            all_results[label] = r

    print("\n" + "=" * 70)
    print("MULTI-SEED RESULTS - t-distribution 95% confidence intervals")
    print("=" * 70)
    nsl_seed_result = process_multi_seed_ci(
        "multi_seed_results.csv", "NSL-KDD (10 seeds)",
        ["accuracy", "static_escalation_rate", "adaptive_escalation_rate", "qpso_escalation_rate",
         "static_silent_failure_rate", "adaptive_silent_failure_rate", "qpso_silent_failure_rate"])
    if nsl_seed_result:
        all_results["NSL-KDD multi-seed"] = nsl_seed_result

    cicids_seed_result = process_multi_seed_ci(
        "cicids_multi_seed_results.csv", "CICIDS2017 (10 seeds)",
        ["accuracy", "f1_macro", "f1_weighted", "botnet_f1"])
    if cicids_seed_result:
        all_results["CICIDS2017 multi-seed"] = cicids_seed_result

    with open(OUT_DIR / "uncertainty_intervals.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved all confidence intervals to {OUT_DIR / 'uncertainty_intervals.json'}")
    print("\nUse these intervals (not bare point estimates) when reporting these metrics in the paper.")


if __name__ == "__main__":
    main()
