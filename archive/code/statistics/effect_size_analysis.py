"""
Analysis D: Effect sizes and confidence intervals for the main arbitration
comparisons. Complements the existing Wilcoxon significance tests with
effect sizes - important given that a statistically significant result
can still be practically negligible (as found in Experiment A).
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = PROJECT_ROOT / "outputs"


def cohens_d_paired(x, y):
    diff = np.array(x) - np.array(y)
    sd = diff.std(ddof=1)
    return float(diff.mean() / sd) if sd > 0 else float("nan")


def rank_biserial_from_wilcoxon(x, y):
    diff = np.array(x) - np.array(y)
    diff = diff[diff != 0]
    if len(diff) == 0:
        return 0.0
    ranks = stats.rankdata(np.abs(diff))
    r_plus = ranks[diff > 0].sum()
    r_minus = ranks[diff < 0].sum()
    total = r_plus + r_minus
    return float((r_plus - r_minus) / total) if total > 0 else 0.0


def mean_diff_ci(x, y, confidence=0.95):
    diff = np.array(x) - np.array(y)
    n = len(diff)
    mean = diff.mean()
    se = diff.std(ddof=1) / np.sqrt(n)
    t_val = stats.t.ppf(1 - (1 - confidence) / 2, df=n - 1)
    return mean, mean - t_val * se, mean + t_val * se


def interpret_d(d):
    ad = abs(d)
    if ad < 0.2: return "negligible"
    if ad < 0.5: return "small"
    if ad < 0.8: return "medium"
    return "large"


def interpret_r(r):
    ar = abs(r)
    if ar < 0.1: return "negligible"
    if ar < 0.3: return "small"
    if ar < 0.5: return "medium"
    return "large"


def analyze_comparison(label, x, y, x_name, y_name):
    x, y = np.asarray(x), np.asarray(y)
    mean_diff, ci_lo, ci_hi = mean_diff_ci(x, y)
    d = cohens_d_paired(x, y)
    r = rank_biserial_from_wilcoxon(x, y)
    print(f"\n  {label}:")
    print(f"    Mean difference ({x_name} - {y_name}): {mean_diff:.4f}  (95% CI [{ci_lo:.4f}, {ci_hi:.4f}])")
    print(f"    Cohen's d (paired): {d:.3f}  ({interpret_d(d)})")
    print(f"    Matched-pairs rank-biserial r: {r:.3f}  ({interpret_r(r)})")
    return {"label": label, "mean_diff": float(mean_diff), "ci_95": [float(ci_lo), float(ci_hi)],
            "cohens_d": d, "rank_biserial_r": r}


def main():
    results = {}

    print("=" * 78)
    print("EFFECT SIZES: NSL-KDD three-way ablation (10 seeds)")
    print("=" * 78)
    path = OUT_DIR / "multi_seed_results.csv"
    if path.exists():
        df = pd.read_csv(path)
        results["nsl_kdd_static_vs_adaptive_escalation"] = analyze_comparison(
            "Static vs Adaptive (escalation rate)", df["static_escalation_rate"], df["adaptive_escalation_rate"],
            "static", "adaptive")
        results["nsl_kdd_adaptive_vs_qpso_escalation"] = analyze_comparison(
            "Adaptive vs QPSO (escalation rate)", df["adaptive_escalation_rate"], df["qpso_escalation_rate"],
            "adaptive", "qpso")
    else:
        print("  [skip] multi_seed_results.csv not found")

    print(f"\n{'='*78}")
    print("EFFECT SIZES: UNSW-NB15 three-way ablation (10 seeds)")
    print("=" * 78)
    path = OUT_DIR / "unsw_multi_seed_results.csv"
    if path.exists():
        df = pd.read_csv(path)
        results["unsw_static_vs_adaptive_escalation"] = analyze_comparison(
            "Static vs Adaptive (escalation rate)", df["static_escalation_rate"], df["adaptive_escalation_rate"],
            "static", "adaptive")
        results["unsw_adaptive_vs_qpso_escalation"] = analyze_comparison(
            "Adaptive vs QPSO (escalation rate)", df["adaptive_escalation_rate"], df["qpso_escalation_rate"],
            "adaptive", "qpso")
    else:
        print("  [skip] unsw_multi_seed_results.csv not found")

    print(f"\n{'='*78}")
    print("EFFECT SIZES: Classical PSO vs QPSO (10 seeds, NSL-KDD)")
    print("=" * 78)
    path = OUT_DIR / "classical_vs_qpso_10seed_results.csv"
    if path.exists():
        df = pd.read_csv(path)
        results["classical_vs_qpso_escalation"] = analyze_comparison(
            "Classical PSO vs QPSO (escalation rate)", df["classical_escalation_rate"], df["qpso_escalation_rate"],
            "classical", "qpso")
        print(f"\n  NOTE: even where the p-value is significant, check the effect size - a negligible "
              f"Cohen's d alongside a significant p-value means the difference is statistically "
              f"detectable but practically unimportant.")
    else:
        print("  [skip] classical_vs_qpso_10seed_results.csv not found")

    with open(OUT_DIR / "effect_size_analysis.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {OUT_DIR / 'effect_size_analysis.json'}")


if __name__ == "__main__":
    main()
