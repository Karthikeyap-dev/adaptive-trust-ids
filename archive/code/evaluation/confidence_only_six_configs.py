"""
Confidence-only arbitration baseline, run identically across all six
dataset-classifier configurations (Issue #6).

Usage: python3 confidence_only_six_configs.py
Edit CONFIG_FILES below to point at your six real predictions CSVs.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = PROJECT_ROOT / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

N_SEEDS = 10
MAX_ACCEPTABLE_SILENT_RATE = 0.05
LATENCY_AUTO_MS, LATENCY_ESCALATE_MS = 5, 4000

CONFIG_FILES = {
    ("NSL-KDD", "RandomForest"): OUT_DIR / "baseline_predictions.csv",
    ("NSL-KDD", "XGBoost"): OUT_DIR / "xgb_baseline_predictions.csv",
    ("UNSW-NB15", "RandomForest"): OUT_DIR / "unsw_baseline_predictions.csv",
    ("UNSW-NB15", "XGBoost"): OUT_DIR / "unsw_xgb_baseline_predictions.csv",
    ("CICIDS2017", "RandomForest"): OUT_DIR / "cicids_baseline_predictions.csv",
    ("CICIDS2017", "XGBoost"): OUT_DIR / "cicids_xgb_baseline_predictions.csv",
}


def evaluate_confidence_threshold(conf_high, conf_arr, correct_arr):
    n = len(conf_arr)
    auto_mask = conf_arr >= conf_high
    auto_count = auto_mask.sum()
    silent_fail = (auto_mask & (correct_arr == 0)).sum()
    silent_rate = silent_fail / max(auto_count, 1)
    escalation_rate = (n - auto_count) / n
    avg_latency = (auto_count * LATENCY_AUTO_MS + (n - auto_count) * LATENCY_ESCALATE_MS) / n
    violation = max(0.0, silent_rate - MAX_ACCEPTABLE_SILENT_RATE)
    metrics = {"silent_failure_rate": float(silent_rate), "escalation_rate": float(escalation_rate)}
    if violation > 0:
        return -5.0 - violation, metrics
    fitness = 0.5 * (1 - silent_rate) + 0.35 * (1 - escalation_rate) + 0.15 * (1 - avg_latency / LATENCY_ESCALATE_MS)
    return fitness, metrics


def optimize_confidence_threshold(conf_arr, correct_arr, seed, n_candidates=200):
    rng = np.random.default_rng(seed)
    base_grid = np.linspace(0.5, 0.999, n_candidates)
    jitter = rng.uniform(-0.001, 0.001, n_candidates)
    candidates = np.clip(base_grid + jitter, 0.5, 0.999)

    best_fitness, best_metrics, best_thresh = -np.inf, None, None
    for thresh in candidates:
        fit, metrics = evaluate_confidence_threshold(thresh, conf_arr, correct_arr)
        if fit > best_fitness:
            best_fitness, best_metrics, best_thresh = fit, metrics, thresh
    return best_thresh, best_metrics


def run_one_seed(preds, seed):
    shuffled = preds.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    conf_arr = shuffled["raw_confidence"].values
    correct_arr = shuffled["correct"].values
    thresh, metrics = optimize_confidence_threshold(conf_arr, correct_arr, seed)
    return {"seed": seed, "threshold": float(thresh), **metrics}


def main():
    all_results = {}
    print("Confidence-only arbitration, all six dataset-classifier configurations\n")

    for (dataset, clf), path in CONFIG_FILES.items():
        key = f"{dataset}_{clf}"
        if not path.exists():
            print(f"[skip] {dataset}/{clf}: {path} not found")
            continue
        preds = pd.read_csv(path)
        print(f"=== {dataset} / {clf} (n={len(preds)}) ===")
        seed_results = []
        for seed in range(N_SEEDS):
            r = run_one_seed(preds, seed)
            seed_results.append(r)
        df = pd.DataFrame(seed_results)
        sf_mean, sf_std = df["silent_failure_rate"].mean(), df["silent_failure_rate"].std()
        esc_mean, esc_std = df["escalation_rate"].mean(), df["escalation_rate"].std()
        print(f"  silent_failure={sf_mean:.4f}+/-{sf_std:.4f}  escalation={esc_mean:.4f}+/-{esc_std:.4f}")
        print(f"  mean optimized threshold={df['threshold'].mean():.4f}\n")

        df.to_csv(OUT_DIR / f"confidence_only_{key}.csv", index=False)
        all_results[key] = {
            "dataset": dataset, "classifier": clf,
            "silent_failure_rate_mean": float(sf_mean), "silent_failure_rate_std": float(sf_std),
            "escalation_rate_mean": float(esc_mean), "escalation_rate_std": float(esc_std),
            "mean_threshold": float(df["threshold"].mean()),
            "per_seed": seed_results,
        }

    print(f"\n{'='*70}\nSUMMARY TABLE (confidence-only, all configurations)\n{'='*70}")
    print(f"{'Dataset':<14}{'Classifier':<14}{'Silent fail':>14}{'Escalation':>14}")
    for key, r in all_results.items():
        print(f"{r['dataset']:<14}{r['classifier']:<14}"
              f"{r['silent_failure_rate_mean']:>14.4f}{r['escalation_rate_mean']:>14.4f}")

    with open(OUT_DIR / "confidence_only_all_six.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved to confidence_only_all_six.json and per-configuration CSVs in {OUT_DIR}")


if __name__ == "__main__":
    main()
