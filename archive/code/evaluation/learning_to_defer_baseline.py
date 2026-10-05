"""
Learning-to-defer / selective-prediction baseline (Issue #5).

Trains a SEPARATE, data-driven rejector model predicting P(the base
classifier is wrong) from confidence + a per-category historical error
rate feature (computed only from a held-out calibration split) -
genuinely distinct from confidence-only thresholding, following the
standard learning-to-defer/selective-prediction philosophy.

Usage: python3 learning_to_defer_baseline.py <predictions_csv> [output_prefix]
"""
from pathlib import Path
import json
import sys
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = PROJECT_ROOT / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

N_SEEDS = 10
MAX_ACCEPTABLE_SILENT_RATE = 0.05
LATENCY_AUTO_MS, LATENCY_ESCALATE_MS = 5, 4000


def evaluate_defer_threshold(reject_prob_threshold, p_wrong, correct_arr):
    n = len(p_wrong)
    auto_mask = p_wrong < reject_prob_threshold
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


def optimize_defer_threshold(p_wrong, correct_arr, seed, n_candidates=200):
    rng = np.random.default_rng(seed)
    base_grid = np.linspace(0.001, 0.5, n_candidates)
    jitter = rng.uniform(-0.0005, 0.0005, n_candidates)
    candidates = np.clip(base_grid + jitter, 0.001, 0.5)
    best_fitness, best_metrics, best_thresh = -np.inf, None, None
    for thresh in candidates:
        fit, metrics = evaluate_defer_threshold(thresh, p_wrong, correct_arr)
        if fit > best_fitness:
            best_fitness, best_metrics, best_thresh = fit, metrics, thresh
    return best_thresh, best_metrics


def run_one_seed(preds, seed):
    shuffled = preds.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    n = len(shuffled)
    split1, split2 = int(n * 0.4), int(n * 0.7)
    rejector_train = shuffled.iloc[:split1]
    threshold_dev = shuffled.iloc[split1:split2]
    locked_eval = shuffled.iloc[split2:]

    cat_error_rate = (1 - rejector_train.groupby("pred_category")["correct"].mean()).to_dict()
    global_error_rate = 1 - rejector_train["correct"].mean()

    def add_features(df):
        df = df.copy()
        df["cat_error_rate"] = df["pred_category"].map(cat_error_rate).fillna(global_error_rate)
        return df

    rejector_train = add_features(rejector_train)
    threshold_dev = add_features(threshold_dev)
    locked_eval = add_features(locked_eval)

    X_train = rejector_train[["raw_confidence", "cat_error_rate"]].values
    y_train = (1 - rejector_train["correct"]).values
    rejector = LogisticRegression(max_iter=1000, random_state=seed)
    rejector.fit(X_train, y_train)

    X_dev = threshold_dev[["raw_confidence", "cat_error_rate"]].values
    p_wrong_dev = rejector.predict_proba(X_dev)[:, 1]
    correct_dev = threshold_dev["correct"].values
    best_thresh, _ = optimize_defer_threshold(p_wrong_dev, correct_dev, seed)

    X_eval = locked_eval[["raw_confidence", "cat_error_rate"]].values
    p_wrong_eval = rejector.predict_proba(X_eval)[:, 1]
    correct_eval = locked_eval["correct"].values
    _, eval_metrics = evaluate_defer_threshold(best_thresh, p_wrong_eval, correct_eval)

    return {
        "seed": seed, "defer_threshold": float(best_thresh),
        "rejector_train_n": len(rejector_train), "threshold_dev_n": len(threshold_dev),
        "locked_eval_n": len(locked_eval),
        "silent_failure_rate": eval_metrics["silent_failure_rate"],
        "escalation_rate": eval_metrics["escalation_rate"],
    }


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 learning_to_defer_baseline.py <predictions_csv> [output_prefix]")
        sys.exit(1)
    preds_path = Path(sys.argv[1])
    prefix = sys.argv[2] if len(sys.argv) > 2 else preds_path.stem

    if not preds_path.exists():
        raise FileNotFoundError(f"Missing {preds_path}")
    preds = pd.read_csv(preds_path)
    print(f"Loaded {len(preds)} predictions from {preds_path}")
    print(f"Running learning-to-defer baseline ({N_SEEDS} seeds)\n")

    results = []
    for seed in range(N_SEEDS):
        r = run_one_seed(preds, seed)
        print(f"  seed {seed}: defer_thresh={r['defer_threshold']:.4f}  "
              f"silent_fail={r['silent_failure_rate']:.4f}  escalation={r['escalation_rate']:.4f}")
        results.append(r)

    df = pd.DataFrame(results)
    csv_path = OUT_DIR / f"{prefix}_learning_to_defer_results.csv"
    df.to_csv(csv_path, index=False)

    sf_mean, sf_std = df["silent_failure_rate"].mean(), df["silent_failure_rate"].std()
    esc_mean, esc_std = df["escalation_rate"].mean(), df["escalation_rate"].std()
    print(f"\n{'='*60}\nSUMMARY (learning-to-defer, mean +/- SD across {N_SEEDS} seeds)\n{'='*60}")
    print(f"silent_failure={sf_mean:.4f}+/-{sf_std:.4f}  escalation={esc_mean:.4f}+/-{esc_std:.4f}")

    summary = {
        "prefix": prefix, "n_seeds": N_SEEDS,
        "silent_failure_rate_mean": float(sf_mean), "silent_failure_rate_std": float(sf_std),
        "escalation_rate_mean": float(esc_mean), "escalation_rate_std": float(esc_std),
        "per_seed": results,
    }
    json_path = OUT_DIR / f"{prefix}_learning_to_defer_results.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved to {csv_path} and {json_path}")


if __name__ == "__main__":
    main()
