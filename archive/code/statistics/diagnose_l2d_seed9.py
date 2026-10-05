"""
Diagnostic for the L2D seed-9 outlier (NSL-KDD/XGBoost).

Reruns seed 9 specifically and scans the full fitness landscape across
the threshold range to check whether the safety constraint creates
multiple disconnected "feasible shelves" (which would make landing on
a different threshold for one seed a real, non-buggy property of the
search landscape rather than an error).

Usage: python3 diagnose_l2d_seed9.py <predictions_csv>
"""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

MAX_ACCEPTABLE_SILENT_RATE = 0.05
LATENCY_AUTO_MS, LATENCY_ESCALATE_MS = 5, 4000
SEED = 9


def evaluate_defer_threshold(reject_prob_threshold, p_wrong, correct_arr):
    n = len(p_wrong)
    auto_mask = p_wrong < reject_prob_threshold
    auto_count = auto_mask.sum()
    silent_fail = (auto_mask & (correct_arr == 0)).sum()
    silent_rate = silent_fail / max(auto_count, 1)
    escalation_rate = (n - auto_count) / n
    avg_latency = (auto_count * LATENCY_AUTO_MS + (n - auto_count) * LATENCY_ESCALATE_MS) / n
    violation = max(0.0, silent_rate - MAX_ACCEPTABLE_SILENT_RATE)
    feasible = violation == 0
    if violation > 0:
        fitness = -5.0 - violation
    else:
        fitness = 0.5 * (1 - silent_rate) + 0.35 * (1 - escalation_rate) + 0.15 * (1 - avg_latency / LATENCY_ESCALATE_MS)
    return fitness, silent_rate, escalation_rate, feasible


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 diagnose_l2d_seed9.py <predictions_csv>")
        sys.exit(1)
    preds = pd.read_csv(sys.argv[1])
    shuffled = preds.sample(frac=1.0, random_state=SEED).reset_index(drop=True)
    n = len(shuffled)
    split1, split2 = int(n * 0.4), int(n * 0.7)
    rejector_train = shuffled.iloc[:split1].copy()
    threshold_dev = shuffled.iloc[split1:split2].copy()
    locked_eval = shuffled.iloc[split2:].copy()

    cat_error_rate = (1 - rejector_train.groupby("pred_category")["correct"].mean()).to_dict()
    global_error_rate = 1 - rejector_train["correct"].mean()
    for df in (rejector_train, threshold_dev, locked_eval):
        df["cat_error_rate"] = df["pred_category"].map(cat_error_rate).fillna(global_error_rate)

    X_train = rejector_train[["raw_confidence", "cat_error_rate"]].values
    y_train = (1 - rejector_train["correct"]).values
    rejector = LogisticRegression(max_iter=1000, random_state=SEED)
    rejector.fit(X_train, y_train)

    p_wrong_dev = rejector.predict_proba(threshold_dev[["raw_confidence", "cat_error_rate"]].values)[:, 1]
    correct_dev = threshold_dev["correct"].values

    print(f"Seed {SEED}: rejector_train n={len(rejector_train)}, threshold_dev n={len(threshold_dev)}, "
          f"locked_eval n={len(locked_eval)}")
    print(f"p_wrong_dev range: [{p_wrong_dev.min():.4f}, {p_wrong_dev.max():.4f}]\n")

    print(f"{'Threshold':>10}{'Silent_rate':>14}{'Escalation':>14}{'Feasible':>10}{'Fitness':>12}")
    thresholds_to_scan = np.linspace(0.001, 0.98, 100)
    feasible_regions = []
    for idx, t in enumerate(thresholds_to_scan):
        fitness, silent_rate, escalation_rate, feasible = evaluate_defer_threshold(t, p_wrong_dev, correct_dev)
        if idx % 5 == 0:
            marker = "  <-- FEASIBLE" if feasible else ""
            print(f"{t:>10.4f}{silent_rate:>14.4f}{escalation_rate:>14.4f}{str(feasible):>10}{fitness:>12.4f}{marker}")
        feasible_regions.append((t, feasible, fitness))

    feasible_thresholds = [t for t, f, _ in feasible_regions if f]
    if feasible_thresholds:
        print(f"\nFeasible threshold range on threshold_dev: [{min(feasible_thresholds):.4f}, {max(feasible_thresholds):.4f}]")
        feasible_sorted = sorted(feasible_thresholds)
        gaps = []
        for i in range(1, len(feasible_sorted)):
            gap = feasible_sorted[i] - feasible_sorted[i-1]
            if gap > 0.05:
                gaps.append((feasible_sorted[i-1], feasible_sorted[i]))
        if gaps:
            print(f"\n*** DISCONNECTED FEASIBLE REGIONS FOUND: {gaps} ***")
            print("Confirms the hypothesis: the constraint creates multiple separate feasible")
            print("'shelves', and grid search + seed-specific data can land on either one -")
            print("this explains the seed-9 outlier as a real search-landscape property, not a bug.")
        else:
            print("\nNo disconnected regions found - feasible range is contiguous.")
            print("The seed-9 outlier likely comes from real rejector/dev-set variation, not landscape disconnection.")
    else:
        print("\nNo feasible thresholds found in this scan on threshold_dev for this seed.")


if __name__ == "__main__":
    main()
