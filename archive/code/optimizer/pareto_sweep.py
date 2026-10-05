"""
Risk-vs-workload trade-off curve (the "8-10% knee" finding).

CORRECTED VERSION: replaces QPSO (with its bounded trust_high in
[0.55, 0.95] and the multi-restart-to-avoid-local-optima pattern) with
the exhaustive grid search. A full grid at a given resolution has no
local-optimum risk, so the restart loop is removed entirely -- there is
only ever one answer per ceiling, not a "best of N tries".

Sweeps the silent-failure ceiling (delta) across several values and
re-runs the grid search at each one, producing the actual risk-vs-
workload trade-off curve: how much does achievable escalation workload
improve as the operator accepts a slightly higher risk budget.

Usage: python3 pareto_sweep.py
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd

from threshold_grid_search import vectorized_grid_search

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = PROJECT_ROOT / "outputs"

CEILINGS_TO_TEST = [0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.10, 0.15, 0.20, 0.30]
LATENCY_AUTO_MS, LATENCY_ESCALATE_MS = 5, 4000


def load_data():
    """Unchanged from the original quantum_optimizer.py."""
    preds = pd.read_csv(OUT_DIR / "baseline_predictions.csv")
    with open(OUT_DIR / "trust_snapshot.json") as f:
        snapshot = json.load(f)
    trust_by_category = {row["category"]: row["trust"] for row in snapshot}
    preds["trust"] = preds["pred_category"].map(trust_by_category).fillna(0.5)
    preds["confidence"] = preds["raw_confidence"]
    return preds[["trust", "confidence", "correct"]].copy()


def main():
    df = load_data()
    trust_arr = df["trust"].values
    conf_arr = df["confidence"].values
    correct_arr = df["correct"].values
    n = len(df)

    results = []
    for ceiling in CEILINGS_TO_TEST:
        print(f"\n--- Grid search for silent-failure ceiling = {ceiling:.0%} ---")

        grid_result = vectorized_grid_search(trust_arr, conf_arr, correct_arr, delta=ceiling)
        trust_high, conf_high = grid_result["tau_h"], grid_result["gamma_h"]

        auto_mask = (trust_arr >= trust_high) & (conf_arr >= conf_high)
        auto_count = int(auto_mask.sum())
        avg_latency_ms = (auto_count * LATENCY_AUTO_MS + (n - auto_count) * LATENCY_ESCALATE_MS) / n

        print(f"  trust_high={trust_high:.3f}  conf_high={conf_high:.3f}  "
              f"failure={grid_result['silent_failure_rate']:.4f}  "
              f"escalation={grid_result['escalation_rate']:.4f}  "
              f"({grid_result['elapsed_seconds']:.1f}s, {grid_result['grid_points_evaluated']:,} policies)")

        results.append({
            "ceiling": ceiling,
            "achieved_silent_failure_rate": grid_result["silent_failure_rate"],
            "escalation_rate": grid_result["escalation_rate"],
            "avg_latency_ms": avg_latency_ms,
            "auto_execute_count": auto_count,
            "params": {"trust_high": round(trust_high, 4), "conf_high": round(conf_high, 4)},
        })

    results_df = pd.DataFrame(results)
    results_df.to_csv(OUT_DIR / "pareto_sweep.csv", index=False)
    with open(OUT_DIR / "pareto_sweep.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\n\n=== Risk-vs-workload trade-off curve ===")
    print(f"{'Ceiling':>10}{'Achieved failure':>18}{'Escalation rate':>18}{'Latency (ms)':>15}")
    for r in results:
        print(f"{r['ceiling']:>10.0%}{r['achieved_silent_failure_rate']:>18.2%}"
              f"{r['escalation_rate']:>18.2%}{r['avg_latency_ms']:>15.1f}")

    print(f"\nSaved full sweep to {OUT_DIR / 'pareto_sweep.csv'} and .json")


if __name__ == "__main__":
    main()
