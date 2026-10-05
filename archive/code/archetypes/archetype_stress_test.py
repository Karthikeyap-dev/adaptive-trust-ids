"""
Archetype stress test - runs the framework's full pipeline (trust engine
+ three-way ablation) separately under each of three analyst archetypes,
at scale (thousands of alerts). Also runs a "mixed team" scenario.

Complements (does not replace) the small real human pilot study.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd

from trust_engine import AdaptiveTrustEngine
from analyst_archetypes import archetype_feedback, ARCHETYPES

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = PROJECT_ROOT / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
TRUST_HIGH, CONF_HIGH, TRUST_LOW, CONF_LOW, W_TRUST = 0.75, 0.80, 0.60, 0.50, 0.6
MAX_ACCEPTABLE_SILENT_RATE = 0.05
TARGET_N_ALERTS = 5000

MIXED_TEAM_WEIGHTS = {"novice": 0.30, "expert": 0.20, "complacent": 0.50}


def evaluate_policy(params, trust_arr, conf_arr, correct_arr):
    trust_high, conf_high, trust_low, conf_low, w_trust = params
    n = len(trust_arr)
    if trust_low >= trust_high or conf_low >= conf_high:
        return None
    auto_mask = (trust_arr >= trust_high) & (conf_arr >= conf_high)
    reject_mask = (~auto_mask) & ((trust_arr <= trust_low) | (conf_arr <= conf_low))
    escalate_mask = ~auto_mask & ~reject_mask
    auto_count = auto_mask.sum()
    silent_fail = (auto_mask & (correct_arr == 0)).sum()
    return {
        "silent_failure_rate": float(silent_fail / max(auto_count, 1)),
        "escalation_rate": float((reject_mask.sum() + escalate_mask.sum()) / n),
        "auto_execute_rate": float(auto_count / n),
    }


def build_alert_stream(preds, target_n, seed):
    rng = np.random.default_rng(seed)
    n_available = len(preds)
    if target_n <= n_available:
        idx = rng.choice(n_available, size=target_n, replace=False)
    else:
        idx = rng.choice(n_available, size=target_n, replace=True)
    return preds.iloc[idx].reset_index(drop=True)


def run_archetype(preds, archetype_name, dataset_key, seed=SEED, weights=None):
    rng = np.random.default_rng(seed)
    stream = build_alert_stream(preds, TARGET_N_ALERTS, seed)
    n = len(stream)

    engine = AdaptiveTrustEngine(decay=0.995)
    response_times = []
    archetype_assignments = []

    for step, row in stream.iterrows():
        cat = row["pred_category"]
        is_correct = bool(row["correct"])
        conf = float(row["raw_confidence"])

        if weights is not None:
            names, probs = zip(*weights.items())
            this_archetype = rng.choice(names, p=probs)
        else:
            this_archetype = archetype_name
        archetype_assignments.append(this_archetype)

        fb, rt = archetype_feedback(this_archetype, is_correct, conf, cat, dataset_key, rng)
        engine.update("primary_agent", cat, fb, step=step)
        response_times.append(rt)

    snapshot = engine.snapshot()
    trust_by_cat = {r["category"]: r["trust"] for r in snapshot}
    trust_arr = stream["pred_category"].map(trust_by_cat).fillna(0.5).values
    conf_arr = stream["raw_confidence"].values
    correct_arr = stream["correct"].values

    fixed_params = [TRUST_HIGH, CONF_HIGH, TRUST_LOW, CONF_LOW, W_TRUST]
    metrics = evaluate_policy(fixed_params, trust_arr, conf_arr, correct_arr)

    return {
        "archetype": archetype_name if weights is None else "mixed_team",
        "n_alerts": n,
        "trust_snapshot": snapshot,
        "arbitration_metrics": metrics,
        "mean_response_time_s": float(np.mean(response_times)),
        "median_response_time_s": float(np.median(response_times)),
        "archetype_composition": (pd.Series(archetype_assignments).value_counts(normalize=True).to_dict()
                                   if weights is not None else None),
    }


def main(preds_filename="baseline_predictions.csv", dataset_key="nsl_kdd", out_suffix=""):
    preds_path = OUT_DIR / preds_filename
    if not preds_path.exists():
        raise FileNotFoundError(f"Missing {preds_path} - run the matching baseline training script first")
    preds = pd.read_csv(preds_path)

    print(f"Dataset: {dataset_key}")
    print(f"Loaded {len(preds)} real predictions. Running each archetype at N={TARGET_N_ALERTS} alerts.\n")

    results = {}
    for archetype_name in ARCHETYPES:
        print(f"{'='*70}\nARCHETYPE: {archetype_name.upper()}\n{'='*70}")
        r = run_archetype(preds, archetype_name, dataset_key)
        results[archetype_name] = r
        print(f"  Mean response time: {r['mean_response_time_s']:.1f}s")
        print(f"  Arbitration: silent_fail={r['arbitration_metrics']['silent_failure_rate']:.4f}  "
              f"escalation={r['arbitration_metrics']['escalation_rate']:.4f}  "
              f"auto_execute={r['arbitration_metrics']['auto_execute_rate']:.4f}")
        print(f"  Trust range: min={min(x['trust'] for x in r['trust_snapshot']):.3f}  "
              f"max={max(x['trust'] for x in r['trust_snapshot']):.3f}")

    print(f"\n{'='*70}\nMIXED TEAM (30% novice / 20% expert / 50% complacent)\n{'='*70}")
    mixed = run_archetype(preds, "mixed_team", dataset_key, weights=MIXED_TEAM_WEIGHTS)
    results["mixed_team"] = mixed
    print(f"  Composition realized: {mixed['archetype_composition']}")
    print(f"  Arbitration: silent_fail={mixed['arbitration_metrics']['silent_failure_rate']:.4f}  "
          f"escalation={mixed['arbitration_metrics']['escalation_rate']:.4f}")

    print(f"\n{'='*70}\nSAFETY CHECK: did the constraint hold under the Complacent archetype?\n{'='*70}")
    complacent_fail = results["complacent"]["arbitration_metrics"]["silent_failure_rate"]
    if complacent_fail > MAX_ACCEPTABLE_SILENT_RATE:
        print(f"  WARNING: Complacent-archetype feedback pushed silent failure rate to "
              f"{complacent_fail:.4f}, EXCEEDING the {MAX_ACCEPTABLE_SILENT_RATE:.0%} design ceiling. "
              f"Report this explicitly - automation-biased feedback can poison the trust signal enough "
              f"to defeat the safety constraint under FIXED thresholds, motivating periodic threshold "
              f"re-optimization or feedback-quality monitoring as necessary safeguards.")
    else:
        print(f"  Silent failure rate ({complacent_fail:.4f}) stayed within the {MAX_ACCEPTABLE_SILENT_RATE:.0%} "
              f"ceiling even under Complacent-archetype feedback - a genuine positive safety finding.")

    out_path = OUT_DIR / f"archetype_stress_test_results{out_suffix}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    import sys
    # Usage: python3 archetype_stress_test.py [nsl_kdd|unsw_nb15|cicids2017]
    target = sys.argv[1] if len(sys.argv) > 1 else "nsl_kdd"
    configs = {
        "nsl_kdd": ("baseline_predictions.csv", "nsl_kdd", "_nsl_kdd"),
        "unsw_nb15": ("unsw_baseline_predictions.csv", "unsw_nb15", "_unsw_nb15"),
        "cicids2017": ("cicids_baseline_predictions.csv", "cicids2017", "_cicids2017"),
    }
    if target not in configs:
        raise ValueError(f"Unknown dataset '{target}' - choose from {list(configs.keys())}")
    main(*configs[target])
