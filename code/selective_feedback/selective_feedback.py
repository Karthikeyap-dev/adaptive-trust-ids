"""
Selective-feedback experiment (Issue #10 - the biggest remaining gap).

Tests three feedback regimes under identical conditions:
  1. FULL: every alert gets feedback (primary experiments' assumption)
  2. ESCALATION-ONLY: feedback only for escalated alerts
  3. RANDOM-AUDIT: escalation feedback + a random 5% audit of auto-executed alerts

Tracks silent failure/escalation rate under each, plus a drift-recovery
test: after a category earns enough trust to be mostly auto-executed,
its reliability is forced to collapse - testing whether each regime can
even DETECT this (since auto-executed alerts may receive no feedback
under escalation-only), not just how fast it recovers once detected.

Usage: python3 selective_feedback.py <predictions_csv> [output_prefix]
"""
from pathlib import Path
import json
import sys
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = PROJECT_ROOT / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

N_SEEDS = 10
DECAY = 0.995
FIXED_THRESHOLDS = (0.75, 0.80, 0.60, 0.50)
AUDIT_RATE = 0.05
TRAJECTORY_LOG_INTERVAL = 50
DRIFT_BURST_LEN = 150


class TrustEngine:
    def __init__(self, decay=0.995):
        self.decay = decay
        self.state = {}

    def update(self, category, feedback_correct):
        a, b = self.state.get(category, (1.0, 1.0))
        a *= self.decay
        b *= self.decay
        if feedback_correct:
            a += 1.0
        else:
            b += 1.0
        self.state[category] = (a, b)

    def trust_for(self, category):
        a, b = self.state.get(category, (1.0, 1.0))
        return a / (a + b)


def arbitration_decision(trust, confidence, params=FIXED_THRESHOLDS):
    trust_high, conf_high, trust_low, conf_low = params
    return "auto" if (trust >= trust_high and confidence >= conf_high) else "escalate"


def simulate_regime(preds, seed, regime, audit_rate=AUDIT_RATE, feedback_error_rate=0.08):
    rng = np.random.default_rng(seed)
    shuffled = preds.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    engine = TrustEngine(decay=DECAY)

    n_auto, n_escalate, n_silent_fail = 0, 0, 0
    trajectory = []

    for i, row in shuffled.iterrows():
        category = row["pred_category"]
        confidence = row["raw_confidence"]
        is_correct = bool(row["correct"])

        current_trust = engine.trust_for(category)
        decision = arbitration_decision(current_trust, confidence)

        if decision == "auto":
            n_auto += 1
            if not is_correct:
                n_silent_fail += 1
        else:
            n_escalate += 1

        feedback_available = False
        if regime == "full":
            feedback_available = True
        elif regime == "escalation_only":
            feedback_available = (decision == "escalate")
        elif regime == "random_audit":
            feedback_available = (decision == "escalate") or (rng.random() < audit_rate)

        if feedback_available:
            observed_correct = is_correct if rng.random() > feedback_error_rate else (not is_correct)
            engine.update(category, observed_correct)

        if i % TRAJECTORY_LOG_INTERVAL == 0:
            trajectory.append({"event": int(i), "category": category, "trust": current_trust})

    silent_failure_rate = n_silent_fail / max(n_auto, 1)
    escalation_rate = n_escalate / len(shuffled)
    return {
        "regime": regime, "silent_failure_rate": float(silent_failure_rate),
        "escalation_rate": float(escalation_rate), "n_auto": n_auto, "n_escalate": n_escalate,
        "final_trust_by_category": {c: engine.trust_for(c) for c in engine.state.keys()},
        "trajectory": trajectory,
    }


def simulate_drift_recovery(preds, seed, regime, target_category=None, feedback_error_rate=0.08):
    rng = np.random.default_rng(seed)
    shuffled = preds.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    if target_category is None:
        # Select the category MOST LIKELY TO BE AUTO-EXECUTED - i.e. highest
        # empirical accuracy/precision when predicted - NOT simply the most
        # frequent category. A common category with low reliability (e.g.
        # NSL-KDD's "normal", trust ~0.60, below the 0.75 auto-execute
        # threshold) is already being escalated regardless of feedback
        # regime, making the escalation-only-vs-full distinction meaningless
        # for it. Categories are ranked by empirical correctness rate among
        # predictions of that category, restricted to categories with at
        # least 20 instances for a stable estimate.
        cat_counts = shuffled["pred_category"].value_counts()
        eligible_cats = cat_counts[cat_counts >= 20].index.tolist()
        if not eligible_cats:
            eligible_cats = cat_counts.index.tolist()
        cat_accuracy = shuffled[shuffled["pred_category"].isin(eligible_cats)].groupby("pred_category")["correct"].mean()
        target_category = cat_accuracy.idxmax()
    target_pool = shuffled[shuffled["pred_category"] == target_category]
    if len(target_pool) < 20:
        return None

    engine = TrustEngine(decay=DECAY)
    build_up = target_pool.sample(n=min(300, len(target_pool) * 3), replace=True, random_state=seed)
    for _, row in build_up.iterrows():
        is_correct = bool(row["correct"])
        observed = is_correct if rng.random() > feedback_error_rate else (not is_correct)
        engine.update(target_category, observed)
    pre_drift_trust = engine.trust_for(target_category)

    drift_pool = target_pool.sample(n=DRIFT_BURST_LEN, replace=True, random_state=seed + 1)
    trust_during_drift = []
    detected_via_feedback = 0
    for _, row in drift_pool.iterrows():
        confidence = row["raw_confidence"]
        is_correct = False
        current_trust = engine.trust_for(target_category)
        decision = arbitration_decision(current_trust, confidence)

        feedback_available = (regime == "full") or \
                              (regime == "escalation_only" and decision == "escalate") or \
                              (regime == "random_audit" and (decision == "escalate" or rng.random() < AUDIT_RATE))
        if feedback_available:
            engine.update(target_category, is_correct)
            detected_via_feedback += 1
        trust_during_drift.append(engine.trust_for(target_category))

    post_drift_trust = engine.trust_for(target_category)
    min_trust_during_drift = min(trust_during_drift) if trust_during_drift else post_drift_trust

    return {
        "target_category": target_category, "pre_drift_trust": float(pre_drift_trust),
        "post_drift_trust": float(post_drift_trust), "min_trust_during_drift": float(min_trust_during_drift),
        "trust_drop": float(pre_drift_trust - post_drift_trust),
        "feedback_events_received_during_drift": detected_via_feedback,
        "drift_events_total": DRIFT_BURST_LEN,
        "feedback_coverage_during_drift": detected_via_feedback / DRIFT_BURST_LEN,
    }


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 selective_feedback.py <predictions_csv> [output_prefix]")
        sys.exit(1)
    preds_path = Path(sys.argv[1])
    prefix = sys.argv[2] if len(sys.argv) > 2 else preds_path.stem

    if not preds_path.exists():
        raise FileNotFoundError(f"Missing {preds_path}")
    preds = pd.read_csv(preds_path)
    print(f"Loaded {len(preds)} predictions from {preds_path}\n")

    regimes = ["full", "escalation_only", "random_audit"]

    print("=" * 70)
    print("PART 1: Silent failure / escalation rate under each feedback regime")
    print("=" * 70)
    part1_results = {r: [] for r in regimes}
    for regime in regimes:
        print(f"\n--- {regime} ---")
        for seed in range(N_SEEDS):
            r = simulate_regime(preds, seed, regime)
            part1_results[regime].append(r)
        sf = [r["silent_failure_rate"] for r in part1_results[regime]]
        esc = [r["escalation_rate"] for r in part1_results[regime]]
        print(f"  silent_failure={np.mean(sf):.4f}+/-{np.std(sf):.4f}  escalation={np.mean(esc):.4f}+/-{np.std(esc):.4f}")

    print(f"\n{'=' * 70}")
    print("SUMMARY TABLE: Part 1")
    print("=" * 70)
    print(f"{'Regime':<20}{'Silent failure':>18}{'Escalation':>16}")
    for regime in regimes:
        sf = [r["silent_failure_rate"] for r in part1_results[regime]]
        esc = [r["escalation_rate"] for r in part1_results[regime]]
        print(f"{regime:<20}{np.mean(sf):>18.4f}{np.mean(esc):>16.4f}")

    print(f"\n{'=' * 70}")
    print("PART 2: Drift-recovery - can each regime detect a reliability collapse")
    print("in an already-trusted, mostly-auto-executed category?")
    print("=" * 70)
    part2_results = {}
    for regime in regimes:
        print(f"\n--- {regime} ---")
        seed_results = []
        for seed in range(N_SEEDS):
            r = simulate_drift_recovery(preds, seed, regime)
            if r is not None:
                seed_results.append(r)
        if not seed_results:
            print("  [insufficient data for this configuration's dominant category]")
            continue
        trust_drops = [r["trust_drop"] for r in seed_results]
        coverage = [r["feedback_coverage_during_drift"] for r in seed_results]
        print(f"  target category: {seed_results[0]['target_category']}")
        print(f"  mean trust drop during drift: {np.mean(trust_drops):.4f} +/- {np.std(trust_drops):.4f}")
        print(f"  mean feedback coverage during drift: {np.mean(coverage):.4f}")
        if np.mean(coverage) < 0.1:
            print("  *** LOW COVERAGE: drift was largely INVISIBLE to the trust estimator under this regime ***")
        part2_results[regime] = {
            "per_seed": seed_results, "mean_trust_drop": float(np.mean(trust_drops)),
            "mean_feedback_coverage": float(np.mean(coverage)),
        }

    all_results = {
        "part1_feedback_regimes": {
            regime: {
                "silent_failure_rate_mean": float(np.mean([r["silent_failure_rate"] for r in results])),
                "silent_failure_rate_std": float(np.std([r["silent_failure_rate"] for r in results])),
                "escalation_rate_mean": float(np.mean([r["escalation_rate"] for r in results])),
                "escalation_rate_std": float(np.std([r["escalation_rate"] for r in results])),
                "per_seed": results,
            } for regime, results in part1_results.items()
        },
        "part2_drift_recovery": part2_results,
    }
    with open(OUT_DIR / f"{prefix}_selective_feedback.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    csv_rows = []
    for regime, results in part1_results.items():
        for r in results:
            csv_rows.append({"regime": regime, "silent_failure_rate": r["silent_failure_rate"],
                              "escalation_rate": r["escalation_rate"]})
    pd.DataFrame(csv_rows).to_csv(OUT_DIR / f"{prefix}_selective_feedback_part1.csv", index=False)
    print(f"\nSaved to {prefix}_selective_feedback.json and {prefix}_selective_feedback_part1.csv in {OUT_DIR}")


if __name__ == "__main__":
    main()
