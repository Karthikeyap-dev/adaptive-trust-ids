"""
Distribution shift experiments (Issue #38) - extends beyond the single
injected mid-stream reliability shift already reported:
  (a) Category-prevalence shift: alert MIX shifts, reliability doesn't
  (b) Concurrent multi-category reliability drift (2 categories at once)

Usage: python3 distribution_shift.py <predictions_csv> [output_prefix]
"""
from pathlib import Path
import json
import sys
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = PROJECT_ROOT / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DECAY = 0.995
N_EVENTS_PER_PHASE = 500


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


def experiment_prevalence_shift(preds, seed=42):
    rng = np.random.default_rng(seed)
    categories = preds["pred_category"].unique().tolist()
    engine = TrustEngine(decay=DECAY)

    phase1 = preds.sample(n=N_EVENTS_PER_PHASE, replace=True, random_state=seed)
    for _, row in phase1.iterrows():
        fb = rng.random() > 0.08
        is_correct = bool(row["correct"]) if fb else not bool(row["correct"])
        engine.update(row["pred_category"], is_correct)
    trust_after_phase1 = {c: engine.trust_for(c) for c in categories if c in engine.state}

    spike_category = categories[0]
    spike_pool = preds[preds["pred_category"] == spike_category]
    if len(spike_pool) < 10:
        spike_category = preds["pred_category"].value_counts().idxmax()
        spike_pool = preds[preds["pred_category"] == spike_category]
    phase2 = spike_pool.sample(n=min(N_EVENTS_PER_PHASE, len(spike_pool) * 5), replace=True, random_state=seed + 1)
    for _, row in phase2.iterrows():
        fb = rng.random() > 0.08
        is_correct = bool(row["correct"]) if fb else not bool(row["correct"])
        engine.update(row["pred_category"], is_correct)
    trust_after_phase2 = {c: engine.trust_for(c) for c in categories if c in engine.state}

    spike_change = None
    if spike_category in trust_after_phase1:
        spike_change = trust_after_phase2.get(spike_category, 0) - trust_after_phase1.get(spike_category, 0)

    non_spike_changes = [abs(trust_after_phase2.get(c, 0) - trust_after_phase1.get(c, 0)) for c in categories
                          if c != spike_category and c in trust_after_phase1]

    return {
        "spike_category": spike_category,
        "trust_after_phase1": trust_after_phase1,
        "trust_after_phase2_prevalence_shift": trust_after_phase2,
        "spike_category_trust_change": spike_change,
        "non_spike_categories_max_change": max(non_spike_changes) if non_spike_changes else None,
    }


def experiment_multi_category_drift(preds, seed=42, n_categories_shifted=2):
    categories = preds["pred_category"].unique().tolist()
    if len(categories) < n_categories_shifted:
        n_categories_shifted = len(categories)
    shifted_categories = categories[:n_categories_shifted]

    rng = np.random.default_rng(seed)
    engine = TrustEngine(decay=DECAY)
    trust_trace = {c: [] for c in shifted_categories}

    phase1 = preds.sample(n=N_EVENTS_PER_PHASE, replace=True, random_state=seed)
    for _, row in phase1.iterrows():
        fb = rng.random() > 0.08
        is_correct = bool(row["correct"]) if fb else not bool(row["correct"])
        engine.update(row["pred_category"], is_correct)
    pre_shift_trust = {c: engine.trust_for(c) for c in shifted_categories}

    drift_events = []
    for c in shifted_categories:
        pool = preds[preds["pred_category"] == c]
        if len(pool) > 0:
            n_sample = min(100, len(pool) * 3)
            drift_events.extend(pool.sample(n=n_sample, replace=True, random_state=seed).to_dict("records"))
    rng.shuffle(drift_events)
    for row in drift_events:
        engine.update(row["pred_category"], False)
        for c in shifted_categories:
            trust_trace[c].append(engine.trust_for(c))
    min_trust_during_drift = {c: (min(trust_trace[c]) if trust_trace[c] else None) for c in shifted_categories}

    recovery_events = []
    for c in shifted_categories:
        pool = preds[(preds["pred_category"] == c) & (preds["correct"] == 1)]
        if len(pool) > 0:
            n_sample = min(200, len(pool) * 3)
            recovery_events.extend(pool.sample(n=n_sample, replace=True, random_state=seed + 2).to_dict("records"))
    for row in recovery_events:
        engine.update(row["pred_category"], True)
    post_recovery_trust = {c: engine.trust_for(c) for c in shifted_categories}

    recovered = {}
    for c in shifted_categories:
        if pre_shift_trust.get(c):
            recovered[c] = bool(post_recovery_trust[c] >= 0.9 * pre_shift_trust[c])
        else:
            recovered[c] = None

    return {
        "shifted_categories": shifted_categories,
        "pre_shift_trust": pre_shift_trust,
        "min_trust_during_concurrent_drift": min_trust_during_drift,
        "post_recovery_trust": post_recovery_trust,
        "recovered_to_within_90pct": recovered,
    }


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 distribution_shift.py <predictions_csv> [output_prefix]")
        sys.exit(1)
    preds_path = Path(sys.argv[1])
    prefix = sys.argv[2] if len(sys.argv) > 2 else preds_path.stem

    if not preds_path.exists():
        raise FileNotFoundError(f"Missing {preds_path}")
    preds = pd.read_csv(preds_path)
    print(f"Loaded {len(preds)} predictions from {preds_path}\n")

    print("=" * 70)
    print("EXPERIMENT A: Category-prevalence shift")
    print("=" * 70)
    result_a = experiment_prevalence_shift(preds)
    print(f"Spike category: {result_a['spike_category']}")
    print(f"Trust after phase 1: {result_a['trust_after_phase1']}")
    print(f"Trust after phase 2 (prevalence spike): {result_a['trust_after_phase2_prevalence_shift']}")
    print(f"Spike category's own trust change: {result_a['spike_category_trust_change']}")
    print(f"Max change among NON-spike categories: {result_a['non_spike_categories_max_change']}")

    print(f"\n{'=' * 70}")
    print("EXPERIMENT B: Concurrent multi-category reliability drift")
    print("=" * 70)
    result_b = experiment_multi_category_drift(preds)
    print(f"Categories shifted simultaneously: {result_b['shifted_categories']}")
    print(f"Pre-shift trust: {result_b['pre_shift_trust']}")
    print(f"Minimum trust during concurrent drift: {result_b['min_trust_during_concurrent_drift']}")
    print(f"Post-recovery trust: {result_b['post_recovery_trust']}")
    print(f"Recovered to within 90%: {result_b['recovered_to_within_90pct']}")

    all_results = {"prevalence_shift": result_a, "multi_category_drift": result_b}
    with open(OUT_DIR / f"{prefix}_distribution_shift.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nSaved to {prefix}_distribution_shift.json in {OUT_DIR}")


if __name__ == "__main__":
    main()
