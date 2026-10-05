"""
Extended UGAA experiments, addressing two follow-up review points:

1. BUDGET COMPARISON: does UGAA at a 5% nominal budget match or exceed
   fixed-rate random audit at LARGER budgets (10%, 15%, 20%)? If so,
   UGAA achieves comparable drift-detection coverage at a fraction of
   the audit workload - a much stronger practical claim than "UGAA
   beats fixed-rate audit at the same budget" alone.

2. K-SENSITIVITY: the sensitivity parameter k in Equation (UGAA) was
   fixed at k=2 throughout the primary experiments. This sweeps
   k in {0.5, 1, 2, 4} at a fixed 5% nominal budget, to test whether
   the reported improvement is robust or depends heavily on this one
   parameter choice.

Both experiments reuse the exact same TrustEngine, arbitration
decision rule, and multi-category drift-injection protocol as
ugaa_experiment.py, so results are directly comparable to what you
already have.

Usage: python3 ugaa_extended_experiments.py <predictions_csv> [output_prefix]
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
DRIFT_BURST_LEN = 150
PRIMARY_UGAA_BUDGET = 0.05
BUDGET_COMPARISON_RATES = [0.05, 0.10, 0.15, 0.20]  # fixed-audit rates to compare UGAA-5% against
K_SWEEP_VALUES = [0.5, 1.0, 2.0, 4.0]


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

    def variance_for(self, category):
        a, b = self.state.get(category, (1.0, 1.0))
        return (a * b) / ((a + b) ** 2 * (a + b + 1))


def arbitration_decision(trust, confidence, params=FIXED_THRESHOLDS):
    trust_high, conf_high, trust_low, conf_low = params
    return "auto" if (trust >= trust_high and confidence >= conf_high) else "escalate"


class UncertaintyGuidedAuditor:
    def __init__(self, target_rate=PRIMARY_UGAA_BUDGET, sensitivity=2.0, ema_alpha=0.02):
        self.target_rate = target_rate
        self.sensitivity = sensitivity
        self.ema_alpha = ema_alpha
        self.running_mean_var = None

    def audit_probability(self, engine, category):
        var = engine.variance_for(category)
        if self.running_mean_var is None:
            self.running_mean_var = var
            return self.target_rate
        relative_uncertainty = var / max(self.running_mean_var, 1e-12)
        prob = self.target_rate * (relative_uncertainty ** self.sensitivity)
        self.running_mean_var = (1 - self.ema_alpha) * self.running_mean_var + self.ema_alpha * var
        return float(np.clip(prob, 0.0, 1.0))


def select_target_category(shuffled):
    cat_counts = shuffled["pred_category"].value_counts()
    eligible = cat_counts[cat_counts >= 20].index.tolist() or cat_counts.index.tolist()
    cat_accuracy = shuffled[shuffled["pred_category"].isin(eligible)].groupby("pred_category")["correct"].mean()
    min_required_instances = int(DRIFT_BURST_LEN * 1.5)
    ranked = cat_accuracy.sort_values(ascending=False)
    for cat in ranked.index:
        if cat_counts[cat] >= min_required_instances:
            return cat
    return None


def drift_test(preds, seed, method, target_category, audit_rate=PRIMARY_UGAA_BUDGET, k=2.0,
                feedback_error_rate=0.08, pre_drift_n=1500):
    """Same multi-category drift-injection protocol as ugaa_experiment.py.
    method: 'fixed_audit' (uses audit_rate) or 'ugaa' (uses audit_rate as
    target budget and k as sensitivity)."""
    rng = np.random.default_rng(seed)
    shuffled = preds.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    n = len(shuffled)
    pre_drift_n = min(pre_drift_n, max(n - DRIFT_BURST_LEN, 100))
    pre_drift_stream = shuffled.iloc[:pre_drift_n]
    post_pool = shuffled.iloc[pre_drift_n:]
    raw_prevalence = (post_pool["pred_category"] == target_category).mean()
    if raw_prevalence <= 0:
        return None
    needed_window = int(np.ceil(DRIFT_BURST_LEN / raw_prevalence * 1.8))
    window_size = min(needed_window, len(post_pool))
    drift_window = post_pool.sample(n=window_size, replace=(window_size > len(post_pool)), random_state=seed + 1).reset_index(drop=True)

    engine = TrustEngine(decay=DECAY)
    auditor = UncertaintyGuidedAuditor(target_rate=audit_rate, sensitivity=k) if method == "ugaa" else None

    def process_event(row, force_incorrect=False):
        category = row["pred_category"]
        confidence = row["raw_confidence"]
        is_correct = False if force_incorrect else bool(row["correct"])
        current_trust = engine.trust_for(category)
        decision = arbitration_decision(current_trust, confidence)

        if method == "fixed_audit":
            feedback_available = (decision == "escalate") or (rng.random() < audit_rate)
        elif method == "ugaa":
            audit_prob = auditor.audit_probability(engine, category)
            feedback_available = (decision == "escalate") or (rng.random() < audit_prob)
        else:
            raise ValueError(method)

        received = False
        if feedback_available:
            observed = is_correct if rng.random() > feedback_error_rate else (not is_correct)
            engine.update(category, observed)
            received = True
        return received

    for _, row in pre_drift_stream.iterrows():
        process_event(row, force_incorrect=False)
    pre_drift_trust = engine.trust_for(target_category)

    detected, drift_events_seen = 0, 0
    for _, row in drift_window.iterrows():
        is_target = (row["pred_category"] == target_category)
        if is_target:
            if drift_events_seen >= DRIFT_BURST_LEN:
                continue
            received = process_event(row, force_incorrect=True)
            drift_events_seen += 1
            if received:
                detected += 1
        else:
            process_event(row, force_incorrect=False)
        if drift_events_seen >= DRIFT_BURST_LEN:
            break

    if drift_events_seen < DRIFT_BURST_LEN:
        return None

    post_drift_trust = engine.trust_for(target_category)
    return {
        "trust_drop": float(pre_drift_trust - post_drift_trust),
        "feedback_coverage": detected / DRIFT_BURST_LEN,
    }


def run_experiment_1_budget_comparison(preds, target_category):
    print("=" * 70)
    print("EXPERIMENT 1: Does UGAA at 5% match/beat fixed-rate audit at LARGER budgets?")
    print("=" * 70)
    results = {}

    print("\n--- UGAA at 5% nominal budget (k=2, primary configuration) ---")
    seed_results = [drift_test(preds, s, "ugaa", target_category, audit_rate=PRIMARY_UGAA_BUDGET, k=2.0)
                     for s in range(N_SEEDS)]
    seed_results = [r for r in seed_results if r is not None]
    ugaa_coverage = np.mean([r["feedback_coverage"] for r in seed_results])
    print(f"  coverage: {ugaa_coverage:.4f} +/- {np.std([r['feedback_coverage'] for r in seed_results]):.4f}")
    results["ugaa_5pct"] = {"mean_coverage": float(ugaa_coverage), "per_seed": seed_results}

    for rate in BUDGET_COMPARISON_RATES:
        print(f"\n--- Fixed-rate random audit at {rate*100:.0f}% ---")
        seed_results = [drift_test(preds, s, "fixed_audit", target_category, audit_rate=rate) for s in range(N_SEEDS)]
        seed_results = [r for r in seed_results if r is not None]
        cov = np.mean([r["feedback_coverage"] for r in seed_results])
        print(f"  coverage: {cov:.4f} +/- {np.std([r['feedback_coverage'] for r in seed_results]):.4f}")
        comparison = "UGAA-5% matches or beats this" if ugaa_coverage >= cov else "this larger budget still beats UGAA-5%"
        print(f"  -> {comparison}")
        results[f"fixed_{int(rate*100)}pct"] = {"mean_coverage": float(cov), "per_seed": seed_results}

    print(f"\n{'='*70}\nSUMMARY: smallest fixed-rate budget that UGAA-5% matches or exceeds\n{'='*70}")
    matched_rate = None
    for rate in BUDGET_COMPARISON_RATES:
        if ugaa_coverage >= results[f"fixed_{int(rate*100)}pct"]["mean_coverage"]:
            matched_rate = rate
    if matched_rate:
        print(f"UGAA at 5% nominal budget achieves coverage >= fixed-rate random audit at {matched_rate*100:.0f}%")
        print(f"({matched_rate/PRIMARY_UGAA_BUDGET:.1f}x the audit workload, for equal or worse coverage)")
    else:
        print("UGAA at 5% does not match fixed-rate audit even at the smallest budget tested (5%) on this configuration")

    return results


def run_experiment_2_k_sensitivity(preds, target_category):
    print(f"\n{'='*70}")
    print("EXPERIMENT 2: k-sensitivity sweep (audit budget fixed at 5%)")
    print("=" * 70)
    results = {}
    for k in K_SWEEP_VALUES:
        print(f"\n--- k = {k} ---")
        seed_results = [drift_test(preds, s, "ugaa", target_category, audit_rate=PRIMARY_UGAA_BUDGET, k=k)
                         for s in range(N_SEEDS)]
        seed_results = [r for r in seed_results if r is not None]
        cov = np.mean([r["feedback_coverage"] for r in seed_results])
        std = np.std([r["feedback_coverage"] for r in seed_results])
        print(f"  coverage: {cov:.4f} +/- {std:.4f}")
        results[f"k_{k}"] = {"mean_coverage": float(cov), "std_coverage": float(std), "per_seed": seed_results}

    print(f"\n{'='*70}\nSUMMARY: coverage range across k values\n{'='*70}")
    covs = [results[f"k_{k}"]["mean_coverage"] for k in K_SWEEP_VALUES]
    print(f"Range: {min(covs):.4f} to {max(covs):.4f} (spread: {(max(covs)-min(covs))*100:.2f} percentage points)")
    print("A narrow spread indicates UGAA's benefit is robust to this parameter choice, not dependent on k=2 specifically.")
    return results


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 ugaa_extended_experiments.py <predictions_csv> [output_prefix]")
        sys.exit(1)
    preds_path = Path(sys.argv[1])
    prefix = sys.argv[2] if len(sys.argv) > 2 else preds_path.stem

    if not preds_path.exists():
        raise FileNotFoundError(f"Missing {preds_path}")
    preds = pd.read_csv(preds_path)
    print(f"Loaded {len(preds)} predictions from {preds_path}")

    target_category = select_target_category(preds)
    if target_category is None:
        print("No eligible target category found with enough instances - cannot run.")
        sys.exit(1)
    print(f"Target category (same selection logic as ugaa_experiment.py): {target_category}\n")

    exp1_results = run_experiment_1_budget_comparison(preds, target_category)
    exp2_results = run_experiment_2_k_sensitivity(preds, target_category)

    all_results = {
        "target_category": target_category,
        "experiment_1_budget_comparison": exp1_results,
        "experiment_2_k_sensitivity": exp2_results,
    }
    with open(OUT_DIR / f"{prefix}_ugaa_extended_results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nSaved to {prefix}_ugaa_extended_results.json in {OUT_DIR}")


if __name__ == "__main__":
    main()
