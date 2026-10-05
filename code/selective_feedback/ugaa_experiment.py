"""
Uncertainty-Guided Adaptive Audit (UGAA) - a new audit-targeting policy
that spends a fixed audit budget preferentially on categories with high
posterior variance (Equation 2 in the paper: Var_{a,c}) - i.e. categories
whose trust estimate rests on comparatively little confirmed evidence -
rather than spreading that same budget uniformly at random across all
auto-executed alerts.

This directly targets the paper's central finding: fixed-rate random
audit (5%) is nowhere near sufficient to detect a reliability collapse
on several configurations (0.3-4.7% coverage even with auditing). UGAA
asks whether the SAME audit budget, spent more intelligently, can do
meaningfully better.

IMPORTANT - fair comparison: UGAA is compared against fixed-rate random
audit at the SAME NOMINAL TARGET RATE (e.g. 5%), not a larger budget.
The script reports the REALIZED overall audit rate for both methods so
you can confirm the comparison is genuinely budget-matched, not just
nominally matched.

Two things are measured for the drift-detection test, matching
selective_feedback.py's protocol exactly so results are directly
comparable:
  - feedback coverage during the forced reliability collapse
  - resulting trust drop (how visible the collapse was to the estimator)
  - realized overall audit rate (confirms fair budget comparison)
  - time-to-first-detection (how many drift events elapsed before the
    FIRST audit-driven feedback arrived, if any) - a new metric not in
    the original selective_feedback.py, added here because "how fast
    can it detect drift at all" is a natural question once you have an
    adaptive-vs-fixed comparison

Usage: python3 ugaa_experiment.py <predictions_csv> [output_prefix]
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
SEED_OFFSET = 10  # LOCKED-EVAL: seeds 10-19, disjoint from the seeds 0-9 used in
                   # ugaa_extended_experiments.py's k-sensitivity sweep that selected
                   # k=4. This mirrors the policy_dev/locked_eval split already used
                   # for the primary arbitration ablation (Section 5.6) - the seeds
                   # used to SELECT the hyperparameter are never reused to REPORT the
                   # resulting performance.
DECAY = 0.995
FIXED_THRESHOLDS = (0.75, 0.80, 0.60, 0.50)
DRIFT_BURST_LEN = 150
TARGET_AUDIT_RATE = 0.05  # matched to the paper's primary random-audit rate
UGAA_SENSITIVITY = 4.0    # updated from 2.0 based on the k-sensitivity sweep (Experiment 2 of ugaa_extended_experiments.py), which showed k=4 matching or beating k=2 on all six configurations tested


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
    """Allocates a fixed nominal audit budget preferentially to
    high-posterior-variance categories, rather than uniformly at random.

    Calibration note: normalizing against an UNWEIGHTED mean variance
    across known categories systematically under-audits common
    categories, since rare categories mechanically have higher variance
    (less evidence) and inflate that unweighted mean. This version
    normalizes against a frequency-weighted running average of variance
    actually observed among auto-executed alerts, updated online, so the
    long-run realized audit rate converges to the target rate."""

    def __init__(self, target_rate=TARGET_AUDIT_RATE, sensitivity=UGAA_SENSITIVITY, ema_alpha=0.02):
        self.target_rate = target_rate
        self.sensitivity = sensitivity
        self.ema_alpha = ema_alpha
        self.running_mean_var = None  # frequency-weighted running average, initialized on first observation

    def audit_probability(self, engine, category):
        var = engine.variance_for(category)
        if self.running_mean_var is None:
            self.running_mean_var = var
            return self.target_rate
        relative_uncertainty = var / max(self.running_mean_var, 1e-12)
        prob = self.target_rate * (relative_uncertainty ** self.sensitivity)
        # Update the running average AFTER computing this event's probability,
        # weighted by actual occurrence (this IS the frequency weighting -
        # common categories update this average more often, proportional to
        # how often they're actually seen among auto-executed alerts)
        self.running_mean_var = (1 - self.ema_alpha) * self.running_mean_var + self.ema_alpha * var
        return float(np.clip(prob, 0.0, 1.0))


def simulate_regime_fixed_audit(preds, seed, audit_rate, feedback_error_rate=0.08):
    """Baseline: escalation feedback + FIXED-RATE random audit."""
    rng = np.random.default_rng(seed)
    shuffled = preds.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    engine = TrustEngine(decay=DECAY)

    n_auto, n_escalate, n_silent_fail, n_audited = 0, 0, 0, 0
    for _, row in shuffled.iterrows():
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

        feedback_available = (decision == "escalate") or (rng.random() < audit_rate)
        if decision == "auto" and feedback_available:
            n_audited += 1
        if feedback_available:
            observed = is_correct if rng.random() > feedback_error_rate else (not is_correct)
            engine.update(category, observed)

    return {
        "silent_failure_rate": n_silent_fail / max(n_auto, 1),
        "escalation_rate": n_escalate / len(shuffled),
        "realized_audit_rate": n_audited / max(n_auto, 1),
    }


def simulate_regime_ugaa(preds, seed, feedback_error_rate=0.08):
    """UGAA: escalation feedback + uncertainty-targeted audit at matched nominal budget."""
    rng = np.random.default_rng(seed)
    shuffled = preds.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    engine = TrustEngine(decay=DECAY)
    auditor = UncertaintyGuidedAuditor()

    n_auto, n_escalate, n_silent_fail, n_audited = 0, 0, 0, 0
    for _, row in shuffled.iterrows():
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

        audit_prob = auditor.audit_probability(engine, category)
        feedback_available = (decision == "escalate") or (rng.random() < audit_prob)
        if decision == "auto" and feedback_available:
            n_audited += 1
        if feedback_available:
            observed = is_correct if rng.random() > feedback_error_rate else (not is_correct)
            engine.update(category, observed)

    return {
        "silent_failure_rate": n_silent_fail / max(n_auto, 1),
        "escalation_rate": n_escalate / len(shuffled),
        "realized_audit_rate": n_audited / max(n_auto, 1),
    }


def drift_test(preds, seed, method, target_category=None, feedback_error_rate=0.08, audit_rate=TARGET_AUDIT_RATE,
                pre_drift_n=1500):
    """Processes a REALISTIC MULTI-CATEGORY stream throughout (not an
    isolated single-category simulation), so that UGAA has real 'other'
    categories present to reallocate its shared audit budget away from -
    this is essential for a fair test, since UGAA's entire mechanism is
    relative budget allocation across categories, which a single-category
    setup cannot exercise at all.

    Protocol: process pre_drift_n real alerts across ALL categories
    normally (building up realistic multi-category trust state, giving
    the auditor real cross-category variance to compare against), THEN
    inject a DRIFT_BURST_LEN-event forced-incorrect run for the target
    category specifically, interleaved with continuing normal traffic
    from other categories, exactly as a real deployment stream would
    look (one category degrading while others continue normally)."""
    rng = np.random.default_rng(seed)
    shuffled = preds.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    if target_category is None:
        cat_counts = shuffled["pred_category"].value_counts()
        eligible = cat_counts[cat_counts >= 20].index.tolist() or cat_counts.index.tolist()
        cat_accuracy = shuffled[shuffled["pred_category"].isin(eligible)].groupby("pred_category")["correct"].mean()
        # Rank categories by accuracy, but only consider ones with enough
        # TOTAL real instances to actually support a DRIFT_BURST_LEN-event
        # burst (with margin) - a category with, say, 25 total occurrences
        # in the whole dataset cannot supply 150 real drift events no
        # matter how the window is sized; fall back through the ranked
        # list rather than failing outright.
        min_required_instances = int(DRIFT_BURST_LEN * 1.5)
        ranked = cat_accuracy.sort_values(ascending=False)
        target_category = None
        for cat in ranked.index:
            if cat_counts[cat] >= min_required_instances:
                target_category = cat
                break
        if target_category is None:
            return None  # no eligible category has enough real instances at all

    engine = TrustEngine(decay=DECAY)
    auditor = UncertaintyGuidedAuditor(target_rate=audit_rate) if method == "ugaa" else None

    n = len(shuffled)
    pre_drift_n = min(pre_drift_n, max(n - DRIFT_BURST_LEN, 100))
    pre_drift_stream = shuffled.iloc[:pre_drift_n]
    post_pool = shuffled.iloc[pre_drift_n:]
    # Guard only against literal zero prevalence (which would make the
    # window size undefined) - NOT a large floor like 0.01, since flooring
    # a genuinely rare category's prevalence UP makes the code think it's
    # more common than it is, producing a window too small to actually
    # contain enough real occurrences. If the category is very rare, the
    # window should be large (capped naturally by len(post_pool) below).
    raw_prevalence = (post_pool["pred_category"] == target_category).mean()
    if raw_prevalence <= 0:
        return None  # category has no remaining occurrences after the pre-drift phase
    target_prevalence = raw_prevalence
    needed_window = int(np.ceil(DRIFT_BURST_LEN / target_prevalence * 1.8))
    window_size = min(needed_window, len(post_pool))
    drift_window = post_pool.sample(n=window_size, replace=(window_size > len(post_pool)), random_state=seed + 1).reset_index(drop=True)

    if seed == SEED_OFFSET:  # print once per configuration, not once per seed
        actual_target_count_in_window = (drift_window["pred_category"] == target_category).sum()
        total_target_count_in_data = (shuffled["pred_category"] == target_category).sum()
        print(f"    [diagnostic] target_category={target_category}  "
              f"total_occurrences_in_full_data={total_target_count_in_data}  "
              f"post_pool_prevalence={target_prevalence:.5f}  "
              f"window_size={window_size}  "
              f"actual_target_occurrences_in_sampled_window={actual_target_count_in_window}")

    def process_event(row, force_incorrect=False):
        category = row["pred_category"]
        confidence = row["raw_confidence"]
        is_correct = False if force_incorrect else bool(row["correct"])
        current_trust = engine.trust_for(category)
        decision = arbitration_decision(current_trust, confidence)

        if method == "escalation_only":
            feedback_available = (decision == "escalate")
        elif method == "fixed_audit":
            feedback_available = (decision == "escalate") or (rng.random() < audit_rate)
        elif method == "ugaa":
            audit_prob = auditor.audit_probability(engine, category)
            feedback_available = (decision == "escalate") or (rng.random() < audit_prob)
        else:
            raise ValueError(method)

        received_feedback = False
        if feedback_available:
            observed = is_correct if rng.random() > feedback_error_rate else (not is_correct)
            engine.update(category, observed)
            received_feedback = True
        return received_feedback

    # Phase 1: normal multi-category traffic, builds realistic cross-category state
    for _, row in pre_drift_stream.iterrows():
        process_event(row, force_incorrect=False)
    pre_drift_trust = engine.trust_for(target_category)

    # Phase 2: drift window - target_category forced incorrect, OTHER categories
    # continue normally, interleaved in realistic stream order
    trust_during_drift, detected, first_detection_event, drift_events_seen = [], 0, None, 0
    for _, row in drift_window.iterrows():
        is_target = (row["pred_category"] == target_category)
        if is_target:
            if drift_events_seen >= DRIFT_BURST_LEN:
                continue  # already collected enough target-category drift events
            received = process_event(row, force_incorrect=True)
            drift_events_seen += 1
            if received:
                detected += 1
                if first_detection_event is None:
                    first_detection_event = drift_events_seen - 1
            trust_during_drift.append(engine.trust_for(target_category))
        else:
            process_event(row, force_incorrect=False)  # other categories continue normally
        if drift_events_seen >= DRIFT_BURST_LEN:
            break

    if drift_events_seen < DRIFT_BURST_LEN:
        return None  # not enough target-category occurrences in the sampled window

    post_drift_trust = engine.trust_for(target_category)
    return {
        "target_category": target_category,
        "pre_drift_trust": float(pre_drift_trust),
        "post_drift_trust": float(post_drift_trust),
        "trust_drop": float(pre_drift_trust - post_drift_trust),
        "feedback_events_received": detected,
        "feedback_coverage": detected / DRIFT_BURST_LEN,
        "first_detection_event": first_detection_event,
    }


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 ugaa_experiment.py <predictions_csv> [output_prefix]")
        sys.exit(1)
    preds_path = Path(sys.argv[1])
    prefix = sys.argv[2] if len(sys.argv) > 2 else preds_path.stem

    if not preds_path.exists():
        raise FileNotFoundError(f"Missing {preds_path}")
    preds = pd.read_csv(preds_path)
    print(f"Loaded {len(preds)} predictions from {preds_path}")
    print(f"LOCKED-EVAL RUN: seeds {SEED_OFFSET}-{SEED_OFFSET + N_SEEDS - 1}, disjoint from seeds "
          f"0-9 used to select k=4 in the sensitivity sweep. This is the primary reported result.\n")

    print("=" * 70)
    print("PART 1: Steady-state silent failure / escalation / realized audit rate")
    print("=" * 70)
    methods_part1 = {
        "fixed_5pct": lambda s: simulate_regime_fixed_audit(preds, s, 0.05),
        "fixed_10pct": lambda s: simulate_regime_fixed_audit(preds, s, 0.10),
        "ugaa_5pct_budget": lambda s: simulate_regime_ugaa(preds, s),
    }
    part1_results = {}
    for name, fn in methods_part1.items():
        seed_results = [fn(seed) for seed in range(SEED_OFFSET, SEED_OFFSET + N_SEEDS)]
        sf = [r["silent_failure_rate"] for r in seed_results]
        esc = [r["escalation_rate"] for r in seed_results]
        rar = [r["realized_audit_rate"] for r in seed_results]
        print(f"  {name}: silent_failure={np.mean(sf):.4f}+/-{np.std(sf):.4f}  "
              f"escalation={np.mean(esc):.4f}+/-{np.std(esc):.4f}  "
              f"realized_audit_rate={np.mean(rar):.4f}+/-{np.std(rar):.4f}")
        part1_results[name] = {
            "silent_failure_mean": float(np.mean(sf)), "escalation_mean": float(np.mean(esc)),
            "realized_audit_rate_mean": float(np.mean(rar)), "per_seed": seed_results,
        }

    print(f"\n{'=' * 70}")
    print("PART 2: Drift-detection test (fixed 5% audit vs. UGAA at matched 5% nominal budget)")
    print("=" * 70)
    methods_part2 = ["escalation_only", "fixed_audit", "ugaa"]
    part2_results = {}
    for method in methods_part2:
        print(f"\n--- {method} ---")
        seed_results = []
        for seed in range(SEED_OFFSET, SEED_OFFSET + N_SEEDS):
            r = drift_test(preds, seed, method)
            if r is not None:
                seed_results.append(r)
        if not seed_results:
            print("  [insufficient data]")
            continue
        coverage = [r["feedback_coverage"] for r in seed_results]
        trust_drop = [r["trust_drop"] for r in seed_results]
        detections = [r["first_detection_event"] for r in seed_results if r["first_detection_event"] is not None]
        never_detected = sum(1 for r in seed_results if r["first_detection_event"] is None)
        print(f"  target category: {seed_results[0]['target_category']}")
        print(f"  feedback coverage: {np.mean(coverage):.4f} +/- {np.std(coverage):.4f}")
        print(f"  trust drop: {np.mean(trust_drop):.4f} +/- {np.std(trust_drop):.4f}")
        if detections:
            print(f"  mean time-to-first-detection: {np.mean(detections):.1f} events "
                  f"(out of {DRIFT_BURST_LEN}), {never_detected}/{len(seed_results)} seeds never detected")
        else:
            print(f"  NEVER detected within the {DRIFT_BURST_LEN}-event burst on any seed")
        part2_results[method] = {
            "mean_coverage": float(np.mean(coverage)), "mean_trust_drop": float(np.mean(trust_drop)),
            "mean_time_to_detection": float(np.mean(detections)) if detections else None,
            "seeds_never_detected": never_detected, "per_seed": seed_results,
        }

    print(f"\n{'=' * 70}\nSUMMARY: does UGAA improve on fixed-rate audit at the SAME nominal budget?\n{'=' * 70}")
    if "fixed_audit" in part2_results and "ugaa" in part2_results:
        fixed_cov = part2_results["fixed_audit"]["mean_coverage"]
        ugaa_cov = part2_results["ugaa"]["mean_coverage"]
        print(f"Fixed 5% audit coverage: {fixed_cov:.4f}")
        print(f"UGAA coverage (same 5% nominal budget): {ugaa_cov:.4f}")
        print(f"Improvement: {(ugaa_cov - fixed_cov) * 100:.2f} percentage points "
              f"({'UGAA better' if ugaa_cov > fixed_cov else 'fixed-rate better or equal'})")

    all_results = {"part1_steady_state": part1_results, "part2_drift_detection": part2_results}
    with open(OUT_DIR / f"{prefix}_ugaa_results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nSaved to {prefix}_ugaa_results.json in {OUT_DIR}")


if __name__ == "__main__":
    main()
