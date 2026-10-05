"""
Three additional drift scenarios, addressing the review concern that UGAA
was validated only against one artificial scenario type (instantaneous,
complete collapse to 0% correctness). This adds:

  Scenario A - Partial reliability decline: reliability drops to a
    non-zero floor (95%->70% and 90%->60% tested) rather than total
    collapse, testing whether UGAA still helps when the signal is
    weaker/more realistic.
  Scenario B - Gradual drift: reliability steps down through several
    intermediate levels (95->90->80->70->60%) rather than a single
    instantaneous jump, testing time-to-detection under a slow decline.
  Scenario C - Two simultaneous categories: one common, one rare category
    drift concurrently, testing whether UGAA's budget reallocation still
    helps when there are two competing signals rather than one.

All three reuse the exact same TrustEngine, arbitration rule, and
multi-category stream embedding as ugaa_experiment.py, so results are
directly comparable. Compares fixed-rate 5% random audit against UGAA
(k=4, 5% nominal budget) on each scenario.

Usage: python3 additional_drift_scenarios.py <predictions_csv> [output_prefix]
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
AUDIT_RATE = 0.05
UGAA_K = 4.0


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
    def __init__(self, target_rate=AUDIT_RATE, sensitivity=UGAA_K, ema_alpha=0.02):
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


def select_target_category(shuffled, min_required_instances=int(DRIFT_BURST_LEN * 1.5)):
    cat_counts = shuffled["pred_category"].value_counts()
    eligible = cat_counts[cat_counts >= 20].index.tolist() or cat_counts.index.tolist()
    cat_accuracy = shuffled[shuffled["pred_category"].isin(eligible)].groupby("pred_category")["correct"].mean()
    ranked = cat_accuracy.sort_values(ascending=False)
    for cat in ranked.index:
        if cat_counts[cat] >= min_required_instances:
            return cat
    return None


def process_stream(engine, auditor, method, stream, rng, feedback_error_rate=0.08,
                    force_reliability=None, target_category=None):
    """Processes a stream of alerts. If force_reliability is set (a float in
    [0,1]) and target_category matches, the outcome is resampled to be
    correct with that probability instead of using the real label - this
    is how partial/gradual drift is injected, generalizing the original
    'always force incorrect' complete-collapse scenario."""
    detected, first_detection = 0, None
    for i, row in enumerate(stream):
        category = row["pred_category"]
        confidence = row["raw_confidence"]
        if force_reliability is not None and category == target_category:
            is_correct = rng.random() < force_reliability
        else:
            is_correct = bool(row["correct"])
        current_trust = engine.trust_for(category)
        decision = arbitration_decision(current_trust, confidence)

        if method == "fixed_audit":
            feedback_available = (decision == "escalate") or (rng.random() < AUDIT_RATE)
        elif method == "ugaa":
            audit_prob = auditor.audit_probability(engine, category)
            feedback_available = (decision == "escalate") or (rng.random() < audit_prob)
        else:
            raise ValueError(method)

        if feedback_available:
            observed = is_correct if rng.random() > feedback_error_rate else (not is_correct)
            engine.update(category, observed)
            if force_reliability is not None and category == target_category:
                detected += 1
                if first_detection is None:
                    first_detection = i
    return detected, first_detection


def build_pre_drift_state(preds, seed, pre_drift_n=1500):
    shuffled = preds.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    target_category = select_target_category(shuffled)
    if target_category is None:
        return None
    n = len(shuffled)
    pre_drift_n = min(pre_drift_n, max(n - DRIFT_BURST_LEN, 100))
    pre_drift_stream = shuffled.iloc[:pre_drift_n].to_dict("records")
    post_pool = shuffled.iloc[pre_drift_n:]
    return target_category, pre_drift_stream, post_pool


def scenario_a_partial_decline(preds, seed, method, decline_to=0.70):
    """Reliability drops to a non-zero floor (e.g. 70%) rather than 0%."""
    setup = build_pre_drift_state(preds, seed)
    if setup is None:
        return None
    target_category, pre_drift_stream, post_pool = setup
    rng = np.random.default_rng(seed)
    engine = TrustEngine(decay=DECAY)
    auditor = UncertaintyGuidedAuditor() if method == "ugaa" else None
    process_stream(engine, auditor, method, pre_drift_stream, rng)
    pre_trust = engine.trust_for(target_category)

    raw_prev = (post_pool["pred_category"] == target_category).mean()
    if raw_prev <= 0:
        return None
    window_size = min(int(np.ceil(DRIFT_BURST_LEN / raw_prev * 1.8)), len(post_pool))
    drift_window = post_pool.sample(n=window_size, replace=(window_size > len(post_pool)),
                                     random_state=seed + 1).to_dict("records")
    drift_events = [r for r in drift_window if r["pred_category"] == target_category][:DRIFT_BURST_LEN]
    other_events = [r for r in drift_window if r["pred_category"] != target_category]
    if len(drift_events) < DRIFT_BURST_LEN:
        return None
    interleaved = drift_events + other_events
    rng.shuffle(interleaved)

    detected, first_det = process_stream(engine, auditor, method, interleaved, rng,
                                          force_reliability=decline_to, target_category=target_category)
    post_trust = engine.trust_for(target_category)
    return {"pre_trust": float(pre_trust), "post_trust": float(post_trust),
            "coverage": detected / DRIFT_BURST_LEN, "first_detection": first_det}


def scenario_b_gradual_drift(preds, seed, method, levels=(0.95, 0.90, 0.80, 0.70, 0.60)):
    """Reliability steps down through several levels rather than jumping instantly."""
    setup = build_pre_drift_state(preds, seed)
    if setup is None:
        return None
    target_category, pre_drift_stream, post_pool = setup
    rng = np.random.default_rng(seed)
    engine = TrustEngine(decay=DECAY)
    auditor = UncertaintyGuidedAuditor() if method == "ugaa" else None
    process_stream(engine, auditor, method, pre_drift_stream, rng)
    pre_trust = engine.trust_for(target_category)

    raw_prev = (post_pool["pred_category"] == target_category).mean()
    if raw_prev <= 0:
        return None
    per_level_len = DRIFT_BURST_LEN // len(levels)
    total_needed = per_level_len * len(levels)
    window_size = min(int(np.ceil(total_needed / raw_prev * 2.0)), len(post_pool))
    drift_window = post_pool.sample(n=window_size, replace=(window_size > len(post_pool)),
                                     random_state=seed + 1)
    target_rows = drift_window[drift_window["pred_category"] == target_category].to_dict("records")
    other_rows = drift_window[drift_window["pred_category"] != target_category].to_dict("records")
    if len(target_rows) < total_needed:
        return None

    total_detected, first_det, event_idx = 0, None, 0
    for level in levels:
        level_events = target_rows[:per_level_len]
        target_rows = target_rows[per_level_len:]
        mix = level_events + other_rows[:len(level_events)]
        other_rows = other_rows[len(level_events):]
        rng.shuffle(mix)
        d, fd = process_stream(engine, auditor, method, mix, rng,
                                force_reliability=level, target_category=target_category)
        total_detected += d
        if first_det is None and fd is not None:
            first_det = event_idx + fd
        event_idx += len(level_events)

    post_trust = engine.trust_for(target_category)
    return {"pre_trust": float(pre_trust), "post_trust": float(post_trust),
            "coverage": total_detected / total_needed, "first_detection": first_det}


def scenario_c_two_simultaneous(preds, seed, method):
    """Two categories (one common, one rare) drift to 0% correctness concurrently."""
    shuffled = preds.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    cat_counts = shuffled["pred_category"].value_counts()
    if len(cat_counts) < 2:
        return None
    common_cat = cat_counts.idxmax()
    rare_eligible = cat_counts[(cat_counts >= DRIFT_BURST_LEN) & (cat_counts.index != common_cat)]
    if len(rare_eligible) == 0:
        return None
    rare_cat = rare_eligible.idxmin()

    # Reserve rows for the drift window FIRST, before building the pre-drift
    # stream, so a random pre-drift slice can never accidentally consume so
    # many of the rare category's (possibly few) total instances that too
    # few remain for the drift window - this guarantees availability
    # whenever the category has enough instances in the WHOLE dataset,
    # rather than hoping enough are left in an arbitrary post-slice.
    rare_pool = shuffled[shuffled["pred_category"] == rare_cat]
    common_pool = shuffled[shuffled["pred_category"] == common_cat]
    if len(rare_pool) < DRIFT_BURST_LEN or len(common_pool) < DRIFT_BURST_LEN:
        return None
    rng_reserve = np.random.default_rng(seed + 100)
    rare_reserved_idx = rng_reserve.choice(rare_pool.index, size=DRIFT_BURST_LEN, replace=False)
    common_reserved_idx = rng_reserve.choice(common_pool.index, size=DRIFT_BURST_LEN, replace=False)
    reserved_idx = set(rare_reserved_idx) | set(common_reserved_idx)

    remaining = shuffled[~shuffled.index.isin(reserved_idx)].reset_index(drop=True)
    n = len(remaining)
    pre_drift_n = min(1500, max(n - 200, 100))
    pre_drift_stream = remaining.iloc[:pre_drift_n].to_dict("records")
    other_post_pool = remaining.iloc[pre_drift_n:]

    rng = np.random.default_rng(seed)
    engine = TrustEngine(decay=DECAY)
    auditor = UncertaintyGuidedAuditor() if method == "ugaa" else None
    process_stream(engine, auditor, method, pre_drift_stream, rng)
    pre_trust_common = engine.trust_for(common_cat)
    pre_trust_rare = engine.trust_for(rare_cat)

    common_events = shuffled.loc[list(common_reserved_idx)].to_dict("records")
    rare_events = shuffled.loc[list(rare_reserved_idx)].to_dict("records")
    other_post_pool = other_post_pool[~other_post_pool["pred_category"].isin([common_cat, rare_cat])]
    other_sample_n = min(len(other_post_pool), DRIFT_BURST_LEN * 2)
    other_events = other_post_pool.sample(n=other_sample_n, random_state=seed + 1).to_dict("records") if other_sample_n > 0 else []

    common_idx_set = set(common_reserved_idx)
    rare_idx_set = set(rare_reserved_idx)
    detected_common, detected_rare = 0, 0
    interleaved = list(zip(common_reserved_idx, common_events)) + list(zip(rare_reserved_idx, rare_events)) + \
                  [(None, r) for r in other_events]
    rng.shuffle(interleaved)
    for idx, row in interleaved:
        category = row["pred_category"]
        confidence = row["raw_confidence"]
        force = (idx in common_idx_set) or (idx in rare_idx_set)
        is_correct = False if force else bool(row["correct"])
        current_trust = engine.trust_for(category)
        decision = arbitration_decision(current_trust, confidence)
        if method == "fixed_audit":
            feedback_available = (decision == "escalate") or (rng.random() < AUDIT_RATE)
        else:
            audit_prob = auditor.audit_probability(engine, category)
            feedback_available = (decision == "escalate") or (rng.random() < audit_prob)
        if feedback_available:
            engine.update(category, is_correct)
            if idx in common_idx_set:
                detected_common += 1
            elif idx in rare_idx_set:
                detected_rare += 1

    return {
        "common_category": common_cat, "rare_category": rare_cat,
        "pre_trust_common": float(pre_trust_common), "pre_trust_rare": float(pre_trust_rare),
        "post_trust_common": float(engine.trust_for(common_cat)),
        "post_trust_rare": float(engine.trust_for(rare_cat)),
        "coverage_common": detected_common / DRIFT_BURST_LEN,
        "coverage_rare": detected_rare / DRIFT_BURST_LEN,
    }


def run_scenario(name, fn, preds, **kwargs):
    print(f"\n{'=' * 70}\n{name}\n{'=' * 70}")
    results = {}
    for method in ["fixed_audit", "ugaa"]:
        seed_results = []
        for seed in range(N_SEEDS):
            r = fn(preds, seed, method, **kwargs)
            if r is not None:
                seed_results.append(r)
        if not seed_results:
            print(f"  {method}: [insufficient data]")
            results[method] = None
            continue
        if "coverage" in seed_results[0]:
            cov = [r["coverage"] for r in seed_results]
            print(f"  {method}: coverage={np.mean(cov):.4f} +/- {np.std(cov):.4f}  (n={len(seed_results)} seeds)")
            results[method] = {"mean_coverage": float(np.mean(cov)), "std_coverage": float(np.std(cov)),
                                "per_seed": seed_results}
        else:
            cov_c = [r["coverage_common"] for r in seed_results]
            cov_r = [r["coverage_rare"] for r in seed_results]
            print(f"  {method}: coverage_common={np.mean(cov_c):.4f}+/-{np.std(cov_c):.4f}  "
                  f"coverage_rare={np.mean(cov_r):.4f}+/-{np.std(cov_r):.4f}  (n={len(seed_results)} seeds)")
            results[method] = {"mean_coverage_common": float(np.mean(cov_c)),
                                "mean_coverage_rare": float(np.mean(cov_r)), "per_seed": seed_results}
    return results


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 additional_drift_scenarios.py <predictions_csv> [output_prefix]")
        sys.exit(1)
    preds_path = Path(sys.argv[1])
    prefix = sys.argv[2] if len(sys.argv) > 2 else preds_path.stem
    if not preds_path.exists():
        raise FileNotFoundError(f"Missing {preds_path}")
    preds = pd.read_csv(preds_path)
    print(f"Loaded {len(preds)} predictions from {preds_path}")

    all_results = {}
    all_results["scenario_a_partial_70pct"] = run_scenario(
        "SCENARIO A: Partial decline (reliability -> 70%)", scenario_a_partial_decline, preds, decline_to=0.70)
    all_results["scenario_a_partial_60pct"] = run_scenario(
        "SCENARIO A: Partial decline (reliability -> 60%)", scenario_a_partial_decline, preds, decline_to=0.60)
    all_results["scenario_b_gradual"] = run_scenario(
        "SCENARIO B: Gradual drift (95->90->80->70->60%)", scenario_b_gradual_drift, preds)
    all_results["scenario_c_two_simultaneous"] = run_scenario(
        "SCENARIO C: Two simultaneous categories (common + rare)", scenario_c_two_simultaneous, preds)

    with open(OUT_DIR / f"{prefix}_additional_drift_scenarios.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nSaved to {prefix}_additional_drift_scenarios.json in {OUT_DIR}")


if __name__ == "__main__":
    main()
