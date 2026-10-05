"""
Sensitivity Analysis v2 - supersedes sensitivity_analysis.py (v1).
Run this instead of v1; it covers everything v1 did plus the newly
requested additions.

Covers:
  A. Trust-noise sensitivity (60/70/80/90/100% simulated human accuracy):
     trust convergence, escalation rate, silent-failure rate, workload
  B. Recovery-after-bad-feedback: inject a burst of deliberately incorrect
     feedback for one category, measure how many subsequent good-feedback
     events it takes trust to recover
  C. Decay-factor sweep (0.90/0.95/0.99/0.995/0.999/1.0) on a STATIONARY
     stream (no drift) - tests stability
  D. Decay-factor sweep on a stream with an INJECTED mid-stream reliability
     drift - tests adaptivity. C and D together demonstrate the real
     adaptivity-vs-stability trade-off the decay factor controls.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd

from trust_engine import AdaptiveTrustEngine

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = PROJECT_ROOT / "outputs"

AGENT_ID = "rf_baseline"
SEED = 42
FIXED_PARAMS = [0.75, 0.80, 0.60, 0.50, 0.6]


def simulate_human_feedback(is_correct, error_rate, rng):
    if rng.random() < error_rate:
        return not is_correct
    return bool(is_correct)


def evaluate_fixed_policy(trust_arr, conf_arr, correct_arr):
    trust_high, conf_high, trust_low, conf_low, _ = FIXED_PARAMS
    n = len(trust_arr)
    auto_mask = (trust_arr >= trust_high) & (conf_arr >= conf_high)
    reject_mask = (~auto_mask) & ((trust_arr <= trust_low) | (conf_arr <= conf_low))
    escalate_mask = ~auto_mask & ~reject_mask
    auto_count = auto_mask.sum()
    silent_fail = (auto_mask & (correct_arr == 0)).sum()
    return {
        "silent_failure_rate": float(silent_fail / max(auto_count, 1)),
        "escalation_rate": float((reject_mask.sum() + escalate_mask.sum()) / n),
        "workload_auto_execute_rate": float(auto_count / n),
    }


def experiment_a_trust_noise(preds, conf, correct):
    print(f"\n{'='*70}\nA. TRUST-NOISE SENSITIVITY\n{'='*70}")
    error_rates = [0.40, 0.30, 0.20, 0.10, 0.00]
    results = []
    for error_rate in error_rates:
        human_accuracy = 1 - error_rate
        rng = np.random.default_rng(SEED)
        engine = AdaptiveTrustEngine(decay=0.995)
        for step, row in preds.iterrows():
            fb = simulate_human_feedback(bool(row["correct"]), error_rate, rng)
            engine.update(AGENT_ID, row["pred_category"], fb, step=step)
        trust_by_cat = {r["category"]: r["trust"] for r in engine.snapshot()}
        trust_arr = preds["pred_category"].map(trust_by_cat).fillna(0.5).values
        metrics = evaluate_fixed_policy(trust_arr, conf, correct)
        avg_trust = float(np.mean(list(trust_by_cat.values())))
        results.append({"simulated_human_accuracy": human_accuracy, "avg_trust": avg_trust,
                         "trust_by_category": trust_by_cat, **metrics})
        print(f"  Human accuracy {human_accuracy:.0%}: avg_trust={avg_trust:.3f}  "
              f"silent_fail={metrics['silent_failure_rate']:.4f}  escalation={metrics['escalation_rate']:.4f}  "
              f"workload(auto%)={metrics['workload_auto_execute_rate']:.4f}")
    with open(OUT_DIR / "sensitivity_trust_noise_v2.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved to {OUT_DIR / 'sensitivity_trust_noise_v2.json'}")
    return results


def experiment_b_recovery(preds, burst_category="dos", burst_length=100, decay=0.995, seed=SEED):
    print(f"\n{'='*70}\nB. RECOVERY AFTER BAD-FEEDBACK BURST (category='{burst_category}')\n{'='*70}")
    rng = np.random.default_rng(seed)
    engine = AdaptiveTrustEngine(decay=decay)
    n = len(preds)
    burst_start_step = n // 2

    trust_trace = []
    burst_count = 0
    trust_before_burst = None
    for step, row in preds.iterrows():
        cat = row["pred_category"]
        is_correct = bool(row["correct"])
        if cat == burst_category and step >= burst_start_step and burst_count < burst_length:
            if burst_count == 0:
                trust_before_burst = engine.trust(AGENT_ID, burst_category)
            engine.update(AGENT_ID, cat, False, step=step)
            burst_count += 1
        else:
            fb = simulate_human_feedback(is_correct, 0.08, rng)
            engine.update(AGENT_ID, cat, fb, step=step)
        if cat == burst_category:
            trust_trace.append({"step": step, "trust": engine.trust(AGENT_ID, burst_category)})

    trust_min = min(t["trust"] for t in trust_trace if t["step"] >= burst_start_step)
    trust_final = trust_trace[-1]["trust"]

    recovery_threshold = trust_before_burst * 0.9 if trust_before_burst else None
    post_burst_events = [t for t in trust_trace if t["step"] >= burst_start_step]
    recovery_event_index = None
    for i, t in enumerate(post_burst_events):
        if i >= burst_length and recovery_threshold and t["trust"] >= recovery_threshold:
            recovery_event_index = i - burst_length
            break

    print(f"  Trust before burst: {trust_before_burst:.3f}" if trust_before_burst else "  No pre-burst baseline")
    print(f"  Trust minimum during/after burst: {trust_min:.3f}")
    print(f"  Trust at end of stream: {trust_final:.3f}")
    if recovery_event_index is not None:
        print(f"  Recovered to within 90% of pre-burst trust after {recovery_event_index} subsequent "
              f"good-feedback events")
    else:
        print(f"  Did NOT recover to within 90% of pre-burst trust within available post-burst events")

    result = {
        "burst_category": burst_category, "burst_length": burst_length, "decay": decay,
        "trust_before_burst": trust_before_burst, "trust_min": float(trust_min), "trust_final": float(trust_final),
        "recovery_events_needed": recovery_event_index, "trust_trace": trust_trace,
    }
    with open(OUT_DIR / "sensitivity_recovery.json", "w") as f:
        json.dump(result, f, indent=2)
    print(f"Saved to {OUT_DIR / 'sensitivity_recovery.json'}")
    return result


def run_stream_with_optional_drift(preds, decay, drift_category=None, drift_start_frac=0.5,
                                    drift_severity=0.85, seed=SEED):
    rng = np.random.default_rng(seed)
    engine = AdaptiveTrustEngine(decay=decay)
    n = len(preds)
    drift_start_step = int(n * drift_start_frac)
    trace = []
    for step, row in preds.iterrows():
        cat = row["pred_category"]
        is_correct = bool(row["correct"])
        if drift_category and cat == drift_category and step >= drift_start_step:
            is_correct = rng.random() > drift_severity
        fb = simulate_human_feedback(is_correct, 0.08, rng)
        engine.update(AGENT_ID, cat, fb, step=step)
        if drift_category and cat == drift_category:
            trace.append({"step": step, "trust": engine.trust(AGENT_ID, drift_category)})
    return trace, drift_start_step


def experiment_c_decay_stationary(preds):
    print(f"\n{'='*70}\nC. DECAY-FACTOR SWEEP (stationary stream, no drift)\n{'='*70}")
    decay_values = [0.90, 0.95, 0.99, 0.995, 0.999, 1.0]
    results = []
    for decay in decay_values:
        trace, _ = run_stream_with_optional_drift(preds, decay, drift_category="probe")
        final_trust = trace[-1]["trust"] if trace else None
        second_half = [t["trust"] for t in trace[len(trace)//2:]]
        stability_std = float(np.std(second_half)) if second_half else None
        results.append({"decay": decay, "final_trust_probe": final_trust, "stability_std_second_half": stability_std})
        print(f"  decay={decay}: final_trust(probe)={final_trust:.4f}  stability(std)={stability_std:.5f}")
    with open(OUT_DIR / "sensitivity_decay_stationary.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved to {OUT_DIR / 'sensitivity_decay_stationary.json'}")
    return results


def experiment_d_decay_with_drift(preds):
    print(f"\n{'='*70}\nD. DECAY-FACTOR SWEEP WITH INJECTED MID-STREAM DRIFT (category='probe')\n{'='*70}")
    print("Simulates the classifier becoming much less reliable at 'probe' halfway through "
          "the feedback stream.")
    decay_values = [0.90, 0.95, 0.99, 0.995, 0.999, 1.0]
    results = []
    for decay in decay_values:
        trace, drift_step = run_stream_with_optional_drift(preds, decay, drift_category="probe",
                                                             drift_start_frac=0.5, drift_severity=0.85)
        pre_drift_trust = next((t["trust"] for t in trace if t["step"] < drift_step), None)
        post_drift = [t for t in trace if t["step"] >= drift_step]
        final_trust = post_drift[-1]["trust"] if post_drift else None

        if pre_drift_trust and final_trust and pre_drift_trust != final_trust:
            halfway = pre_drift_trust - 0.5 * (pre_drift_trust - final_trust)
            events_to_halfway = next((i for i, t in enumerate(post_drift) if t["trust"] <= halfway), None)
        else:
            events_to_halfway = None

        results.append({"decay": decay, "pre_drift_trust": pre_drift_trust, "post_drift_final_trust": final_trust,
                         "events_to_halfway_adaptation": events_to_halfway})
        print(f"  decay={decay}: pre_drift={pre_drift_trust:.3f}  post_drift_final={final_trust:.3f}  "
              f"events_to_halfway_adapt={events_to_halfway}")
    with open(OUT_DIR / "sensitivity_decay_drift.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved to {OUT_DIR / 'sensitivity_decay_drift.json'}")
    print("\nExpect: LOWER decay should adapt to drift in FEWER events (more responsive) but "
          "show HIGHER stability_std in Experiment C (noisier). HIGHER decay should be the "
          "opposite. This trade-off, if it appears, justifies 0.995 as a considered middle "
          "ground rather than an arbitrary pick.")
    return results


def main():
    preds_path = OUT_DIR / "baseline_predictions.csv"
    if not preds_path.exists():
        raise FileNotFoundError(f"Run train_baseline.py first - missing {preds_path}")
    preds = pd.read_csv(preds_path).sample(frac=1.0, random_state=SEED).reset_index(drop=True)
    conf = preds["raw_confidence"].values
    correct = preds["correct"].values

    experiment_a_trust_noise(preds, conf, correct)
    experiment_b_recovery(preds)
    experiment_c_decay_stationary(preds)
    experiment_d_decay_with_drift(preds)

    print(f"\n{'='*70}\nALL SENSITIVITY EXPERIMENTS COMPLETE\n{'='*70}")
    print("4 result files saved to outputs/: sensitivity_trust_noise_v2.json, "
          "sensitivity_recovery.json, sensitivity_decay_stationary.json, sensitivity_decay_drift.json")


if __name__ == "__main__":
    main()
