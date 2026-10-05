"""
Milestone 9 (completion): Static trust baseline.

This is the THIRD arm needed for a proper ablation. Until now we've only
compared:
  (a) adaptive per-category trust + fixed thresholds
  (b) adaptive per-category trust + QPSO-tuned thresholds

Both of those already use the adaptive trust engine - so they don't tell us
how much the CONTEXTUAL, PER-CATEGORY nature of the trust engine itself is
worth. This script adds the missing baseline:
  (c) ONE single global trust score for the whole model (no per-category
      breakdown at all) + the same fixed thresholds

Comparing (c) vs (a) isolates the value of contextual trust modeling.
Comparing (a) vs (b) isolates the value of the QPSO optimization layer.
Together, all three give a genuine three-way ablation.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = PROJECT_ROOT / "outputs"

HUMAN_ERROR_RATE = 0.08  # same as simulate_feedback.py, for a fair comparison
SEED = 42
DECAY = 0.995  # same decay as the adaptive trust engine

# Same fixed thresholds used in decision_arbitration.py, for a fair comparison
TRUST_HIGH, CONF_HIGH = 0.75, 0.80
TRUST_LOW, CONF_LOW = 0.60, 0.50
W_TRUST, W_CONF = 0.6, 0.4

LATENCY_AUTO_MS = 5
LATENCY_ESCALATE_MS = 4000


def simulate_human_feedback(is_correct, rng):
    if rng.random() < HUMAN_ERROR_RATE:
        return not is_correct
    return bool(is_correct)


def main():
    preds_path = OUT_DIR / "baseline_predictions.csv"
    if not preds_path.exists():
        raise FileNotFoundError(f"Run train_baseline.py first - missing {preds_path}")
    preds = pd.read_csv(preds_path)
    preds = preds.sample(frac=1.0, random_state=SEED).reset_index(drop=True)  # same shuffle as simulate_feedback.py

    # --- Compute ONE global Beta(a,b) trust score, ignoring category entirely ---
    rng = np.random.default_rng(SEED)
    a, b = 1.0, 1.0  # Beta(1,1) uniform prior, same as the adaptive engine's prior
    trust_history = []
    for step, row in preds.iterrows():
        a *= DECAY
        b *= DECAY
        model_was_correct = bool(row["correct"])
        human_feedback = simulate_human_feedback(model_was_correct, rng)
        if human_feedback:
            a += 1.0
        else:
            b += 1.0
        trust_history.append(a / (a + b))

    global_trust = a / (a + b)
    global_uncertainty = (a * b) / ((a + b) ** 2 * (a + b + 1))
    print(f"Static global trust score (whole model, no category breakdown): {global_trust:.4f}")
    print(f"Uncertainty: {global_uncertainty:.6f}")
    print("(Compare this ONE number to the per-category spread in trust_snapshot.json: "
          "dos~0.89, u2r~0.50-0.57 - a global score cannot see that spread at all.)\n")

    # --- Apply the SAME fixed-threshold arbitration logic, but every row uses
    #     the single global_trust value instead of a per-category lookup ---
    n = len(preds)
    confidence = preds["raw_confidence"].values
    correct = preds["correct"].values
    trust = np.full(n, global_trust)  # identical trust value for every single row, regardless of category

    fused_score = W_TRUST * trust + W_CONF * confidence
    auto_mask = (trust >= TRUST_HIGH) & (confidence >= CONF_HIGH)
    reject_mask = (~auto_mask) & ((trust <= TRUST_LOW) | (confidence <= CONF_LOW))
    escalate_mask = ~auto_mask & ~reject_mask

    auto_count = auto_mask.sum()
    silent_failures = (auto_mask & (correct == 0)).sum()
    silent_rate = silent_failures / max(auto_count, 1)
    escalation_rate = (reject_mask.sum() + escalate_mask.sum()) / n
    avg_latency = (auto_count * LATENCY_AUTO_MS + (n - auto_count) * LATENCY_ESCALATE_MS) / n

    print("Static-trust arbitration results (same fixed thresholds as the adaptive baseline):")
    print(f"  Auto-executed:        {auto_count} ({auto_count/n:.1%})")
    print(f"  Escalated/rejected:   {reject_mask.sum()+escalate_mask.sum()} ({(reject_mask.sum()+escalate_mask.sum())/n:.1%})")
    print(f"  Silent failure rate (within auto-executed): {silent_rate:.2%}")
    print(f"  Escalation rate: {escalation_rate:.2%}")
    print(f"  Avg latency: {avg_latency:.1f} ms")

    results = {
        "global_trust": float(global_trust),
        "global_uncertainty": float(global_uncertainty),
        "auto_execute_count": int(auto_count),
        "silent_failure_count": int(silent_failures),
        "silent_failure_rate": float(silent_rate),
        "escalation_rate": float(escalation_rate),
        "avg_latency_ms": float(avg_latency),
        "thresholds_used": {
            "trust_high": TRUST_HIGH, "conf_high": CONF_HIGH,
            "trust_low": TRUST_LOW, "conf_low": CONF_LOW,
            "w_trust": W_TRUST, "w_conf": W_CONF,
        },
    }
    with open(OUT_DIR / "static_trust_baseline_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {OUT_DIR / 'static_trust_baseline_results.json'}")

    # --- Print the three-way comparison table if the other two results exist ---
    qpso_path = OUT_DIR / "qpso_results.json"
    if qpso_path.exists():
        with open(qpso_path) as f:
            qpso = json.load(f)
        print("\n=== THREE-WAY ABLATION ===")
        print(f"{'System':<38}{'Silent fail rate':>18}{'Escalation rate':>18}{'Latency (ms)':>14}")
        print(f"{'(a) Static trust + fixed thresh':<38}{silent_rate:>18.4f}{escalation_rate:>18.4f}{avg_latency:>14.1f}")
        print(f"{'(b) Adaptive trust + fixed thresh':<38}"
              f"{qpso['baseline_metrics']['silent_failure_rate']:>18.4f}"
              f"{qpso['baseline_metrics']['escalation_rate']:>18.4f}"
              f"{qpso['baseline_metrics']['avg_latency_ms']:>14.1f}")
        print(f"{'(c) Adaptive trust + QPSO-tuned':<38}"
              f"{qpso['qpso_metrics']['silent_failure_rate']:>18.4f}"
              f"{qpso['qpso_metrics']['escalation_rate']:>18.4f}"
              f"{qpso['qpso_metrics']['avg_latency_ms']:>14.1f}")
    else:
        print(f"\n(Run quantum_optimizer.py to also see the full three-way ablation table here.)")


if __name__ == "__main__":
    main()
