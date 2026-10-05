"""
Milestone 6: Decision arbitration.

Combines the Adaptive Trust Engine's per-category trust score with the
model's confidence into one fused score, then decides:
  - auto_execute : act on the AI's decision without human involvement
  - escalate     : send to a human analyst for review
  - reject       : trust/confidence both too low, discard/flag for audit

This version uses FIXED thresholds (deliberately naive) so we can measure
how many "silent failures" happen - cases auto-executed with high
confidence that were actually wrong. Milestone 7 (the quantum-inspired
optimizer) tunes these thresholds instead of leaving them fixed.
"""
from pathlib import Path
import json
import pandas as pd

from trust_engine import AdaptiveTrustEngine

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = PROJECT_ROOT / "outputs"

AGENT_ID = "rf_baseline"

# --- Fixed thresholds (intentionally naive - this is the baseline the
#     optimizer in Milestone 7 will improve on) ---
TRUST_HIGH = 0.75
CONF_HIGH = 0.80
TRUST_LOW = 0.60
CONF_LOW = 0.50

# Fusion weights: how much trust vs confidence matters in the combined score
W_TRUST = 0.6
W_CONF = 0.4

# Simulated cost proxies for evaluation (milliseconds) - not physically
# measured, just illustrative constants so we can report "average decision
# latency" as one of the framework's stated multi-objective metrics.
LATENCY_AUTO_MS = 5
LATENCY_ESCALATE_MS = 4000  # human review takes much longer


def arbitrate(trust, confidence):
    fused_score = W_TRUST * trust + W_CONF * confidence
    if trust >= TRUST_HIGH and confidence >= CONF_HIGH:
        return "auto_execute", fused_score
    if trust <= TRUST_LOW or confidence <= CONF_LOW:
        return "reject_or_escalate", fused_score
    return "escalate_review", fused_score


def main():
    preds_path = OUT_DIR / "baseline_predictions.csv"
    if not preds_path.exists():
        raise FileNotFoundError(f"Run train_baseline.py first - missing {preds_path}")

    preds = pd.read_csv(preds_path)

    # Rebuild trust snapshot the same way simulate_feedback.py did, so this
    # script can run standalone using the same trust scores already computed
    snapshot_path = OUT_DIR / "trust_snapshot.json"
    if not snapshot_path.exists():
        raise FileNotFoundError(f"Run simulate_feedback.py first - missing {snapshot_path}")
    with open(snapshot_path) as f:
        snapshot = json.load(f)
    trust_by_category = {row["category"]: row["trust"] for row in snapshot}

    decisions = []
    for _, row in preds.iterrows():
        category = row["pred_category"]
        confidence = row["raw_confidence"]
        trust = trust_by_category.get(category, 0.5)  # unseen category -> neutral prior
        decision, fused_score = arbitrate(trust, confidence)
        decisions.append({
            "true_category": row["true_category"],
            "pred_category": category,
            "trust": trust,
            "confidence": confidence,
            "fused_score": fused_score,
            "correct": row["correct"],
            "decision": decision,
        })

    df = pd.DataFrame(decisions)
    df.to_csv(OUT_DIR / "arbitration_decisions.csv", index=False)

    # --- Key metrics ---
    n = len(df)
    auto = df[df.decision == "auto_execute"]
    escalated = df[df.decision == "escalate_review"]
    rejected = df[df.decision == "reject_or_escalate"]

    silent_failures = auto[auto.correct == 0]
    escalation_rate = (len(escalated) + len(rejected)) / n
    avg_latency = (
        len(auto) * LATENCY_AUTO_MS +
        (len(escalated) + len(rejected)) * LATENCY_ESCALATE_MS
    ) / n

    print(f"Total decisions: {n}")
    print(f"  Auto-executed:        {len(auto)} ({len(auto)/n:.1%})")
    print(f"  Escalated for review: {len(escalated)} ({len(escalated)/n:.1%})")
    print(f"  Rejected/low-trust:   {len(rejected)} ({len(rejected)/n:.1%})")
    print()
    print(f"Silent failures (auto-executed but WRONG): {len(silent_failures)} "
          f"out of {len(auto)} auto-executed ({len(silent_failures)/max(len(auto),1):.2%})")
    print(f"Human escalation rate (workload proxy): {escalation_rate:.2%}")
    print(f"Average simulated decision latency: {avg_latency:.1f} ms")

    print("\nSilent failures broken down by true category (this is the risk the")
    print("fixed-threshold approach is missing - watch which categories dominate):")
    print(silent_failures["true_category"].value_counts())

    summary = {
        "total": n,
        "auto_execute_count": len(auto),
        "escalate_count": len(escalated),
        "reject_count": len(rejected),
        "silent_failure_count": len(silent_failures),
        "silent_failure_rate_within_auto": len(silent_failures) / max(len(auto), 1),
        "escalation_rate": escalation_rate,
        "avg_latency_ms": avg_latency,
        "thresholds_used": {
            "trust_high": TRUST_HIGH, "conf_high": CONF_HIGH,
            "trust_low": TRUST_LOW, "conf_low": CONF_LOW,
            "w_trust": W_TRUST, "w_conf": W_CONF,
        },
    }
    with open(OUT_DIR / "arbitration_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nSaved decisions to {OUT_DIR / 'arbitration_decisions.csv'}")
    print(f"Saved summary to {OUT_DIR / 'arbitration_summary.json'}")


if __name__ == "__main__":
    main()
