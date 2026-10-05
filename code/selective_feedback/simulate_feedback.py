"""
Milestone 3b: Stream baseline predictions through the Trust Engine with
simulated human feedback (ground truth + configurable human error rate,
since real SOC analysts aren't available at this project scope).

Produces:
- outputs/trust_evolution.csv  (trust score per category at every step)
- outputs/trust_snapshot.json  (final trust + uncertainty per category)
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd

from trust_engine import AdaptiveTrustEngine

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = PROJECT_ROOT / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

AGENT_ID = "rf_baseline"
HUMAN_ERROR_RATE = 0.08  # simulated analyst is wrong ~8% of the time when giving feedback
SEED = 42


def simulate_human_feedback(is_correct, rng):
    """With probability HUMAN_ERROR_RATE, the simulated human's feedback
    flips the ground truth (mimicking an imperfect analyst)."""
    if rng.random() < HUMAN_ERROR_RATE:
        return not is_correct
    return bool(is_correct)


def main():
    pred_path = OUT_DIR / "baseline_predictions.csv"
    if not pred_path.exists():
        raise FileNotFoundError(
            f"Could not find {pred_path}\n"
            f"Run train_baseline.py first - it generates this file."
        )

    rng = np.random.default_rng(SEED)
    preds = pd.read_csv(pred_path)

    # Shuffle to simulate alerts arriving in a realistic, non-sorted-by-class order
    preds = preds.sample(frac=1.0, random_state=SEED).reset_index(drop=True)

    engine = AdaptiveTrustEngine(decay=0.995)

    for step, row in preds.iterrows():
        category = row["pred_category"]  # trust is tracked on what the AI claimed the category was
        model_was_correct = bool(row["correct"])
        human_feedback = simulate_human_feedback(model_was_correct, rng)
        engine.update(AGENT_ID, category, human_feedback, step=step)

    history_df = pd.DataFrame(engine.history)
    history_df.to_csv(OUT_DIR / "trust_evolution.csv", index=False)

    snapshot = engine.snapshot()
    with open(OUT_DIR / "trust_snapshot.json", "w") as f:
        json.dump(snapshot, f, indent=2)

    print(f"Simulated {len(preds)} feedback events (human error rate = {HUMAN_ERROR_RATE})\n")
    print("Final trust scores per category:")
    for row in sorted(snapshot, key=lambda r: -r["trust"]):
        print(f"  {row['category']:8s}  trust={row['trust']:.3f}  "
              f"uncertainty={row['uncertainty']:.5f}  evidence={row['evidence_count']:.0f}")

    print(f"\nReliability ranking (most to least trusted):")
    ranked = engine.rank_categories(AGENT_ID, preds["pred_category"].unique().tolist())
    print("  " + " > ".join(ranked))

    print(f"\nSaved trust evolution over time to {OUT_DIR / 'trust_evolution.csv'}")
    print(f"Saved final snapshot to {OUT_DIR / 'trust_snapshot.json'}")


if __name__ == "__main__":
    main()
