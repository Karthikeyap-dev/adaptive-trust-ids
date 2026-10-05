"""
Reliability heterogeneity across all 6 dataset-classifier configurations.

The primary experiments computed H(a) = std of per-category trust only
for RandomForest across 3 datasets (Table 8). This extends the same
computation to the 3 XGBoost configurations.

Requires each dataset's XGBoost baseline_predictions.csv (columns:
pred_category, raw_confidence, correct - same format used throughout).

IMPORTANT: fill in STATIC_ESCALATION and ADAPTIVE_ESCALATION below with
your real Table 5 values for the three XGBoost rows before running the
benefit/correlation part - these are the two numbers whose difference
gives the "adaptive benefit" in percentage points, computed the same
way as the existing RandomForest rows in Table 8.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = PROJECT_ROOT / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42

# Fill in from your real Table 5 (already have RF rows; need XGB rows confirmed)
STATIC_ESCALATION = {
    ("NSL-KDD", "RandomForest"): 1.000,
    ("UNSW-NB15", "RandomForest"): 1.000,
    ("CICIDS2017", "RandomForest"): 0.0017,
    ("NSL-KDD", "XGBoost"): 0.525,      # bimodal - see manuscript note on this configuration
    ("UNSW-NB15", "XGBoost"): 1.000,
    ("CICIDS2017", "XGBoost"): 0.0007,
}
ADAPTIVE_ESCALATION = {
    ("NSL-KDD", "RandomForest"): 0.696,
    ("UNSW-NB15", "RandomForest"): 0.487,
    ("CICIDS2017", "RandomForest"): 0.0017,
    ("NSL-KDD", "XGBoost"): 0.642,
    ("UNSW-NB15", "XGBoost"): 0.446,
    ("CICIDS2017", "XGBoost"): 0.0020,
}


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


def simulate_feedback(is_correct, rng, error_rate=0.08):
    if rng.random() < error_rate:
        return not is_correct
    return bool(is_correct)


def compute_heterogeneity(preds_path, seed=SEED):
    preds = pd.read_csv(preds_path).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    rng = np.random.default_rng(seed)
    engine = TrustEngine(decay=0.995)
    for _, row in preds.iterrows():
        fb = simulate_feedback(bool(row["correct"]), rng)
        engine.update(row["pred_category"], fb)
    trust_values = [engine.trust_for(c) for c in engine.state.keys()]
    return float(np.std(trust_values)), {c: engine.trust_for(c) for c in engine.state.keys()}


def main():
    configs = [
        ("NSL-KDD", "RandomForest", OUT_DIR / "baseline_predictions.csv"),
        ("UNSW-NB15", "RandomForest", OUT_DIR / "unsw_baseline_predictions.csv"),
        ("CICIDS2017", "RandomForest", OUT_DIR / "cicids_baseline_predictions.csv"),
        ("NSL-KDD", "XGBoost", OUT_DIR / "xgb_baseline_predictions.csv"),
        ("UNSW-NB15", "XGBoost", OUT_DIR / "unsw_xgb_baseline_predictions.csv"),
        ("CICIDS2017", "XGBoost", OUT_DIR / "cicids_xgb_baseline_predictions.csv"),
    ]

    results = []
    print(f"{'Dataset':<14}{'Classifier':<14}{'Trust std (H)':>15}{'Static esc.':>13}{'Adaptive esc.':>15}{'Benefit (pp)':>14}")
    for dataset, clf, path in configs:
        if not path.exists():
            print(f"  [skip] {dataset}/{clf}: {path} not found")
            continue
        h, trust_dict = compute_heterogeneity(path)
        static_e = STATIC_ESCALATION.get((dataset, clf))
        adaptive_e = ADAPTIVE_ESCALATION.get((dataset, clf))
        benefit_pp = (static_e - adaptive_e) * 100 if (static_e is not None and adaptive_e is not None) else None
        benefit_str = f"{benefit_pp:.1f}" if benefit_pp is not None else "N/A"
        print(f"{dataset:<14}{clf:<14}{h:>15.4f}{static_e:>13.4f}{adaptive_e:>15.4f}{benefit_str:>14}")
        results.append({
            "dataset": dataset, "classifier": clf, "heterogeneity_std": h,
            "static_escalation": static_e, "adaptive_escalation": adaptive_e,
            "adaptive_benefit_pp": benefit_pp, "per_category_trust": trust_dict,
        })

    with open(OUT_DIR / "heterogeneity_all_six_configs.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {OUT_DIR / 'heterogeneity_all_six_configs.json'}")

    valid = [r for r in results if r["adaptive_benefit_pp"] is not None]
    if len(valid) >= 4:
        hs = [r["heterogeneity_std"] for r in valid]
        bs = [r["adaptive_benefit_pp"] for r in valid]
        r_val = np.corrcoef(hs, bs)[0, 1]
        print(f"\nDescriptive correlation across {len(valid)} configurations: r={r_val:.3f}")
        print("(Still reported as descriptive/qualitative given small n, per the existing caveat in the paper.)")


if __name__ == "__main__":
    main()
