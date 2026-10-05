"""
Beta prior sensitivity analysis (Issue #34). Tests whether ablation
conclusions change under alternative Beta priors instead of Beta(1,1).

Usage: python3 prior_sensitivity.py <predictions_csv> [output_prefix]
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
FIXED_THRESHOLDS = (0.75, 0.80, 0.60, 0.50)
PRIORS = {
    "Beta(1,1) [primary]": (1.0, 1.0),
    "Beta(2,2)": (2.0, 2.0),
    "Beta(0.5,0.5) [Jeffreys]": (0.5, 0.5),
    "Beta(5,5)": (5.0, 5.0),
}


class TrustEngine:
    def __init__(self, prior_a, prior_b, decay=0.995):
        self.decay = decay
        self.prior_a = prior_a
        self.prior_b = prior_b
        self.state = {}

    def update(self, category, feedback_correct):
        a, b = self.state.get(category, (self.prior_a, self.prior_b))
        a *= self.decay
        b *= self.decay
        if feedback_correct:
            a += 1.0
        else:
            b += 1.0
        self.state[category] = (a, b)

    def trust_for(self, category):
        a, b = self.state.get(category, (self.prior_a, self.prior_b))
        return a / (a + b)


def simulate_feedback(is_correct, rng, error_rate=0.08):
    if rng.random() < error_rate:
        return not is_correct
    return bool(is_correct)


def evaluate_policy(params, trust_arr, conf_arr, correct_arr):
    trust_high, conf_high, trust_low, conf_low = params
    n = len(trust_arr)
    auto_mask = (trust_arr >= trust_high) & (conf_arr >= conf_high)
    reject_mask = (~auto_mask) & ((trust_arr <= trust_low) | (conf_arr <= conf_low))
    escalate_mask = ~auto_mask & ~reject_mask
    auto_count = auto_mask.sum()
    silent_fail = (auto_mask & (correct_arr == 0)).sum()
    silent_rate = silent_fail / max(auto_count, 1)
    escalation_rate = (reject_mask.sum() + escalate_mask.sum()) / n
    return {"silent_failure_rate": float(silent_rate), "escalation_rate": float(escalation_rate)}


def run_one_seed_one_prior(preds, seed, prior_a, prior_b):
    shuffled = preds.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    rng = np.random.default_rng(seed)
    engine = TrustEngine(prior_a, prior_b, decay=0.995)
    for _, row in shuffled.iterrows():
        fb = simulate_feedback(bool(row["correct"]), rng)
        engine.update(row["pred_category"], fb)

    trust_arr = shuffled["pred_category"].map(lambda c: engine.trust_for(c)).values
    conf_arr = shuffled["raw_confidence"].values
    correct_arr = shuffled["correct"].values
    metrics = evaluate_policy(FIXED_THRESHOLDS, trust_arr, conf_arr, correct_arr)
    per_cat_trust = {c: engine.trust_for(c) for c in engine.state.keys()}
    return metrics, per_cat_trust


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 prior_sensitivity.py <predictions_csv> [output_prefix]")
        sys.exit(1)
    preds_path = Path(sys.argv[1])
    prefix = sys.argv[2] if len(sys.argv) > 2 else preds_path.stem

    if not preds_path.exists():
        raise FileNotFoundError(f"Missing {preds_path}")
    preds = pd.read_csv(preds_path)
    print(f"Loaded {len(preds)} predictions from {preds_path}")
    print(f"Running prior sensitivity across {len(PRIORS)} priors, {N_SEEDS} seeds each\n")

    all_results = {}
    for prior_name, (a, b) in PRIORS.items():
        print(f"=== {prior_name} ===")
        seed_results = []
        last_per_cat = None
        for seed in range(N_SEEDS):
            metrics, per_cat = run_one_seed_one_prior(preds, seed, a, b)
            seed_results.append({"seed": seed, **metrics})
            last_per_cat = per_cat
        df = pd.DataFrame(seed_results)
        sf_mean, sf_std = df["silent_failure_rate"].mean(), df["silent_failure_rate"].std()
        esc_mean, esc_std = df["escalation_rate"].mean(), df["escalation_rate"].std()
        print(f"  silent_failure={sf_mean:.4f}+/-{sf_std:.4f}  escalation={esc_mean:.4f}+/-{esc_std:.4f}")
        print(f"  per-category trust (last seed): {last_per_cat}\n")

        all_results[prior_name] = {
            "prior_a": a, "prior_b": b,
            "silent_failure_rate_mean": float(sf_mean), "silent_failure_rate_std": float(sf_std),
            "escalation_rate_mean": float(esc_mean), "escalation_rate_std": float(esc_std),
            "example_per_category_trust": last_per_cat, "per_seed": seed_results,
        }

    print(f"{'='*70}\nSUMMARY TABLE (prior sensitivity)\n{'='*70}")
    print(f"{'Prior':<28}{'Silent fail':>14}{'Escalation':>14}")
    for name, r in all_results.items():
        print(f"{name:<28}{r['silent_failure_rate_mean']:>14.4f}{r['escalation_rate_mean']:>14.4f}")

    csv_rows = [{"prior": name, "silent_failure_mean": r["silent_failure_rate_mean"],
                 "silent_failure_std": r["silent_failure_rate_std"],
                 "escalation_mean": r["escalation_rate_mean"],
                 "escalation_std": r["escalation_rate_std"]} for name, r in all_results.items()]
    pd.DataFrame(csv_rows).to_csv(OUT_DIR / f"{prefix}_prior_sensitivity.csv", index=False)
    with open(OUT_DIR / f"{prefix}_prior_sensitivity.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved to {prefix}_prior_sensitivity.csv/.json in {OUT_DIR}")


if __name__ == "__main__":
    main()
