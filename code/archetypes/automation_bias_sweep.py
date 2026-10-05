"""
Automation-bias sensitivity sweep - varies the automation_bias parameter
continuously (0.0 to 1.0) while holding other archetype parameters
fixed, converting the discrete three-archetype comparison into a
continuous robustness analysis. Identifies the specific bias level at
which the arbitration mechanism transitions from safe to unsafe.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = PROJECT_ROOT / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
TARGET_N_ALERTS = 5000
TRUST_HIGH, CONF_HIGH, TRUST_LOW, CONF_LOW = 0.75, 0.80, 0.60, 0.50
MAX_ACCEPTABLE_SILENT_RATE = 0.05

BASE_ACCURACY = 0.75
DISTRUST_BIAS = 0.02
CONF_HIGH_THRESHOLD = 0.70
CONF_LOW_THRESHOLD = 0.30

AUTOMATION_BIAS_LEVELS = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]


def sweep_feedback(is_correct, ai_confidence, automation_bias, rng):
    r = rng.random()
    if r < automation_bias and ai_confidence >= CONF_HIGH_THRESHOLD:
        return True
    elif r < automation_bias + DISTRUST_BIAS and ai_confidence <= CONF_LOW_THRESHOLD:
        return False
    else:
        judged_correctly = rng.random() < BASE_ACCURACY
        return is_correct if judged_correctly else (not is_correct)


def evaluate_policy(trust_arr, conf_arr, correct_arr):
    auto_mask = (trust_arr >= TRUST_HIGH) & (conf_arr >= CONF_HIGH)
    n = len(trust_arr)
    reject_mask = (~auto_mask) & ((trust_arr <= TRUST_LOW) | (conf_arr <= CONF_LOW))
    escalate_mask = ~auto_mask & ~reject_mask
    auto_count = auto_mask.sum()
    silent_fail = (auto_mask & (correct_arr == 0)).sum()
    return {
        "silent_failure_rate": float(silent_fail / max(auto_count, 1)),
        "escalation_rate": float((reject_mask.sum() + escalate_mask.sum()) / n),
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


def run_sweep_point(automation_bias, preds, seed=SEED):
    rng = np.random.default_rng(seed)
    n_available = len(preds)
    if TARGET_N_ALERTS <= n_available:
        idx = rng.choice(n_available, size=TARGET_N_ALERTS, replace=False)
    else:
        idx = rng.choice(n_available, size=TARGET_N_ALERTS, replace=True)
    stream = preds.iloc[idx].reset_index(drop=True)

    engine = TrustEngine(decay=0.995)
    for _, row in stream.iterrows():
        cat = row["pred_category"]
        fb = sweep_feedback(bool(row["correct"]), float(row["raw_confidence"]), automation_bias, rng)
        engine.update(cat, fb)

    trust_arr = stream["pred_category"].map(lambda c: engine.trust_for(c)).values
    conf_arr = stream["raw_confidence"].values
    correct_arr = stream["correct"].values
    return evaluate_policy(trust_arr, conf_arr, correct_arr)


def main():
    configs = [
        ("NSL-KDD", OUT_DIR / "baseline_predictions.csv"),
        ("UNSW-NB15", OUT_DIR / "unsw_baseline_predictions.csv"),
        ("CICIDS2017", OUT_DIR / "cicids_baseline_predictions.csv"),
    ]

    all_results = {}
    for dataset_name, path in configs:
        if not path.exists():
            print(f"[skip] {dataset_name}: {path} not found")
            continue
        preds = pd.read_csv(path)
        print(f"\n{'='*60}\n{dataset_name}\n{'='*60}")
        results = []
        crossed_ceiling_at = None
        for bias in AUTOMATION_BIAS_LEVELS:
            m = run_sweep_point(bias, preds)
            results.append({"automation_bias": bias, **m})
            flag = ""
            if m["silent_failure_rate"] > MAX_ACCEPTABLE_SILENT_RATE:
                flag = "  *** EXCEEDS 5% SAFETY CEILING ***"
                if crossed_ceiling_at is None:
                    crossed_ceiling_at = bias
            print(f"  automation_bias={bias:.1f}: silent_fail={m['silent_failure_rate']:.4f}  "
                  f"escalation={m['escalation_rate']:.4f}{flag}")
        all_results[dataset_name] = {"sweep": results, "safety_transition_automation_bias": crossed_ceiling_at}
        if crossed_ceiling_at is not None:
            print(f"  --> Safety ceiling first exceeded at automation_bias={crossed_ceiling_at}")
        else:
            print(f"  --> Safety ceiling never exceeded across the tested range")

    with open(OUT_DIR / "automation_bias_sweep.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved to {OUT_DIR / 'automation_bias_sweep.json'}")


if __name__ == "__main__":
    main()
