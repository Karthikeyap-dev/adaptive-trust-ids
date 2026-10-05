"""
10-seed corrected exhaustive grid search, all six dataset-classifier
configurations. Extends run_all_six.py: instead of loading each RF
config's pre-saved static trust_snapshot.json (single realization), trust
is now computed FRESH for all six configs, varying the feedback-simulation
seed 0-9 -- matching the N_SEEDS=10 convention from classical_vs_qpso_10seed.py.
Grid search itself is deterministic; the only seed-to-seed variance is in
which simulated feedback events land correct/incorrect, which shapes the
trust engine's alpha/beta accumulation.

Verified identical simulation procedure across all six source ablation
scripts (simulate_feedback.py, unsw_ablation.py, cicids_ablation.py,
xgb_ablation.py, unsw_xgb_ablation.py, cicids_xgb_ablation.py):
AdaptiveTrustEngine(decay=0.995), HUMAN_ERROR_RATE=0.08, only AGENT_ID and
the seed differ.

Runtime note: CICIDS2017's ~500K-row grid search takes ~40s per seed: with
10 seeds x 2 classifiers, CICIDS2017 alone takes roughly 13-14 minutes.
NSL-KDD and UNSW-NB15 are much faster (~1-4s/seed). Full six-config,
10-seed run: approximately 15-20 minutes total.
"""
from pathlib import Path
import json
import time
import numpy as np
import pandas as pd

from threshold_grid_search import evaluate_policy, vectorized_grid_search

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = PROJECT_ROOT / "outputs"

N_SEEDS = 10
HUMAN_ERROR_RATE = 0.08
DECAY = 0.995


class AdaptiveTrustEngine:
    """Matches src/trust_engine.py exactly."""
    def __init__(self, decay=0.98, prior_a=1.0, prior_b=1.0):
        self.decay = decay
        self.prior_a = prior_a
        self.prior_b = prior_b
        self.store = {}

    def update(self, agent_id, category, human_confirms_correct, step=None):
        a, b = self.store.get((agent_id, category), [self.prior_a, self.prior_b])
        a *= self.decay
        b *= self.decay
        if human_confirms_correct:
            a += 1.0
        else:
            b += 1.0
        self.store[(agent_id, category)] = [a, b]

    def snapshot(self):
        rows = []
        for (agent_id, category), (a, b) in self.store.items():
            rows.append({"agent_id": agent_id, "category": category, "trust": a / (a + b)})
        return rows


def simulate_human_feedback(is_correct, rng):
    if rng.random() < HUMAN_ERROR_RATE:
        return not is_correct
    return bool(is_correct)


def compute_trust_for_seed(preds, agent_id, seed):
    # Matches every one of the six source ablation scripts exactly: alerts
    # are shuffled with the SAME seed before the sequential trust simulation
    # (this was missing in an earlier version of this script -- confirmed
    # by comparing against the pre-saved unsw_trust_snapshot.json, which
    # differed until this shuffle was added).
    preds = preds.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    rng = np.random.default_rng(seed)
    engine = AdaptiveTrustEngine(decay=DECAY)
    for step, row in preds.iterrows():
        fb = simulate_human_feedback(bool(row["correct"]), rng)
        engine.update(agent_id, row["pred_category"], fb, step=step)
    return {r["category"]: r["trust"] for r in engine.snapshot()}


# (dataset, classifier): (predictions_csv, agent_id) -- agent_id verified
# from each config's own ablation script (all six confirmed to use the
# identical AdaptiveTrustEngine/simulate_human_feedback procedure).
CONFIGS = {
    ("NSL-KDD", "RandomForest"):    ("baseline_predictions.csv", "rf_baseline"),
    ("NSL-KDD", "XGBoost"):         ("xgb_baseline_predictions.csv", "xgb_baseline"),
    ("UNSW-NB15", "RandomForest"):  ("unsw_baseline_predictions.csv", "unsw_rf_baseline"),
    ("UNSW-NB15", "XGBoost"):       ("unsw_xgb_baseline_predictions.csv", "unsw_xgb_baseline"),
    ("CICIDS2017", "RandomForest"):("cicids_baseline_predictions.csv", "cicids_rf_baseline"),
    ("CICIDS2017", "XGBoost"):     ("cicids_xgb_baseline_predictions.csv", "cicids_xgb_baseline"),
}

# Real confidence-only mean thresholds, from confidence_only_<config>.csv
CONF_ONLY_THRESH = {
    ("NSL-KDD", "RandomForest"): 0.9662,
    ("NSL-KDD", "XGBoost"): 0.9988,
    ("UNSW-NB15", "RandomForest"): 0.7966,
    ("UNSW-NB15", "XGBoost"): 0.9033,
    ("CICIDS2017", "RandomForest"): 0.5003,
    ("CICIDS2017", "XGBoost"): 0.5003,
}


def run_config(dataset, clf, preds_file, agent_id, n_seeds=N_SEEDS, grid_step=0.005):
    preds_path = OUT_DIR / preds_file
    if not preds_path.exists():
        print(f"[skip] {dataset}/{clf}: {preds_path} not found")
        return None

    preds = pd.read_csv(preds_path)
    conf_arr = preds["raw_confidence"].values
    correct_arr = preds["correct"].values
    gamma_conf = CONF_ONLY_THRESH[(dataset, clf)]

    per_seed = []
    t_config_start = time.time()
    for seed in range(n_seeds):
        trust_by_cat = compute_trust_for_seed(preds, agent_id, seed)
        trust_arr = preds["pred_category"].map(trust_by_cat).fillna(0.5).values

        grid_result = vectorized_grid_search(trust_arr, conf_arr, correct_arr, delta=0.05, step=grid_step)
        per_seed.append({
            "seed": seed,
            "tau_h": grid_result["tau_h"], "gamma_h": grid_result["gamma_h"],
            "silent_failure_rate": grid_result["silent_failure_rate"],
            "escalation_rate": grid_result["escalation_rate"],
        })
        print(f"  [{dataset}/{clf}] seed {seed}: tau_h={grid_result['tau_h']:.3f} "
              f"e={grid_result['escalation_rate']:.4f} ({grid_result['elapsed_seconds']:.1f}s)")

    df = pd.DataFrame(per_seed)
    s_conf, e_conf = evaluate_policy(0.0, gamma_conf, preds["pred_category"].map(
        compute_trust_for_seed(preds, agent_id, 0)).fillna(0.5).values, conf_arr, correct_arr)

    result = {
        "dataset": dataset, "classifier": clf, "n_seeds": n_seeds,
        "confidence_only": {"gamma_h": gamma_conf, "s": float(s_conf), "e": float(e_conf)},
        "grid_tuned": {
            "escalation_mean": float(df["escalation_rate"].mean()),
            "escalation_std": float(df["escalation_rate"].std()),
            "silent_failure_mean": float(df["silent_failure_rate"].mean()),
            "silent_failure_std": float(df["silent_failure_rate"].std()),
            "tau_h_mean": float(df["tau_h"].mean()),
            "tau_h_std": float(df["tau_h"].std()),
        },
        "per_seed": per_seed,
        "total_elapsed_seconds": time.time() - t_config_start,
    }
    print(f"  --> {dataset}/{clf}: escalation {df['escalation_rate'].mean():.4f}"
          f"+/-{df['escalation_rate'].std():.4f}  "
          f"(tau_h {df['tau_h'].mean():.3f}+/-{df['tau_h'].std():.3f}, "
          f"total {result['total_elapsed_seconds']:.1f}s)\n")
    return result


def main(n_seeds=N_SEEDS, grid_step=0.005):
    t_start = time.time()
    all_results = []
    for (dataset, clf), (preds_file, agent_id) in CONFIGS.items():
        print(f"\n=== {dataset} / {clf} ===")
        r = run_config(dataset, clf, preds_file, agent_id, n_seeds=n_seeds, grid_step=grid_step)
        if r is not None:
            all_results.append(r)

    print(f"\n\n{'='*100}")
    print(f"SUMMARY -- {n_seeds}-seed corrected grid search, all six configurations")
    print(f"{'='*100}")
    print(f"{'Config':<28}{'Conf-only e':>13}{'Grid-tuned e (mean+/-sd)':>28}{'tau_h (mean+/-sd)':>20}")
    for r in all_results:
        cfg = f"{r['dataset']}/{r['classifier']}"
        gt = r['grid_tuned']
        esc_str = f"{gt['escalation_mean']:.4f}+/-{gt['escalation_std']:.4f}"
        tau_str = f"{gt['tau_h_mean']:.3f}+/-{gt['tau_h_std']:.3f}"
        print(f"{cfg:<28}{r['confidence_only']['e']:>13.4f}{esc_str:>28}{tau_str:>20}")

    with open(OUT_DIR / "corrected_grid_search_10seed_all_six.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nTotal runtime: {(time.time()-t_start)/60:.1f} minutes")
    print(f"Saved to {OUT_DIR / 'corrected_grid_search_10seed_all_six.json'}")


if __name__ == "__main__":
    import sys
    n_seeds = int(sys.argv[1]) if len(sys.argv) > 1 else N_SEEDS
    step = float(sys.argv[2]) if len(sys.argv) > 2 else 0.005
    main(n_seeds=n_seeds, grid_step=step)
