"""
Multi-seed evaluation for Q1/Q2 rigor.

Reruns the full pipeline (baseline classifier -> trust simulation ->
static-trust arm -> adaptive-trust arm -> QPSO-tuned arm) across N
independent seeds, reports mean +/- std for every headline metric, and
runs a paired Wilcoxon signed-rank test between each pair of ablation arms
(static vs adaptive, adaptive vs QPSO-tuned) on silent-failure rate and
escalation rate.

Each seed varies: the RandomForest's internal randomness (bootstrap
sampling / feature subsampling), the simulated human-feedback stream's
randomness, and the QPSO optimizer's randomness. The official NSL-KDD
train/test split itself is NOT varied - that split is the standard
protocol, not something to randomize.

Runtime: ~10-40s per seed depending on your machine. 10 seeds is the
minimum a Q1/Q2 reviewer will expect; the script defaults to 10.
"""
from pathlib import Path
import json
import time
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import RandomForestClassifier

from data_loader import load_raw, encode_features
from trust_engine import AdaptiveTrustEngine
import quantum_optimizer as qo

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = PROJECT_ROOT / "outputs"

N_SEEDS = 10
HUMAN_ERROR_RATE = 0.08
TRUST_HIGH, CONF_HIGH, TRUST_LOW, CONF_LOW, W_TRUST = 0.75, 0.80, 0.60, 0.50, 0.6  # fixed-threshold arm


def simulate_human_feedback(is_correct, rng):
    if rng.random() < HUMAN_ERROR_RATE:
        return not is_correct
    return bool(is_correct)


def run_one_seed(seed, train_X, train_y, test_X, test_y, cat_encoder):
    rng = np.random.default_rng(seed)

    # --- Baseline classifier (varies via random_state) ---
    clf = RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=seed, class_weight="balanced_subsample")
    clf.fit(train_X, train_y)
    pred_proba = clf.predict_proba(test_X)
    pred_y = pred_proba.argmax(axis=1)
    accuracy = (pred_y == test_y).mean()

    preds_df = pd.DataFrame({
        "pred_category": [cat_encoder.classes_[i] for i in pred_y],
        "raw_confidence": pred_proba.max(axis=1),
        "correct": (pred_y == test_y).astype(int),
    }).sample(frac=1.0, random_state=seed).reset_index(drop=True)

    # --- Adaptive per-category trust simulation ---
    adaptive_engine = AdaptiveTrustEngine(decay=0.995)
    for step, row in preds_df.iterrows():
        fb = simulate_human_feedback(bool(row["correct"]), rng)
        adaptive_engine.update("rf", row["pred_category"], fb, step=step)

    # --- Static (global) trust simulation ---
    a, b = 1.0, 1.0
    rng2 = np.random.default_rng(seed)
    for _, row in preds_df.iterrows():
        a *= 0.995; b *= 0.995
        fb = simulate_human_feedback(bool(row["correct"]), rng2)
        if fb: a += 1.0
        else: b += 1.0
    static_trust = a / (a + b)

    def evaluate_arm(trust_lookup, params):
        trust_high, conf_high, trust_low, conf_low, w_trust = params
        trust = preds_df["pred_category"].map(trust_lookup).fillna(0.5).values if callable(trust_lookup) is False else None
        return trust

    n = len(preds_df)
    conf = preds_df["raw_confidence"].values
    correct = preds_df["correct"].values

    def arm_metrics(trust_arr, trust_high, conf_high, trust_low, conf_low):
        auto_mask = (trust_arr >= trust_high) & (conf >= conf_high)
        reject_mask = (~auto_mask) & ((trust_arr <= trust_low) | (conf <= conf_low))
        escalate_mask = ~auto_mask & ~reject_mask
        auto_count = auto_mask.sum()
        silent_fail = (auto_mask & (correct == 0)).sum()
        return {
            "silent_failure_rate": silent_fail / max(auto_count, 1),
            "escalation_rate": (reject_mask.sum() + escalate_mask.sum()) / n,
            "auto_execute_rate": auto_count / n,
        }

    # Arm A: static global trust
    static_trust_arr = np.full(n, static_trust)
    arm_static = arm_metrics(static_trust_arr, TRUST_HIGH, CONF_HIGH, TRUST_LOW, CONF_LOW)

    # Arm B: adaptive per-category trust, fixed thresholds
    adaptive_trust_arr = preds_df["pred_category"].map(
        lambda c: adaptive_engine.trust("rf", c)
    ).values
    arm_adaptive = arm_metrics(adaptive_trust_arr, TRUST_HIGH, CONF_HIGH, TRUST_LOW, CONF_LOW)

    # Arm C: adaptive trust + QPSO-tuned thresholds
    qpso_df = pd.DataFrame({"trust": adaptive_trust_arr, "confidence": conf, "correct": correct})
    qo.MAX_ACCEPTABLE_SILENT_RATE = 0.05
    best_params, best_fitness, _ = qo.qpso_optimize(qpso_df, n_particles=20, n_iterations=25, seed=seed)
    _, qpso_metrics_raw = qo.evaluate_policy(best_params, qpso_df)
    arm_qpso = {
        "silent_failure_rate": qpso_metrics_raw["silent_failure_rate"],
        "escalation_rate": qpso_metrics_raw["escalation_rate"],
        "auto_execute_rate": qpso_metrics_raw["auto_execute_count"] / n,
    }

    return {
        "seed": seed, "accuracy": accuracy,
        "static_silent_failure_rate": arm_static["silent_failure_rate"],
        "static_escalation_rate": arm_static["escalation_rate"],
        "adaptive_silent_failure_rate": arm_adaptive["silent_failure_rate"],
        "adaptive_escalation_rate": arm_adaptive["escalation_rate"],
        "qpso_silent_failure_rate": arm_qpso["silent_failure_rate"],
        "qpso_escalation_rate": arm_qpso["escalation_rate"],
    }


def main():
    print(f"Loading data (shared across all {N_SEEDS} seeds - only stochastic components vary)...")
    train_df = load_raw("train")
    test_df = load_raw("test")
    train_X, train_y, test_X, test_y, cat_encoder = encode_features(train_df, test_df)

    results = []
    for seed in range(N_SEEDS):
        t0 = time.time()
        r = run_one_seed(seed, train_X, train_y, test_X, test_y, cat_encoder)
        results.append(r)
        print(f"  seed {seed}: accuracy={r['accuracy']:.4f}  "
              f"static_esc={r['static_escalation_rate']:.3f}  "
              f"adaptive_esc={r['adaptive_escalation_rate']:.3f}  "
              f"qpso_esc={r['qpso_escalation_rate']:.3f}  ({time.time()-t0:.1f}s)")

    results_df = pd.DataFrame(results)
    results_df.to_csv(OUT_DIR / "multi_seed_results.csv", index=False)

    print(f"\n{'='*70}\nSUMMARY (mean +/- std across {N_SEEDS} seeds)\n{'='*70}")
    summary = {}
    for col in results_df.columns:
        if col == "seed":
            continue
        summary[col] = {"mean": float(results_df[col].mean()), "std": float(results_df[col].std())}
        print(f"  {col:35s}  {results_df[col].mean():.4f} +/- {results_df[col].std():.4f}")

    # --- Paired statistical tests (Wilcoxon signed-rank, since arms are paired by seed) ---
    print(f"\n{'='*70}\nPAIRED STATISTICAL TESTS (Wilcoxon signed-rank)\n{'='*70}")
    tests = {}
    for metric in ["silent_failure_rate", "escalation_rate"]:
        for pair_name, (a_col, b_col) in [
            ("static_vs_adaptive", (f"static_{metric}", f"adaptive_{metric}")),
            ("adaptive_vs_qpso", (f"adaptive_{metric}", f"qpso_{metric}")),
        ]:
            a_vals = results_df[a_col].values
            b_vals = results_df[b_col].values
            if np.allclose(a_vals, b_vals):
                stat, p = float("nan"), 1.0
            else:
                stat, p = stats.wilcoxon(a_vals, b_vals)
            key = f"{metric}__{pair_name}"
            tests[key] = {"statistic": float(stat), "p_value": float(p), "significant_at_0.05": bool(p < 0.05)}
            print(f"  {metric:22s} {pair_name:22s}  W={stat:.2f}  p={p:.4f}  "
                  f"{'*** significant' if p < 0.05 else 'not significant'}")

    with open(OUT_DIR / "multi_seed_summary.json", "w") as f:
        json.dump({"n_seeds": N_SEEDS, "summary": summary, "statistical_tests": tests}, f, indent=2)

    print(f"\nSaved per-seed results to {OUT_DIR / 'multi_seed_results.csv'}")
    print(f"Saved summary + tests to {OUT_DIR / 'multi_seed_summary.json'}")


if __name__ == "__main__":
    main()
