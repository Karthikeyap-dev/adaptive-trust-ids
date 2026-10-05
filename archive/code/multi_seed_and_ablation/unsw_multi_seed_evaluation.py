"""
Full 10-seed statistical validation for UNSW-NB15 - mirrors
multi_seed_evaluation.py's methodology for NSL-KDD, completing the
"10 seeds across all three datasets" item from the review.

Each seed varies: the RandomForest's internal randomness, the simulated
human-feedback stream's randomness, and QPSO's randomness. Data loading
happens ONCE; only retraining and re-simulating happen per seed.

Runtime warning: UNSW-NB15 is larger than NSL-KDD (194 encoded features
vs 122, more rows) - expect this to take noticeably longer than the
NSL-KDD multi-seed run.
"""
from pathlib import Path
import json
import time
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import RandomForestClassifier

from unsw_data_loader import load_raw, encode_features
from trust_engine import AdaptiveTrustEngine

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = PROJECT_ROOT / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

N_SEEDS = 10
HUMAN_ERROR_RATE = 0.08
TRUST_HIGH, CONF_HIGH, TRUST_LOW, CONF_LOW, W_TRUST = 0.75, 0.80, 0.60, 0.50, 0.6
MAX_ACCEPTABLE_SILENT_RATE = 0.05
LATENCY_AUTO_MS, LATENCY_ESCALATE_MS = 5, 4000
LOWER = np.array([0.55, 0.55, 0.05, 0.05, 0.0])
UPPER = np.array([0.95, 0.95, 0.65, 0.65, 1.0])


def simulate_human_feedback(is_correct, rng):
    if rng.random() < HUMAN_ERROR_RATE:
        return not is_correct
    return bool(is_correct)


def evaluate_policy(params, trust_arr, conf_arr, correct_arr):
    trust_high, conf_high, trust_low, conf_low, w_trust = params
    n = len(trust_arr)
    if trust_low >= trust_high or conf_low >= conf_high:
        return -10.0, None
    auto_mask = (trust_arr >= trust_high) & (conf_arr >= conf_high)
    reject_mask = (~auto_mask) & ((trust_arr <= trust_low) | (conf_arr <= conf_low))
    escalate_mask = ~auto_mask & ~reject_mask
    auto_count = auto_mask.sum()
    silent_fail = (auto_mask & (correct_arr == 0)).sum()
    silent_rate = silent_fail / max(auto_count, 1)
    escalation_rate = (reject_mask.sum() + escalate_mask.sum()) / n
    avg_latency = (auto_count * LATENCY_AUTO_MS + (n - auto_count) * LATENCY_ESCALATE_MS) / n
    violation = max(0.0, silent_rate - MAX_ACCEPTABLE_SILENT_RATE)
    fitness = (-5.0 - violation) if violation > 0 else (
        0.5 * (1 - silent_rate) + 0.35 * (1 - escalation_rate) + 0.15 * (1 - avg_latency / LATENCY_ESCALATE_MS))
    return fitness, {"silent_failure_rate": float(silent_rate), "escalation_rate": float(escalation_rate)}


def qpso_optimize(trust_arr, conf_arr, correct_arr, n_particles=25, n_iterations=40, seed=42):
    rng = np.random.default_rng(seed)
    dim = 5
    swarm = rng.uniform(LOWER, UPPER, size=(n_particles, dim))
    pbest = swarm.copy()
    pbest_fitness = np.array([evaluate_policy(p, trust_arr, conf_arr, correct_arr)[0] for p in swarm])
    gbest = pbest[np.argmax(pbest_fitness)].copy()
    gbest_fitness = pbest_fitness.max()
    for it in range(n_iterations):
        beta = 1.0 - 0.6 * (it / n_iterations)
        mbest = pbest.mean(axis=0)
        for i in range(n_particles):
            phi = rng.uniform(0, 1, size=dim)
            attractor = phi * pbest[i] + (1 - phi) * gbest
            u = rng.uniform(1e-6, 1.0, size=dim)
            sign = rng.choice([-1, 1], size=dim)
            swarm[i] = np.clip(attractor + sign * beta * np.abs(mbest - swarm[i]) * np.log(1.0 / u), LOWER, UPPER)
            fit, _ = evaluate_policy(swarm[i], trust_arr, conf_arr, correct_arr)
            if fit > pbest_fitness[i]:
                pbest[i], pbest_fitness[i] = swarm[i].copy(), fit
        if pbest_fitness.max() > gbest_fitness:
            gbest, gbest_fitness = pbest[np.argmax(pbest_fitness)].copy(), pbest_fitness.max()
    _, metrics = evaluate_policy(gbest, trust_arr, conf_arr, correct_arr)
    return metrics


def run_one_seed(seed, train_X, train_y, test_X, test_y, cat_encoder):
    rng = np.random.default_rng(seed)

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

    adaptive_engine = AdaptiveTrustEngine(decay=0.995)
    for step, row in preds_df.iterrows():
        fb = simulate_human_feedback(bool(row["correct"]), rng)
        adaptive_engine.update("unsw_rf", row["pred_category"], fb, step=step)
    trust_by_cat = {r["category"]: r["trust"] for r in adaptive_engine.snapshot()}
    adaptive_trust_arr = preds_df["pred_category"].map(trust_by_cat).fillna(0.5).values

    a, b = 1.0, 1.0
    rng2 = np.random.default_rng(seed)
    for _, row in preds_df.iterrows():
        a *= 0.995; b *= 0.995
        fb = simulate_human_feedback(bool(row["correct"]), rng2)
        if fb: a += 1.0
        else: b += 1.0
    static_trust_arr = np.full(len(preds_df), a / (a + b))

    conf = preds_df["raw_confidence"].values
    correct = preds_df["correct"].values
    fixed_params = [TRUST_HIGH, CONF_HIGH, TRUST_LOW, CONF_LOW, W_TRUST]

    _, static_metrics = evaluate_policy(fixed_params, static_trust_arr, conf, correct)
    _, adaptive_metrics = evaluate_policy(fixed_params, adaptive_trust_arr, conf, correct)
    qpso_metrics = qpso_optimize(adaptive_trust_arr, conf, correct, seed=seed)

    return {
        "seed": seed, "accuracy": accuracy,
        "static_silent_failure_rate": static_metrics["silent_failure_rate"],
        "static_escalation_rate": static_metrics["escalation_rate"],
        "adaptive_silent_failure_rate": adaptive_metrics["silent_failure_rate"],
        "adaptive_escalation_rate": adaptive_metrics["escalation_rate"],
        "qpso_silent_failure_rate": qpso_metrics["silent_failure_rate"],
        "qpso_escalation_rate": qpso_metrics["escalation_rate"],
    }


def main():
    print(f"Loading and encoding UNSW-NB15 (once - shared across all {N_SEEDS} seeds)...")
    train_df = load_raw("train")
    test_df = load_raw("test")
    train_X, train_y, test_X, test_y, cat_encoder = encode_features(train_df, test_df)
    print(f"Train: {train_X.shape}, Test: {test_X.shape}\n")

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
    results_df.to_csv(OUT_DIR / "unsw_multi_seed_results.csv", index=False)

    print(f"\n{'='*70}\nSUMMARY (mean +/- std across {N_SEEDS} seeds)\n{'='*70}")
    summary = {}
    for col in results_df.columns:
        if col == "seed":
            continue
        summary[col] = {"mean": float(results_df[col].mean()), "std": float(results_df[col].std())}
        print(f"  {col:35s}  {results_df[col].mean():.4f} +/- {results_df[col].std():.4f}")

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

    with open(OUT_DIR / "unsw_multi_seed_summary.json", "w") as f:
        json.dump({"n_seeds": N_SEEDS, "summary": summary, "statistical_tests": tests}, f, indent=2)

    print(f"\nSaved per-seed results to {OUT_DIR / 'unsw_multi_seed_results.csv'}")
    print(f"Saved summary + tests to {OUT_DIR / 'unsw_multi_seed_summary.json'}")


if __name__ == "__main__":
    main()
