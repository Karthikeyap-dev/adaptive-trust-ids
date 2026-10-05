"""
Three-way ablation on UNSW-NB15's XGBoost predictions - trust engine and
optimization code imported UNCHANGED from the RandomForest pipeline.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd

from trust_engine import AdaptiveTrustEngine

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = PROJECT_ROOT / "outputs"

AGENT_ID = "unsw_xgb_baseline"
HUMAN_ERROR_RATE = 0.08
SEED = 42
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


def main():
    preds_path = OUT_DIR / "unsw_xgb_baseline_predictions.csv"
    if not preds_path.exists():
        raise FileNotFoundError(f"Run unsw_xgb_baseline.py first - missing {preds_path}")
    preds = pd.read_csv(preds_path).sample(frac=1.0, random_state=SEED).reset_index(drop=True)
    n = len(preds)
    conf = preds["raw_confidence"].values
    correct = preds["correct"].values
    print(f"Loaded {n} XGBoost predictions on UNSW-NB15. Overall accuracy: {correct.mean():.4f}\n")

    rng = np.random.default_rng(SEED)
    engine = AdaptiveTrustEngine(decay=0.995)
    for step, row in preds.iterrows():
        fb = simulate_human_feedback(bool(row["correct"]), rng)
        engine.update(AGENT_ID, row["pred_category"], fb, step=step)
    snapshot = engine.snapshot()
    print("Adaptive trust scores per category (XGBoost, UNSW-NB15):")
    for row in sorted(snapshot, key=lambda r: -r["trust"]):
        print(f"  {row['category']:16s}  trust={row['trust']:.3f}  uncertainty={row['uncertainty']:.5f}")

    a, b = 1.0, 1.0
    rng2 = np.random.default_rng(SEED)
    for _, row in preds.iterrows():
        a *= 0.995; b *= 0.995
        fb = simulate_human_feedback(bool(row["correct"]), rng2)
        if fb: a += 1.0
        else: b += 1.0
    static_trust = a / (a + b)
    print(f"\nStatic (global) trust: {static_trust:.4f}")

    trust_by_cat = {r["category"]: r["trust"] for r in snapshot}
    adaptive_trust_arr = preds["pred_category"].map(trust_by_cat).fillna(0.5).values
    static_trust_arr = np.full(n, static_trust)
    fixed_params = [TRUST_HIGH, CONF_HIGH, TRUST_LOW, CONF_LOW, W_TRUST]

    _, static_metrics = evaluate_policy(fixed_params, static_trust_arr, conf, correct)
    _, adaptive_metrics = evaluate_policy(fixed_params, adaptive_trust_arr, conf, correct)
    print("\nRunning policy optimization...")
    qpso_metrics = qpso_optimize(adaptive_trust_arr, conf, correct, seed=SEED)

    print(f"\n{'='*70}\nTHREE-WAY ABLATION - XGBOOST on UNSW-NB15\n{'='*70}")
    print(f"{'System':<38}{'Silent fail rate':>18}{'Escalation rate':>18}")
    print(f"{'(a) Static trust + fixed thresh':<38}{static_metrics['silent_failure_rate']:>18.4f}{static_metrics['escalation_rate']:>18.4f}")
    print(f"{'(b) Adaptive trust + fixed thresh':<38}{adaptive_metrics['silent_failure_rate']:>18.4f}{adaptive_metrics['escalation_rate']:>18.4f}")
    print(f"{'(c) Adaptive trust + optimized':<38}{qpso_metrics['silent_failure_rate']:>18.4f}{qpso_metrics['escalation_rate']:>18.4f}")

    results = {
        "classifier": "XGBoost", "dataset": "UNSW-NB15",
        "static_trust_metrics": static_metrics, "adaptive_trust_metrics": adaptive_metrics,
        "qpso_metrics": qpso_metrics, "static_trust_value": static_trust,
    }
    with open(OUT_DIR / "unsw_xgb_ablation_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {OUT_DIR / 'unsw_xgb_ablation_results.json'}")


if __name__ == "__main__":
    main()
