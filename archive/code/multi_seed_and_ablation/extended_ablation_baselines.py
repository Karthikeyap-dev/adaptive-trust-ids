"""
Extended arbitration baseline comparison - addresses the reviewer's
"Experiment 1" and "Experiment 2" (strongest, highest-priority critique).

Six non-redundant arms (mapping to the reviewer's 7-item list - "fixed
threshold" and "global trust" are the same underlying idea here: a trust
value that does not adapt, combined with non-optimized thresholds - so
they are consolidated into Arm 2 rather than duplicated):

  Arm 1: Confidence-only        - no trust concept at all; naive baseline
  Arm 2: Static/global trust    - trust doesn't adapt per category; fixed thresholds
  Arm 3: Adaptive trust (fixed) - per-category trust; hand-picked thresholds
  Arm 4: Adaptive + grid search - per-category trust; brute-force optimized thresholds
  Arm 5: Adaptive + classical PSO - per-category trust; standard velocity-based PSO
  Arm 6: Adaptive + QPSO        - per-category trust; quantum-inspired PSO

All six are evaluated against the IDENTICAL fitness function and the
IDENTICAL hard accuracy-floor constraint, so the comparison is fair -
this directly answers "does QPSO actually contribute, or would classical
optimization have done just as well?"
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
HUMAN_ERROR_RATE = 0.08
SEED = 42
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
    if violation > 0:
        fitness = -5.0 - violation
    else:
        fitness = 0.5 * (1 - silent_rate) + 0.35 * (1 - escalation_rate) + 0.15 * (1 - avg_latency / LATENCY_ESCALATE_MS)
    metrics = {
        "silent_failure_rate": float(silent_rate), "escalation_rate": float(escalation_rate),
        "avg_latency_ms": float(avg_latency), "auto_execute_count": int(auto_count), "fitness": float(fitness),
    }
    return fitness, metrics


def confidence_only_policy(conf_arr, correct_arr):
    n = len(conf_arr)
    best_fitness, best_thresh, best_metrics = -1e9, None, None
    for thresh in np.linspace(0.5, 0.99, 50):
        auto_mask = conf_arr >= thresh
        escalate_mask = ~auto_mask
        auto_count = auto_mask.sum()
        silent_fail = (auto_mask & (correct_arr == 0)).sum()
        silent_rate = silent_fail / max(auto_count, 1)
        escalation_rate = escalate_mask.sum() / n
        avg_latency = (auto_count * LATENCY_AUTO_MS + escalate_mask.sum() * LATENCY_ESCALATE_MS) / n
        violation = max(0.0, silent_rate - MAX_ACCEPTABLE_SILENT_RATE)
        fitness = (-5.0 - violation) if violation > 0 else (
            0.5 * (1 - silent_rate) + 0.35 * (1 - escalation_rate) + 0.15 * (1 - avg_latency / LATENCY_ESCALATE_MS))
        if fitness > best_fitness:
            best_fitness = fitness
            best_thresh = thresh
            best_metrics = {"silent_failure_rate": float(silent_rate), "escalation_rate": float(escalation_rate),
                             "avg_latency_ms": float(avg_latency), "auto_execute_count": int(auto_count),
                             "fitness": float(fitness), "confidence_threshold": float(thresh)}
    return best_metrics


def grid_search_policy(trust_arr, conf_arr, correct_arr, n_per_dim=6):
    best_fitness, best_params, best_metrics = -1e9, None, None
    grid = [np.linspace(LOWER[i], UPPER[i], n_per_dim) for i in range(5)]
    count = 0
    for th in grid[0]:
        for ch in grid[1]:
            for tl in grid[2]:
                for cl in grid[3]:
                    for w in grid[4]:
                        count += 1
                        fit, metrics = evaluate_policy([th, ch, tl, cl, w], trust_arr, conf_arr, correct_arr)
                        if metrics is not None and fit > best_fitness:
                            best_fitness, best_params, best_metrics = fit, [th, ch, tl, cl, w], metrics
    print(f"    grid search evaluated {count} candidate policies")
    return best_params, best_metrics


def classical_pso_optimize(trust_arr, conf_arr, correct_arr, n_particles=25, n_iterations=40, seed=42,
                            w=0.7, c1=1.5, c2=1.5):
    rng = np.random.default_rng(seed)
    dim = 5
    swarm = rng.uniform(LOWER, UPPER, size=(n_particles, dim))
    velocity = rng.uniform(-1, 1, size=(n_particles, dim)) * (UPPER - LOWER) * 0.1
    pbest = swarm.copy()
    pbest_fitness = np.array([evaluate_policy(p, trust_arr, conf_arr, correct_arr)[0] for p in swarm])
    gbest = pbest[np.argmax(pbest_fitness)].copy()
    gbest_fitness = pbest_fitness.max()

    for _ in range(n_iterations):
        r1 = rng.uniform(0, 1, size=(n_particles, dim))
        r2 = rng.uniform(0, 1, size=(n_particles, dim))
        velocity = w * velocity + c1 * r1 * (pbest - swarm) + c2 * r2 * (gbest - swarm)
        swarm = np.clip(swarm + velocity, LOWER, UPPER)
        for i in range(n_particles):
            fit, _ = evaluate_policy(swarm[i], trust_arr, conf_arr, correct_arr)
            if fit > pbest_fitness[i]:
                pbest[i], pbest_fitness[i] = swarm[i].copy(), fit
        if pbest_fitness.max() > gbest_fitness:
            gbest, gbest_fitness = pbest[np.argmax(pbest_fitness)].copy(), pbest_fitness.max()

    _, metrics = evaluate_policy(gbest, trust_arr, conf_arr, correct_arr)
    return gbest, metrics


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
    return gbest, metrics


def main():
    preds_path = OUT_DIR / "baseline_predictions.csv"
    if not preds_path.exists():
        raise FileNotFoundError(f"Run train_baseline.py first - missing {preds_path}")
    preds = pd.read_csv(preds_path)
    preds = preds.sample(frac=1.0, random_state=SEED).reset_index(drop=True)
    n = len(preds)
    conf = preds["raw_confidence"].values
    correct = preds["correct"].values

    print(f"Loaded {n} predictions. Overall accuracy: {correct.mean():.4f}\n")

    rng = np.random.default_rng(SEED)
    engine = AdaptiveTrustEngine(decay=0.995)
    for step, row in preds.iterrows():
        fb = simulate_human_feedback(bool(row["correct"]), rng)
        engine.update(AGENT_ID, row["pred_category"], fb, step=step)
    trust_by_cat = {r["category"]: r["trust"] for r in engine.snapshot()}
    adaptive_trust_arr = preds["pred_category"].map(trust_by_cat).fillna(0.5).values

    a, b = 1.0, 1.0
    rng2 = np.random.default_rng(SEED)
    for _, row in preds.iterrows():
        a *= 0.995; b *= 0.995
        fb = simulate_human_feedback(bool(row["correct"]), rng2)
        if fb: a += 1.0
        else: b += 1.0
    static_trust_arr = np.full(n, a / (a + b))

    fixed_params = [0.75, 0.80, 0.60, 0.50, 0.6]
    results = {}

    print("Arm 1: Confidence-only...")
    results["confidence_only"] = confidence_only_policy(conf, correct)

    print("Arm 2: Static/global trust + fixed thresholds...")
    _, results["static_trust"] = evaluate_policy(fixed_params, static_trust_arr, conf, correct)

    print("Arm 3: Adaptive trust + fixed (hand-picked) thresholds...")
    _, results["adaptive_fixed"] = evaluate_policy(fixed_params, adaptive_trust_arr, conf, correct)

    print("Arm 4: Adaptive trust + grid search (this takes a moment)...")
    grid_params, results["adaptive_grid"] = grid_search_policy(adaptive_trust_arr, conf, correct)

    print("Arm 5: Adaptive trust + classical PSO...")
    pso_params, results["adaptive_classical_pso"] = classical_pso_optimize(adaptive_trust_arr, conf, correct, seed=SEED)

    print("Arm 6: Adaptive trust + QPSO...")
    qpso_params, results["adaptive_qpso"] = qpso_optimize(adaptive_trust_arr, conf, correct, seed=SEED)

    print(f"\n{'='*78}\nEXTENDED ARBITRATION BASELINE COMPARISON\n{'='*78}")
    print(f"{'Arm':<32}{'Silent fail rate':>18}{'Escalation rate':>18}")
    for name, label in [
        ("confidence_only", "1. Confidence-only"), ("static_trust", "2. Static/global trust"),
        ("adaptive_fixed", "3. Adaptive trust (fixed)"), ("adaptive_grid", "4. Adaptive + grid search"),
        ("adaptive_classical_pso", "5. Adaptive + classical PSO"), ("adaptive_qpso", "6. Adaptive + QPSO"),
    ]:
        m = results[name]
        print(f"{label:<32}{m['silent_failure_rate']:>18.4f}{m['escalation_rate']:>18.4f}")

    output = {
        "results": results,
        "grid_search_params": dict(zip(["trust_high", "conf_high", "trust_low", "conf_low", "w_trust"], grid_params)),
        "classical_pso_params": dict(zip(["trust_high", "conf_high", "trust_low", "conf_low", "w_trust"], pso_params.tolist())),
        "qpso_params": dict(zip(["trust_high", "conf_high", "trust_low", "conf_low", "w_trust"], qpso_params.tolist())),
    }
    with open(OUT_DIR / "extended_baselines_results.json", "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {OUT_DIR / 'extended_baselines_results.json'}")

    print(f"\n--- Key comparison for the paper's 'why QPSO' question ---")
    esc_classical = results["adaptive_classical_pso"]["escalation_rate"]
    esc_qpso = results["adaptive_qpso"]["escalation_rate"]
    esc_grid = results["adaptive_grid"]["escalation_rate"]
    print(f"Grid search escalation:      {esc_grid:.4f}")
    print(f"Classical PSO escalation:    {esc_classical:.4f}")
    print(f"QPSO escalation:             {esc_qpso:.4f}")
    if abs(esc_qpso - esc_classical) < 0.005:
        print("QPSO and classical PSO reach essentially the same result on this problem - "
              "be prepared to report this honestly and soften the QPSO necessity claim accordingly.")
    else:
        print(f"QPSO {'outperforms' if esc_qpso < esc_classical else 'underperforms relative to'} "
              f"classical PSO by {abs(esc_qpso-esc_classical)*100:.2f} percentage points on escalation rate.")


if __name__ == "__main__":
    main()
