"""
Experiment B: Static/global trust threshold sweep.

Addresses: "you selected a poor static-trust policy." This script:
  1. Sweeps trust_high across a wide range (other params held at standard
     defaults), producing a risk-vs-workload curve for static trust,
     analogous to the QPSO Pareto sweep already produced for adaptive trust.
  2. Runs QPSO on the static trust value itself, to find the BEST POSSIBLE
     static-trust policy - not a hand-picked one.

Because static trust is a single constant value (identical for every
alert), the trust condition has only two possible regimes as trust_high
varies: always-satisfied (degrading to confidence-only behavior) or
never-satisfied (100% escalation) - no continuous middle ground.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd

from trust_engine import AdaptiveTrustEngine

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = PROJECT_ROOT / "outputs"

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


def evaluate_policy_full(params, trust_arr, conf_arr, correct_arr):
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
    metrics = {"silent_failure_rate": float(silent_rate), "escalation_rate": float(escalation_rate),
               "auto_execute_rate": float(auto_count / n)}
    return fitness, metrics


def qpso_optimize(trust_arr, conf_arr, correct_arr, n_particles=25, n_iterations=40, seed=42):
    rng = np.random.default_rng(seed)
    dim = 5
    swarm = rng.uniform(LOWER, UPPER, size=(n_particles, dim))
    pbest = swarm.copy()
    pbest_fitness = np.array([evaluate_policy_full(p, trust_arr, conf_arr, correct_arr)[0] for p in swarm])
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
            fit, _ = evaluate_policy_full(swarm[i], trust_arr, conf_arr, correct_arr)
            if fit > pbest_fitness[i]:
                pbest[i], pbest_fitness[i] = swarm[i].copy(), fit
        if pbest_fitness.max() > gbest_fitness:
            gbest, gbest_fitness = pbest[np.argmax(pbest_fitness)].copy(), pbest_fitness.max()
    _, metrics = evaluate_policy_full(gbest, trust_arr, conf_arr, correct_arr)
    return gbest, metrics


def main():
    preds_path = OUT_DIR / "baseline_predictions.csv"
    if not preds_path.exists():
        raise FileNotFoundError(f"Run train_baseline.py first - missing {preds_path}")
    preds = pd.read_csv(preds_path).sample(frac=1.0, random_state=SEED).reset_index(drop=True)
    conf = preds["raw_confidence"].values
    correct = preds["correct"].values
    n = len(preds)

    a, b = 1.0, 1.0
    rng = np.random.default_rng(SEED)
    for _, row in preds.iterrows():
        a *= 0.995; b *= 0.995
        fb = simulate_human_feedback(bool(row["correct"]), rng)
        if fb: a += 1.0
        else: b += 1.0
    static_trust_value = a / (a + b)
    static_trust_arr = np.full(n, static_trust_value)
    print(f"Static (global) trust value: {static_trust_value:.4f}\n")

    print("=" * 70)
    print("PART 1: Sweep of trust_high (holding conf_high=0.80, trust_low=0.60, "
          "conf_low=0.50, w_trust=0.6)")
    print("=" * 70)
    sweep_results = []
    for trust_high in np.linspace(0.50, 0.98, 25):
        if 0.60 >= trust_high:
            continue
        params = [trust_high, 0.80, 0.60, 0.50, 0.6]
        _, metrics = evaluate_policy_full(params, static_trust_arr, conf, correct)
        if metrics is None:
            continue
        sweep_results.append({"trust_high": float(trust_high), **metrics})
        print(f"  trust_high={trust_high:.3f}: silent_fail={metrics['silent_failure_rate']:.4f}  "
              f"escalation={metrics['escalation_rate']:.4f}  auto_rate={metrics['auto_execute_rate']:.4f}")

    with open(OUT_DIR / "static_trust_threshold_sweep.json", "w") as f:
        json.dump(sweep_results, f, indent=2)
    print(f"\nSaved sweep to {OUT_DIR / 'static_trust_threshold_sweep.json'}")
    print(f"\nNote the step-function behavior: because static trust is a SINGLE constant "
          f"value ({static_trust_value:.4f}) applied to every alert, the trust condition "
          f"only has two regimes - always-satisfied (trust_high <= {static_trust_value:.4f}) "
          f"or never-satisfied (trust_high > {static_trust_value:.4f}).")

    print(f"\n{'='*70}")
    print("PART 2: Best possible static-trust policy (QPSO applied to the SAME 5D space)")
    print("=" * 70)
    best_params, best_metrics = qpso_optimize(static_trust_arr, conf, correct, seed=SEED)
    print(f"Best static-trust policy found:")
    print(f"  params: trust_high={best_params[0]:.3f}, conf_high={best_params[1]:.3f}, "
          f"trust_low={best_params[2]:.3f}, conf_low={best_params[3]:.3f}, w_trust={best_params[4]:.3f}")
    print(f"  silent_failure_rate={best_metrics['silent_failure_rate']:.4f}")
    print(f"  escalation_rate={best_metrics['escalation_rate']:.4f}")
    print(f"  auto_execute_rate={best_metrics['auto_execute_rate']:.4f}")

    confidence_only_note = OUT_DIR / "extended_baselines_results.json"
    if confidence_only_note.exists():
        ext = json.load(open(confidence_only_note))
        conf_only_esc = ext["results"]["confidence_only"]["escalation_rate"]
        print(f"\nFor comparison, the confidence-only baseline's escalation rate: {conf_only_esc:.4f}")
        print(f"If the best static-trust policy's escalation rate is close to this, that confirms "
              f"the structural prediction: an optimally-tuned static trust threshold can at best "
              f"degrade to behaving like confidence-only - it cannot outperform it, because a "
              f"constant trust value carries no information to differentiate decisions.")

    result = {
        "static_trust_value": static_trust_value,
        "sweep": sweep_results,
        "best_optimized_static_policy": {
            "params": dict(zip(["trust_high", "conf_high", "trust_low", "conf_low", "w_trust"], best_params.tolist())),
            "metrics": best_metrics,
        },
    }
    with open(OUT_DIR / "static_trust_sweep_full_results.json", "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved full results to {OUT_DIR / 'static_trust_sweep_full_results.json'}")


if __name__ == "__main__":
    main()
