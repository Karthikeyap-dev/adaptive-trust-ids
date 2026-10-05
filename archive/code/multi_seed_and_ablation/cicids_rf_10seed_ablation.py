"""
Full 10-seed three-way ablation for CICIDS2017/RandomForest - closes
the one statistical-depth asymmetry in the primary experiments.

WARNING: CICIDS2017 is the largest dataset (~500K test rows) - this is
why it was skipped originally. Expect a long runtime.

Requires cicids_baseline_predictions.csv (columns: pred_category,
raw_confidence, correct).
"""
from pathlib import Path
import json
import time
import numpy as np
import pandas as pd
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = PROJECT_ROOT / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

N_SEEDS = 10
MAX_ACCEPTABLE_SILENT_RATE = 0.05
LATENCY_AUTO_MS, LATENCY_ESCALATE_MS = 5, 4000
LOWER = np.array([0.55, 0.55, 0.05, 0.05])
UPPER = np.array([0.95, 0.95, 0.65, 0.65])

STATIC_TRUST_VALUE = 0.6748
FIXED_THRESHOLDS = (0.75, 0.80, 0.60, 0.50)


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


def evaluate_policy(params, trust_arr, conf_arr, correct_arr):
    trust_high, conf_high, trust_low, conf_low = params
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
    metrics = {"silent_failure_rate": float(silent_rate), "escalation_rate": float(escalation_rate)}
    if violation > 0:
        return -5.0 - violation, metrics
    fitness = 0.5 * (1 - silent_rate) + 0.35 * (1 - escalation_rate) + 0.15 * (1 - avg_latency / LATENCY_ESCALATE_MS)
    return fitness, metrics


def qpso_optimize(trust_arr, conf_arr, correct_arr, n_particles=25, n_iterations=40, seed=42):
    rng = np.random.default_rng(seed)
    dim = 4
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


def run_one_seed(preds, seed):
    rng = np.random.default_rng(seed)
    shuffled = preds.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    engine = TrustEngine(decay=0.995)
    for _, row in shuffled.iterrows():
        fb = simulate_feedback(bool(row["correct"]), rng)
        engine.update(row["pred_category"], fb)
    adaptive_trust_arr = shuffled["pred_category"].map(lambda c: engine.trust_for(c)).values
    conf_arr = shuffled["raw_confidence"].values
    correct_arr = shuffled["correct"].values

    static_trust_arr = np.full(len(shuffled), STATIC_TRUST_VALUE)

    _, static_metrics = evaluate_policy(FIXED_THRESHOLDS, static_trust_arr, conf_arr, correct_arr)
    _, adaptive_fixed_metrics = evaluate_policy(FIXED_THRESHOLDS, adaptive_trust_arr, conf_arr, correct_arr)
    best_params, adaptive_opt_metrics = qpso_optimize(adaptive_trust_arr, conf_arr, correct_arr, seed=seed)

    return {
        "seed": seed,
        "static_silent_failure_rate": static_metrics["silent_failure_rate"],
        "static_escalation_rate": static_metrics["escalation_rate"],
        "adaptive_silent_failure_rate": adaptive_fixed_metrics["silent_failure_rate"],
        "adaptive_escalation_rate": adaptive_fixed_metrics["escalation_rate"],
        "optimized_silent_failure_rate": adaptive_opt_metrics["silent_failure_rate"],
        "optimized_escalation_rate": adaptive_opt_metrics["escalation_rate"],
    }


def main():
    preds_path = OUT_DIR / "cicids_baseline_predictions.csv"
    if not preds_path.exists():
        raise FileNotFoundError(f"Missing {preds_path}")
    preds = pd.read_csv(preds_path)
    print(f"Loaded {len(preds)} CICIDS2017 predictions. Running {N_SEEDS}-seed ablation "
          f"(large dataset - expect a long runtime)...\n")

    results = []
    for seed in range(N_SEEDS):
        t0 = time.time()
        r = run_one_seed(preds, seed)
        elapsed = time.time() - t0
        print(f"  seed {seed}: static_esc={r['static_escalation_rate']:.4f}  "
              f"adaptive_esc={r['adaptive_escalation_rate']:.4f}  "
              f"optimized_esc={r['optimized_escalation_rate']:.4f}  ({elapsed:.1f}s)")
        results.append(r)

    df = pd.DataFrame(results)
    df.to_csv(OUT_DIR / "cicids_rf_10seed_results.csv", index=False)

    print(f"\n{'='*70}\nSUMMARY (mean +/- SD across {N_SEEDS} seeds)\n{'='*70}")
    for arm in ["static", "adaptive", "optimized"]:
        sf = df[f"{arm}_silent_failure_rate"]
        esc = df[f"{arm}_escalation_rate"]
        print(f"{arm:>10}: silent_failure={sf.mean():.4f}+/-{sf.std():.4f}  escalation={esc.mean():.4f}+/-{esc.std():.4f}")

    print("\nPaired significance tests:")
    for pair in [("static", "adaptive"), ("adaptive", "optimized")]:
        a, b = pair
        for metric in ["silent_failure_rate", "escalation_rate"]:
            x, y = df[f"{a}_{metric}"], df[f"{b}_{metric}"]
            if np.allclose(x, y):
                print(f"  {a} vs {b} ({metric}): identical across all seeds")
                continue
            try:
                w, p = stats.wilcoxon(x, y)
                print(f"  {a} vs {b} ({metric}): Wilcoxon p={p:.4f}")
            except ValueError as e:
                print(f"  {a} vs {b} ({metric}): {e}")

    print(f"\nSaved per-seed results to {OUT_DIR / 'cicids_rf_10seed_results.csv'}")


if __name__ == "__main__":
    main()
