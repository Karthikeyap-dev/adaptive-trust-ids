"""
Locked-evaluation arbitration ablation - fixes the policy/test-set
leakage concern (Issue #1). Splits each configuration's predictions
into policy_dev (used ONLY for feedback simulation + QPSO optimization)
and locked_eval (untouched during optimization, evaluated exactly once
for the reported numbers).

Usage: python3 locked_eval_ablation.py <predictions_csv> [output_prefix]
Example: python3 locked_eval_ablation.py ../outputs/baseline_predictions.csv nslkdd_rf

Prints all results to console AND saves to CSV/JSON.
"""
from pathlib import Path
import json
import sys
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
    return gbest


def run_one_seed(preds, seed):
    shuffled = preds.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    n = len(shuffled)
    split_point = n // 2
    policy_dev = shuffled.iloc[:split_point].reset_index(drop=True)
    locked_eval = shuffled.iloc[split_point:].reset_index(drop=True)

    rng_dev = np.random.default_rng(seed)
    engine = TrustEngine(decay=0.995)
    for _, row in policy_dev.iterrows():
        fb = simulate_feedback(bool(row["correct"]), rng_dev)
        engine.update(row["pred_category"], fb)

    dev_trust_arr = policy_dev["pred_category"].map(lambda c: engine.trust_for(c)).values
    dev_conf_arr = policy_dev["raw_confidence"].values
    dev_correct_arr = policy_dev["correct"].values

    optimized_params = qpso_optimize(dev_trust_arr, dev_conf_arr, dev_correct_arr, seed=seed)

    static_engine = TrustEngine(decay=0.995)
    for _, row in policy_dev.iterrows():
        fb = simulate_feedback(bool(row["correct"]), rng_dev)
        static_engine.update("__global__", fb)
    static_trust_value = static_engine.trust_for("__global__")

    rng_eval = np.random.default_rng(seed + 100000)
    for _, row in locked_eval.iterrows():
        fb = simulate_feedback(bool(row["correct"]), rng_eval)
        engine.update(row["pred_category"], fb)
        static_engine.update("__global__", fb)

    eval_trust_arr = locked_eval["pred_category"].map(lambda c: engine.trust_for(c)).values
    eval_conf_arr = locked_eval["raw_confidence"].values
    eval_correct_arr = locked_eval["correct"].values
    eval_static_trust_arr = np.full(len(locked_eval), static_trust_value)

    _, static_metrics = evaluate_policy(FIXED_THRESHOLDS, eval_static_trust_arr, eval_conf_arr, eval_correct_arr)
    _, adaptive_fixed_metrics = evaluate_policy(FIXED_THRESHOLDS, eval_trust_arr, eval_conf_arr, eval_correct_arr)
    _, optimized_metrics = evaluate_policy(optimized_params, eval_trust_arr, eval_conf_arr, eval_correct_arr)

    return {
        "seed": seed,
        "policy_dev_n": len(policy_dev), "locked_eval_n": len(locked_eval),
        "optimized_params": optimized_params.tolist(),
        "static_silent_failure_rate": static_metrics["silent_failure_rate"],
        "static_escalation_rate": static_metrics["escalation_rate"],
        "adaptive_silent_failure_rate": adaptive_fixed_metrics["silent_failure_rate"],
        "adaptive_escalation_rate": adaptive_fixed_metrics["escalation_rate"],
        "optimized_silent_failure_rate": optimized_metrics["silent_failure_rate"],
        "optimized_escalation_rate": optimized_metrics["escalation_rate"],
    }


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 locked_eval_ablation.py <predictions_csv> [output_prefix]")
        sys.exit(1)

    preds_path = Path(sys.argv[1])
    prefix = sys.argv[2] if len(sys.argv) > 2 else preds_path.stem

    if not preds_path.exists():
        raise FileNotFoundError(f"Missing {preds_path}")
    preds = pd.read_csv(preds_path)
    print(f"Loaded {len(preds)} predictions from {preds_path}")
    print(f"Running {N_SEEDS}-seed LOCKED-EVALUATION ablation\n")

    results = []
    for seed in range(N_SEEDS):
        r = run_one_seed(preds, seed)
        print(f"  seed {seed}: [dev_n={r['policy_dev_n']}, eval_n={r['locked_eval_n']}] "
              f"static_esc={r['static_escalation_rate']:.4f}  adaptive_esc={r['adaptive_escalation_rate']:.4f}  "
              f"optimized_esc={r['optimized_escalation_rate']:.4f}")
        results.append(r)

    df = pd.DataFrame(results)
    csv_path = OUT_DIR / f"{prefix}_locked_eval_results.csv"
    df.to_csv(csv_path, index=False)

    print(f"\n{'='*70}\nSUMMARY (mean +/- SD across {N_SEEDS} seeds, LOCKED EVALUATION)\n{'='*70}")
    for arm in ["static", "adaptive", "optimized"]:
        sf = df[f"{arm}_silent_failure_rate"]
        esc = df[f"{arm}_escalation_rate"]
        print(f"{arm:>10}: silent_failure={sf.mean():.4f}+/-{sf.std():.4f}  escalation={esc.mean():.4f}+/-{esc.std():.4f}")

    print("\nPaired significance tests (locked evaluation):")
    sig_results = {}
    for pair in [("static", "adaptive"), ("adaptive", "optimized")]:
        a, b = pair
        for metric in ["silent_failure_rate", "escalation_rate"]:
            x, y = df[f"{a}_{metric}"], df[f"{b}_{metric}"]
            if np.allclose(x, y):
                print(f"  {a} vs {b} ({metric}): identical across all seeds")
                sig_results[f"{a}_vs_{b}_{metric}"] = None
                continue
            try:
                w, p = stats.wilcoxon(x, y)
                print(f"  {a} vs {b} ({metric}): Wilcoxon p={p:.4f}")
                sig_results[f"{a}_vs_{b}_{metric}"] = float(p)
            except ValueError as e:
                print(f"  {a} vs {b} ({metric}): {e}")
                sig_results[f"{a}_vs_{b}_{metric}"] = None

    summary = {
        "prefix": prefix, "n_seeds": N_SEEDS, "per_seed": results,
        "summary_means": {
            arm: {
                "silent_failure_rate_mean": float(df[f"{arm}_silent_failure_rate"].mean()),
                "silent_failure_rate_std": float(df[f"{arm}_silent_failure_rate"].std()),
                "escalation_rate_mean": float(df[f"{arm}_escalation_rate"].mean()),
                "escalation_rate_std": float(df[f"{arm}_escalation_rate"].std()),
            } for arm in ["static", "adaptive", "optimized"]
        },
        "significance_tests": sig_results,
    }
    json_path = OUT_DIR / f"{prefix}_locked_eval_results.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved to {csv_path} and {json_path}")


if __name__ == "__main__":
    main()
