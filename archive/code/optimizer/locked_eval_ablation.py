"""
Locked-evaluation arbitration ablation - fixes the policy/test-set
leakage concern (Issue #1). Splits each configuration's predictions
into policy_dev (used ONLY for feedback simulation + threshold search)
and locked_eval (untouched during optimization, evaluated exactly once
for the reported numbers).

CORRECTED VERSION: replaces the QPSO search (bounded trust_high in
[0.55, 0.95], which structurally cannot reach confidence-only-equivalent
policies) with an exhaustive grid search over the full [0,1]^2 for
(trust_high, conf_high) -- the only two parameters that affect the
AUTO/ESCALATE decision (trust_low/conf_low affect only queue ordering of
already-escalated alerts, confirmed against Section 3.4 of the paper, so
they are no longer searched). Objective also changed from the weighted
0.5/0.35/0.15 fitness to Section 3.1's actual stated objective: minimize
escalation subject to silent_failure_rate <= delta, tie-broken by lower
silent_failure_rate. Everything else -- the locked-eval split logic, the
static-trust arm, the significance tests -- is unchanged from the original.

Usage: python3 locked_eval_ablation.py <predictions_csv> [output_prefix]
Example: python3 locked_eval_ablation.py ../outputs/baseline_predictions.csv nslkdd_rf
"""
from pathlib import Path
import json
import sys
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
FIXED_THRESHOLDS = (0.75, 0.80, 0.60, 0.50)   # unchanged -- still used for the "adaptive (fixed)" arm
GRID_STEP = 0.005                              # 201 x 201 = 40,401 policies


class TrustEngine:
    """Unchanged from the original."""
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


def evaluate_policy(trust_high, conf_high, trust_arr, conf_arr, correct_arr):
    """
    2-parameter version -- trust_low/conf_low removed, since they do not
    affect the AUTO/ESCALATE decision (Section 3.4: they affect only
    whether an already-escalated alert is additionally flagged for
    priority review, not whether it is escalated at all).
    """
    auto_mask = (trust_arr >= trust_high) & (conf_arr >= conf_high)
    auto_count = auto_mask.sum()
    n = len(trust_arr)
    silent_failures = (auto_mask & (correct_arr == 0)).sum()
    silent_rate = silent_failures / max(auto_count, 1)
    escalation_rate = (n - auto_count) / n
    return float(silent_rate), float(escalation_rate)


def evaluate_policy_4param(params, trust_arr, conf_arr, correct_arr):
    """
    Kept for the 'static' and 'adaptive (fixed)' arms only, which still
    use the original 4-parameter FIXED_THRESHOLDS tuple unchanged -- this
    is NOT searched, so the trust_low/conf_low simplification above does
    not apply here; these arms are untouched from the original script.
    """
    trust_high, conf_high, trust_low, conf_low = params
    n = len(trust_arr)
    auto_mask = (trust_arr >= trust_high) & (conf_arr >= conf_high)
    auto_count = auto_mask.sum()
    silent_fail = (auto_mask & (correct_arr == 0)).sum()
    silent_rate = silent_fail / max(auto_count, 1)
    escalation_rate = (n - auto_count) / n
    return {"silent_failure_rate": float(silent_rate), "escalation_rate": float(escalation_rate)}


def vectorized_grid_search(trust_arr, conf_arr, correct_arr, delta=MAX_ACCEPTABLE_SILENT_RATE, step=GRID_STEP):
    """
    Exhaustive search over (trust_high, conf_high) in [0,1]^2. Replaces
    qpso_optimize entirely -- no bounds, no swarm, no local-optimum risk.
    Objective: minimize escalation_rate subject to silent_rate <= delta,
    tie-broken by lower silent_rate (matches Section 3.1 exactly, not the
    old weighted fitness).
    """
    grid_vals = np.arange(0.0, 1.0 + step / 2, step)
    n_grid = len(grid_vals)
    n_alerts = len(trust_arr)

    th_grid = grid_vals.reshape(n_grid, 1, 1)
    ch_grid = grid_vals.reshape(1, n_grid, 1)
    trust_b = trust_arr.reshape(1, 1, n_alerts)
    conf_b = conf_arr.reshape(1, 1, n_alerts)
    correct_b = correct_arr.reshape(1, 1, n_alerts)

    silent_rate = np.zeros((n_grid, n_grid))
    escalation_rate = np.zeros((n_grid, n_grid))
    chunk = max(1, 2_000_000 // (n_grid * n_alerts) or 1)
    for start in range(0, n_grid, chunk):
        end = min(start + chunk, n_grid)
        auto_mask = (trust_b >= th_grid[start:end]) & (conf_b >= ch_grid)
        auto_count = auto_mask.sum(axis=2)
        silent_fail = (auto_mask & (correct_b == 0)).sum(axis=2)
        silent_rate[start:end] = silent_fail / np.maximum(auto_count, 1)
        escalation_rate[start:end] = (n_alerts - auto_count) / n_alerts

    feasible = silent_rate <= delta
    if not feasible.any():
        raise RuntimeError(f"No feasible policy at delta={delta}, step={step}.")

    masked_escalation = np.where(feasible, escalation_rate, np.inf)
    min_escalation = masked_escalation.min()
    tie_mask = feasible & np.isclose(masked_escalation, min_escalation)
    tie_silent = np.where(tie_mask, silent_rate, np.inf)
    best_idx = np.unravel_index(np.argmin(tie_silent), tie_silent.shape)

    return {
        "trust_high": float(grid_vals[best_idx[0]]),
        "conf_high": float(grid_vals[best_idx[1]]),
        "silent_failure_rate": float(silent_rate[best_idx]),
        "escalation_rate": float(escalation_rate[best_idx]),
    }


def run_one_seed(preds, seed):
    shuffled = preds.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    n = len(shuffled)
    split_point = n // 2
    policy_dev = shuffled.iloc[:split_point].reset_index(drop=True)
    locked_eval = shuffled.iloc[split_point:].reset_index(drop=True)

    # --- Unchanged from the original: trust simulation on policy_dev ---
    rng_dev = np.random.default_rng(seed)
    engine = TrustEngine(decay=0.995)
    for _, row in policy_dev.iterrows():
        fb = simulate_feedback(bool(row["correct"]), rng_dev)
        engine.update(row["pred_category"], fb)

    dev_trust_arr = policy_dev["pred_category"].map(lambda c: engine.trust_for(c)).values
    dev_conf_arr = policy_dev["raw_confidence"].values
    dev_correct_arr = policy_dev["correct"].values

    # --- CHANGED: grid search replaces qpso_optimize ---
    grid_result = vectorized_grid_search(dev_trust_arr, dev_conf_arr, dev_correct_arr)
    optimized_params = (grid_result["trust_high"], grid_result["conf_high"])

    # --- Unchanged: static-trust arm (same RNG-stream-continuation behavior as original) ---
    static_engine = TrustEngine(decay=0.995)
    for _, row in policy_dev.iterrows():
        fb = simulate_feedback(bool(row["correct"]), rng_dev)
        static_engine.update("__global__", fb)
    static_trust_value = static_engine.trust_for("__global__")

    # --- Unchanged: continue updating trust through locked_eval ---
    rng_eval = np.random.default_rng(seed + 100000)
    for _, row in locked_eval.iterrows():
        fb = simulate_feedback(bool(row["correct"]), rng_eval)
        engine.update(row["pred_category"], fb)
        static_engine.update("__global__", fb)

    eval_trust_arr = locked_eval["pred_category"].map(lambda c: engine.trust_for(c)).values
    eval_conf_arr = locked_eval["raw_confidence"].values
    eval_correct_arr = locked_eval["correct"].values
    eval_static_trust_arr = np.full(len(locked_eval), static_trust_value)

    static_sf, static_esc = evaluate_policy(FIXED_THRESHOLDS[0], FIXED_THRESHOLDS[1],
                                              eval_static_trust_arr, eval_conf_arr, eval_correct_arr)
    adaptive_sf, adaptive_esc = evaluate_policy(FIXED_THRESHOLDS[0], FIXED_THRESHOLDS[1],
                                                  eval_trust_arr, eval_conf_arr, eval_correct_arr)
    optimized_sf, optimized_esc = evaluate_policy(optimized_params[0], optimized_params[1],
                                                    eval_trust_arr, eval_conf_arr, eval_correct_arr)

    return {
        "seed": seed,
        "policy_dev_n": len(policy_dev), "locked_eval_n": len(locked_eval),
        "optimized_params": {"trust_high": optimized_params[0], "conf_high": optimized_params[1]},
        "static_silent_failure_rate": static_sf, "static_escalation_rate": static_esc,
        "adaptive_silent_failure_rate": adaptive_sf, "adaptive_escalation_rate": adaptive_esc,
        "optimized_silent_failure_rate": optimized_sf, "optimized_escalation_rate": optimized_esc,
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
    print(f"Running {N_SEEDS}-seed LOCKED-EVALUATION ablation (corrected: exhaustive grid, no QPSO bounds)\n")

    results = []
    t0 = time.time()
    for seed in range(N_SEEDS):
        r = run_one_seed(preds, seed)
        print(f"  seed {seed}: [dev_n={r['policy_dev_n']}, eval_n={r['locked_eval_n']}] "
              f"static_esc={r['static_escalation_rate']:.4f}  adaptive_esc={r['adaptive_escalation_rate']:.4f}  "
              f"optimized_esc={r['optimized_escalation_rate']:.4f}  "
              f"(trust_high={r['optimized_params']['trust_high']:.3f})")
        results.append(r)
    print(f"\nTotal: {time.time()-t0:.1f}s")

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
