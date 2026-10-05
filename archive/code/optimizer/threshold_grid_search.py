"""
Exhaustive grid search over the arbitration policy's two decision-relevant
parameters (tau_h, gamma_h), replacing the swarm-based optimizer.

Per Section 3.4 of the paper: the decision D_pi depends only on
pi_A = (tau_h, gamma_h); tau_l/gamma_l affect only queue ordering of
already-escalated alerts, never the AUTO/ESCALATE decision itself, and are
therefore NOT part of this search (they are fixed separately, unchanged).

Objective matches Section 3.1 exactly: minimize e(pi) subject to
s(pi) <= delta, tie-broken by lower s(pi). This replaces Equation 6's
weighted fitness -- no weights, no ambiguity about what "best" means.

s(pi)'s denominator is the AUTO-decision count (matching the paper's own
prose "conditional error rate of automated decisions", and matching every
one of the original optimizer scripts' actual implementation -- the
Equation 1 OMML rendering bug was a manuscript typo, not a computation bug).
"""
from pathlib import Path
import json
import time
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = PROJECT_ROOT / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

MAX_ACCEPTABLE_SILENT_RATE = 0.05  # delta
GRID_STEP = 0.005                  # 201 points per axis, 201*201 = 40,401 policies


# ---------------------------------------------------------------------------
# Core policy evaluation -- 2 parameters only (tau_h, gamma_h)
# ---------------------------------------------------------------------------

def evaluate_policy(tau_h, gamma_h, trust_arr, conf_arr, correct_arr):
    """
    s(pi): silent-failure rate among AUTO-executed alerts only (correct
           convention, verified against the original optimizer scripts).
    e(pi): escalation rate over ALL alerts.
    tau_l/gamma_l are deliberately absent -- they do not affect this
    decision (Section 3.4), so they play no role in s(pi) or e(pi).
    """
    auto_mask = (trust_arr >= tau_h) & (conf_arr >= gamma_h)
    auto_count = auto_mask.sum()
    n = len(trust_arr)

    silent_failures = (auto_mask & (correct_arr == 0)).sum()
    silent_rate = silent_failures / max(auto_count, 1)
    escalation_rate = (n - auto_count) / n

    return float(silent_rate), float(escalation_rate)


# ---------------------------------------------------------------------------
# FULL-FEEDBACK regime: trust trajectory is identical for every policy
# (feedback arrives regardless of the decision), so trust_arr is computed
# ONCE and the entire grid is scored with vectorized NumPy broadcasting.
# ---------------------------------------------------------------------------

def vectorized_grid_search(trust_arr, conf_arr, correct_arr,
                            delta=MAX_ACCEPTABLE_SILENT_RATE, step=GRID_STEP):
    t0 = time.time()

    grid_vals = np.arange(0.0, 1.0 + step / 2, step)  # inclusive of 1.0
    n_grid = len(grid_vals)
    n_alerts = len(trust_arr)

    # Broadcast: (n_grid, 1, 1) thresholds vs (1, n_alerts) data arrays
    tau_h_grid = grid_vals.reshape(n_grid, 1, 1)
    gamma_h_grid = grid_vals.reshape(1, n_grid, 1)
    trust_b = trust_arr.reshape(1, 1, n_alerts)
    conf_b = conf_arr.reshape(1, 1, n_alerts)
    correct_b = correct_arr.reshape(1, 1, n_alerts)

    # This full (n_grid, n_grid, n_alerts) boolean array can be large;
    # process in row-chunks over tau_h to bound peak memory.
    silent_rate = np.zeros((n_grid, n_grid))
    escalation_rate = np.zeros((n_grid, n_grid))

    chunk = max(1, 2_000_000 // (n_grid * n_alerts) or 1)
    for start in range(0, n_grid, chunk):
        end = min(start + chunk, n_grid)
        th_slice = tau_h_grid[start:end]  # (chunk, 1, 1)
        auto_mask = (trust_b >= th_slice) & (conf_b >= gamma_h_grid)  # (chunk, n_grid, n_alerts)
        auto_count = auto_mask.sum(axis=2)
        silent_fail = (auto_mask & (correct_b == 0)).sum(axis=2)
        silent_rate[start:end] = silent_fail / np.maximum(auto_count, 1)
        escalation_rate[start:end] = (n_alerts - auto_count) / n_alerts

    feasible = silent_rate <= delta
    if not feasible.any():
        raise RuntimeError(
            f"No feasible policy found at delta={delta} on this grid step={step}. "
            f"Min achievable silent_rate was {silent_rate.min():.4f}."
        )

    # Among feasible cells: minimize escalation, tie-break by lower silent_rate.
    masked_escalation = np.where(feasible, escalation_rate, np.inf)
    min_escalation = masked_escalation.min()
    tie_mask = feasible & np.isclose(masked_escalation, min_escalation)
    # Among ties, pick lowest silent_rate; if still tied, take the first.
    tie_silent = np.where(tie_mask, silent_rate, np.inf)
    best_idx = np.unravel_index(np.argmin(tie_silent), tie_silent.shape)

    best_tau_h = grid_vals[best_idx[0]]
    best_gamma_h = grid_vals[best_idx[1]]
    best_s = silent_rate[best_idx]
    best_e = escalation_rate[best_idx]

    elapsed = time.time() - t0
    return {
        "tau_h": float(best_tau_h),
        "gamma_h": float(best_gamma_h),
        "silent_failure_rate": float(best_s),
        "escalation_rate": float(best_e),
        "grid_points_evaluated": int(n_grid * n_grid),
        "elapsed_seconds": elapsed,
    }


# ---------------------------------------------------------------------------
# SELECTIVE-FEEDBACK regime: trust trajectory depends on which alerts were
# escalated (feedback only arrives for escalated/audited alerts), so each
# policy needs its OWN sequential re-simulation. A full 201x201 grid at this
# cost is likely too slow -- use a coarse grid first, then refine locally
# around the best coarse cell. Plug your actual per-alert feedback loop into
# `simulate_one_policy` below; this is a template, not tested against your
# real selective-feedback code (which wasn't available to write this from).
# ---------------------------------------------------------------------------

def simulate_one_policy(tau_h, gamma_h, preds_df, feedback_fn, trust_engine_cls, decay=0.995):
    """
    TEMPLATE -- adapt the inner loop to match your actual selective-feedback
    simulation (e.g. selective_feedback.py / ugaa_experiment.py). This
    version assumes: alerts processed in a fixed order; an alert is
    AUTO-executed if trust>=tau_h and confidence>=gamma_h (else ESCALATED);
    ESCALATED alerts always receive feedback; AUTO-executed alerts do not
    (adjust if your audit/UGAA logic applies here too).
    """
    engine = trust_engine_cls(decay=decay)
    auto_count = 0
    silent_failures = 0
    n = len(preds_df)

    for _, row in preds_df.iterrows():
        category = row["pred_category"]
        trust = engine.trust_for(category)
        conf = row["raw_confidence"]
        is_auto = (trust >= tau_h) and (conf >= gamma_h)

        if is_auto:
            auto_count += 1
            if row["correct"] == 0:
                silent_failures += 1
            # no feedback in the pure escalation-only regime; adjust here
            # if this policy point should also be subject to audit sampling
        else:
            fb = feedback_fn(bool(row["correct"]))
            engine.update(category, fb)

    silent_rate = silent_failures / max(auto_count, 1)
    escalation_rate = (n - auto_count) / n
    return silent_rate, escalation_rate


def coarse_to_fine_grid_search(preds_df, feedback_fn, trust_engine_cls,
                                delta=MAX_ACCEPTABLE_SILENT_RATE,
                                coarse_step=0.05, fine_step=0.005, fine_radius=0.06):
    t0 = time.time()
    coarse_vals = np.arange(0.0, 1.0 + coarse_step / 2, coarse_step)

    coarse_results = []
    for tau_h in coarse_vals:
        for gamma_h in coarse_vals:
            s, e = simulate_one_policy(tau_h, gamma_h, preds_df, feedback_fn, trust_engine_cls)
            coarse_results.append((tau_h, gamma_h, s, e))

    coarse_df = pd.DataFrame(coarse_results, columns=["tau_h", "gamma_h", "s", "e"])
    feasible = coarse_df[coarse_df["s"] <= delta]
    if feasible.empty:
        raise RuntimeError(f"No feasible policy in coarse grid at delta={delta}.")
    best_coarse = feasible.loc[feasible["e"].idxmin()]

    # Refine: fine grid within +/- fine_radius of the best coarse cell
    fine_tau_vals = np.arange(
        max(0.0, best_coarse["tau_h"] - fine_radius),
        min(1.0, best_coarse["tau_h"] + fine_radius) + fine_step / 2, fine_step)
    fine_gamma_vals = np.arange(
        max(0.0, best_coarse["gamma_h"] - fine_radius),
        min(1.0, best_coarse["gamma_h"] + fine_radius) + fine_step / 2, fine_step)

    fine_results = []
    for tau_h in fine_tau_vals:
        for gamma_h in fine_gamma_vals:
            s, e = simulate_one_policy(tau_h, gamma_h, preds_df, feedback_fn, trust_engine_cls)
            fine_results.append((tau_h, gamma_h, s, e))

    fine_df = pd.DataFrame(fine_results, columns=["tau_h", "gamma_h", "s", "e"])
    feasible_fine = fine_df[fine_df["s"] <= delta]
    best = (feasible_fine if not feasible_fine.empty else feasible).loc[
        (feasible_fine if not feasible_fine.empty else feasible)["e"].idxmin()]

    elapsed = time.time() - t0
    return {
        "tau_h": float(best["tau_h"]),
        "gamma_h": float(best["gamma_h"]),
        "silent_failure_rate": float(best["s"]),
        "escalation_rate": float(best["e"]),
        "coarse_points_evaluated": len(coarse_results),
        "fine_points_evaluated": len(fine_results),
        "elapsed_seconds": elapsed,
    }


# ---------------------------------------------------------------------------
# Usage example / entry point -- full-feedback regime, matching the data
# loading convention from your other optimizer/ scripts.
# ---------------------------------------------------------------------------

def main():
    preds_path = OUT_DIR / "baseline_predictions.csv"
    if not preds_path.exists():
        raise FileNotFoundError(f"Run train_baseline.py first - missing {preds_path}")
    preds = pd.read_csv(preds_path)

    with open(OUT_DIR / "trust_snapshot.json") as f:
        snapshot = json.load(f)
    trust_by_category = {row["category"]: row["trust"] for row in snapshot}

    trust_arr = preds["pred_category"].map(trust_by_category).fillna(0.5).values
    conf_arr = preds["raw_confidence"].values
    correct_arr = preds["correct"].values

    print(f"Loaded {len(preds)} alerts.")
    print(f"Grid: {int(1/GRID_STEP)+1} x {int(1/GRID_STEP)+1} = "
          f"{(int(1/GRID_STEP)+1)**2:,} policies, step={GRID_STEP}\n")

    # --- Comparison baselines, for the new Table 9 ---
    s_conf, e_conf = evaluate_policy(0.0, 0.80, trust_arr, conf_arr, correct_arr)
    # NOTE: confirm the actual confidence-only gamma_h threshold used
    # elsewhere in your pipeline (baselines/calibrate_confidence.py or
    # wherever confidence_only_six_configs.py sets it) and use that exact
    # value here for a fair, consistent baseline -- 0.80 is a placeholder.
    print(f"Confidence-only (gamma_h=0.80, placeholder - confirm real value): "
          f"s={s_conf:.4f}, e={e_conf:.4f}")

    s_fixed, e_fixed = evaluate_policy(0.75, 0.80, trust_arr, conf_arr, correct_arr)
    print(f"Adaptive trust (fixed thresholds, tau_h=0.75/gamma_h=0.80): "
          f"s={s_fixed:.4f}, e={e_fixed:.4f}\n")

    # --- The grid search itself ---
    print("Running exhaustive grid search...")
    result = vectorized_grid_search(trust_arr, conf_arr, correct_arr,
                                     delta=MAX_ACCEPTABLE_SILENT_RATE, step=GRID_STEP)
    print(f"\nDone in {result['elapsed_seconds']:.2f}s "
          f"({result['grid_points_evaluated']:,} policies evaluated)")
    print(f"Best policy: tau_h={result['tau_h']:.3f}, gamma_h={result['gamma_h']:.3f}")
    print(f"  silent_failure_rate = {result['silent_failure_rate']:.4f}")
    print(f"  escalation_rate     = {result['escalation_rate']:.4f}")

    print(f"\n=== Comparison ===")
    print(f"{'Arm':<35}{'Silent failure':>16}{'Escalation':>14}")
    print(f"{'Confidence-only (placeholder)':<35}{s_conf:>16.4f}{e_conf:>14.4f}")
    print(f"{'Adaptive (fixed thresholds)':<35}{s_fixed:>16.4f}{e_fixed:>14.4f}")
    print(f"{'Adaptive (grid-tuned)':<35}{result['silent_failure_rate']:>16.4f}{result['escalation_rate']:>14.4f}")

    with open(OUT_DIR / "threshold_grid_search_results.json", "w") as f:
        json.dump({
            "delta": MAX_ACCEPTABLE_SILENT_RATE,
            "grid_step": GRID_STEP,
            "confidence_only": {"tau_h": 0.0, "gamma_h": 0.80, "silent_failure_rate": s_conf, "escalation_rate": e_conf},
            "adaptive_fixed": {"tau_h": 0.75, "gamma_h": 0.80, "silent_failure_rate": s_fixed, "escalation_rate": e_fixed},
            "adaptive_grid_tuned": result,
        }, f, indent=2)
    print(f"\nSaved to {OUT_DIR / 'threshold_grid_search_results.json'}")


if __name__ == "__main__":
    main()
