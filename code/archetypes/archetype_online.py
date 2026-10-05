"""
Online-trust re-run of the archetype stress test (Table 15) and the
automation-bias sweep.

Problems in the original scripts:
  1. Trust was the END-OF-STREAM snapshot: every alert was judged with trust
     that already contained its own feedback and all later feedback.
  2. One seed (42) only, so Table 15 has no variance estimate.
  3. With a snapshot, the policy only depends on WHICH categories end above
     tau_h, so different archetypes that leave the same categories above the
     threshold produce byte-identical metrics (the original mixed-team and
     Complacent rows on NSL-KDD are identical to 16 digits for this reason).

This script:
  - decides each alert with the trust held BEFORE its own feedback (online);
  - runs 10 seeds (0-9), 5,000 resampled alerts each, as in the original;
  - evaluates two policies on the same feedback stream:
      fixed  = (tau_h, gamma_h) = (0.75, 0.80)  -- the paper's original setting
      tuned  = the seed-matched policy selected by locked_eval_robust.py
               (reads <cfg>_locked_eval_robust.json);
  - reproduces the original seed-42 snapshot numbers first, as a check.

Feedback is available on every alert (full-feedback assumption, as in the
original archetype test).

Usage: python3 archetype_online.py            (runs NSL-KDD, UNSW-NB15, CICIDS2017; RandomForest)
"""
from pathlib import Path
import json, sys
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyst_archetypes import archetype_feedback, ARCHETYPES
from automation_bias_sweep import sweep_feedback, AUTOMATION_BIAS_LEVELS

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = PROJECT_ROOT / "outputs"
N_ALERTS, SEEDS, DELTA, LAMBDA = 5000, range(10), 0.05, 0.995
FIXED = (0.75, 0.80)
MIXED = {"novice": 0.30, "expert": 0.20, "complacent": 0.50}
CONFIGS = [  # (label, predictions, dataset_key for expert categories, robust-eval prefix)
    ("NSL-KDD", "baseline_predictions.csv", "nsl_kdd", "nslkdd_rf"),
    ("UNSW-NB15", "unsw_baseline_predictions.csv", "unsw_nb15", "unsw_rf"),
    ("CICIDS2017", "cicids_baseline_predictions.csv", "cicids2017", "cicids_rf"),
]


def sample_stream(preds, seed):
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(preds), size=N_ALERTS, replace=N_ALERTS > len(preds))
    return preds.iloc[idx].reset_index(drop=True)


def simulate(stream, feedback_fn, seed):
    """Returns (trust before each alert's feedback, end-of-stream trust per alert)."""
    rng = np.random.default_rng(seed)
    state, pre = {}, np.empty(len(stream))
    cats = stream["pred_category"].values
    corr = stream["correct"].values.astype(bool)
    conf = stream["raw_confidence"].values.astype(float)
    for i in range(len(stream)):
        a, b = state.get(cats[i], (1.0, 1.0))
        pre[i] = a / (a + b)
        fb = feedback_fn(bool(corr[i]), float(conf[i]), cats[i], rng)
        a, b = a * LAMBDA, b * LAMBDA
        state[cats[i]] = (a + 1, b) if fb else (a, b + 1)
    end = np.array([state[c][0] / sum(state[c]) for c in cats])
    return pre, end


def evaluate(th, ch, trust, stream):
    auto = (trust >= th) & (stream["raw_confidence"].values >= ch)
    wrong = stream["correct"].values == 0
    return float((auto & wrong).sum() / max(auto.sum(), 1)), float(1 - auto.mean())


def archetype_fn(name, key):
    if name == "mixed_team":
        names, probs = zip(*MIXED.items())
        def fn(c, conf, cat, rng):
            who = rng.choice(names, p=probs)
            return archetype_feedback(who, c, conf, cat, key, rng)[0]
        return fn
    return lambda c, conf, cat, rng: archetype_feedback(name, c, conf, cat, key, rng)[0]


def tuned_policies(prefix):
    path = OUT_DIR / f"{prefix}_locked_eval_robust.json"
    d = json.load(open(path))
    return {r["seed"]: (r["adaptive_tuned_tau_h"], r["adaptive_tuned_gamma_h"]) for r in d["per_seed"]}


def summarize(vals):
    a = np.array(vals)
    return {"silent_mean": a[:, 0].mean(), "silent_sd": a[:, 0].std(ddof=1), "silent_max": a[:, 0].max(),
            "over_budget": int((a[:, 0] > DELTA).sum()), "esc_mean": a[:, 1].mean(), "esc_sd": a[:, 1].std(ddof=1)}


def main():
    out = {}
    for label, fname, key, prefix in CONFIGS:
        preds = pd.read_csv(OUT_DIR / fname)
        tuned = tuned_policies(prefix)
        res = {"reproduction_seed42_snapshot": {}, "archetypes": {}, "bias_sweep": {}}

        # Reproduction of the original (seed 42, snapshot, fixed thresholds)
        for name in list(ARCHETYPES) + ["mixed_team"]:
            st = sample_stream(preds, 42)
            _, end = simulate(st, archetype_fn(name, key), 42)
            res["reproduction_seed42_snapshot"][name] = evaluate(*FIXED, end, st)

        print(f"\n{'=' * 86}\n{label} (RandomForest)\n{'=' * 86}")
        print("Reproduction of original (seed 42, snapshot, fixed): " +
              "  ".join(f"{n}={v[0]:.4f}/{v[1]:.4f}" for n, v in res["reproduction_seed42_snapshot"].items()))
        print(f"\n{'archetype':<12}{'FIXED online: silent':>24}{'esc':>8}{'>δ':>5}   "
              f"{'TUNED online: silent':>22}{'esc':>8}{'>δ':>5}")
        for name in list(ARCHETYPES) + ["mixed_team"]:
            fx, tu = [], []
            for s in SEEDS:
                st = sample_stream(preds, s)
                pre, _ = simulate(st, archetype_fn(name, key), s)
                fx.append(evaluate(*FIXED, pre, st))
                tu.append(evaluate(*tuned[s], pre, st))
            f, t = summarize(fx), summarize(tu)
            res["archetypes"][name] = {"fixed": f, "tuned": t}
            print(f"{name:<12}{f['silent_mean']:>14.4f}±{f['silent_sd']:.4f}{f['esc_mean']:>8.3f}{f['over_budget']:>4}/10   "
                  f"{t['silent_mean']:>12.4f}±{t['silent_sd']:.4f}{t['esc_mean']:>8.3f}{t['over_budget']:>4}/10")

        print(f"\nAutomation-bias sweep (Complacent base parameters):")
        print(f"{'bias':<6}{'FIXED silent':>14}{'>δ':>7}{'TUNED silent':>15}{'>δ':>7}")
        for bias in AUTOMATION_BIAS_LEVELS:
            fx, tu = [], []
            for s in SEEDS:
                st = sample_stream(preds, s)
                pre, _ = simulate(st, lambda c, conf, cat, rng: sweep_feedback(c, conf, bias, rng), s)
                fx.append(evaluate(*FIXED, pre, st))
                tu.append(evaluate(*tuned[s], pre, st))
            f, t = summarize(fx), summarize(tu)
            res["bias_sweep"][str(bias)] = {"fixed": f, "tuned": t}
            print(f"{bias:<6}{f['silent_mean']:>14.4f}{f['over_budget']:>5}/10{t['silent_mean']:>13.4f}{t['over_budget']:>5}/10")
        out[label] = res

    with open(OUT_DIR / "archetype_online_results.json", "w") as fh:
        json.dump(out, fh, indent=2, default=float)
    print("\nSaved archetype_online_results.json")


if __name__ == "__main__":
    main()
