"""
Robust locked-evaluation arbitration comparison (replaces the naive-grid
selection in locked_eval_ablation.py).

Why: the naive exhaustive grid places tau_h within <0.005 of some category's
dev-final trust on every seed (knife-edge). When that category's trust moves
during locked evaluation, it enters AUTO and the silent-failure budget breaks
(UNSW-NB15/RF 9.7%, UNSW-NB15/XGB 11.9%, NSL-KDD/XGB 7.7%).

What changes (all selection uses the policy-dev half ONLY):
  1. Online (prequential) trust: each alert is decided with the trust value
     held BEFORE its own feedback, in both selection and evaluation. The old
     script used the end-of-locked-eval snapshot, which includes feedback from
     the very alerts being evaluated.
  2. Chance-constrained selection: K replicas of the dev stream (reshuffled
     order + fresh feedback noise); a policy is feasible only if it meets the
     budget on ALL K replicas.
  3. UCB constraint: feasibility uses the one-sided 95% Clopper-Pearson upper
     bound of the silent-failure rate, not the point estimate. Applied
     identically to every tuned arm (trust, static, confidence-only, rejector).
  4. Objective = Section 3.1: minimise escalation s.t. s(pi) <= delta.
     Exhaustive grid over (tau_h, gamma_h) in [0,1]^2, step 0.005.

Usage: python3 locked_eval_robust.py <predictions_csv> [output_prefix] [K]
"""
from pathlib import Path
import json, sys, time
import numpy as np
import pandas as pd
from scipy import stats
from scipy.signal import lfilter
from sklearn.linear_model import LogisticRegression
import warnings
warnings.filterwarnings("ignore", message=".*encountered in matmul", category=RuntimeWarning)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = PROJECT_ROOT / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

N_SEEDS = 10
DELTA = 0.05
LAMBDA = 0.995
FEEDBACK_ERROR = 0.08
UCB_LEVEL = 0.95
GRID_STEP = 0.005
GRID = np.arange(0.0, 1.0 + GRID_STEP / 2, GRID_STEP)
FIXED_THRESHOLDS = (0.75, 0.80)          # the hand-picked "fixed" arms, unchanged


def feedback(correct, u):
    """Simulated analyst feedback with FEEDBACK_ERROR flip rate."""
    return np.where(u < FEEDBACK_ERROR, correct == 0, correct == 1).astype(float)


def online_trust(cat_idx, fb, n_cats, a0=None, b0=None):
    """Prequential Beta trust: value BEFORE each alert's own update.
    Exponential decay applied per update, per category (Eq. 2 and Section 3.3)."""
    A = np.ones(n_cats) if a0 is None else a0.copy()
    B = np.ones(n_cats) if b0 is None else b0.copy()
    pre = np.empty(len(cat_idx))
    for k in range(n_cats):
        m = np.where(cat_idx == k)[0]
        if len(m) == 0:
            continue
        f = fb[m]
        a = lfilter([1], [1, -LAMBDA], f, zi=[LAMBDA * A[k]])[0]
        b = lfilter([1], [1, -LAMBDA], 1 - f, zi=[LAMBDA * B[k]])[0]
        ap = np.concatenate([[A[k]], a[:-1]])
        bp = np.concatenate([[B[k]], b[:-1]])
        pre[m] = ap / (ap + bp)
        A[k], B[k] = a[-1], b[-1]
    return pre, A, B


def grid_rates(trust, conf, wrong):
    """Silent-failure (Eq. 1, denominator = AUTO count) and escalation over GRID^2."""
    n, G = len(trust), len(GRID)
    o = np.argsort(conf)
    conf, trust, wrong = conf[o], trust[o], wrong[o]
    S, E, N = np.zeros((G, G)), np.zeros((G, G)), np.zeros((G, G))
    for i, th in enumerate(GRID):
        m = trust >= th
        cf, wr = conf[m], wrong[m]
        idx = np.searchsorted(cf, GRID, side="left")
        tot = len(cf) - idx
        cw = np.concatenate([np.cumsum(wr[::-1])[::-1], [0]])
        S[i] = cw[idx] / np.maximum(tot, 1)
        E[i] = (n - tot) / n
        N[i] = tot
    return S, E, N


def ucb(S, N):
    x = np.rint(S * N)
    return np.where(N > 0, stats.beta.ppf(UCB_LEVEL, x + 1, np.maximum(N - x, 1)), 0.0)


def evaluate(th, ch, trust, conf, wrong):
    auto = (trust >= th) & (conf >= ch)
    return float((auto & wrong).sum() / max(auto.sum(), 1)), float(1 - auto.mean())


def select_chance_constrained(dev_cat, dev_conf, dev_correct, n_cats, K, rng):
    """Min mean escalation s.t. UCB(silent) <= DELTA on all K dev replicas."""
    feas, esc = None, 0.0
    for _ in range(K):
        perm = rng.permutation(len(dev_cat))
        fb = feedback(dev_correct[perm], rng.random(len(perm)))
        pre, _, _ = online_trust(dev_cat[perm], fb, n_cats)
        S, E, N = grid_rates(pre, dev_conf[perm], dev_correct[perm] == 0)
        ok = ucb(S, N) <= DELTA
        feas = ok if feas is None else (feas & ok)
        esc = esc + E / K
    if not feas.any():
        raise RuntimeError("No policy feasible on all replicas.")
    i, j = np.unravel_index(np.argmin(np.where(feas, esc, np.inf)), esc.shape)
    return GRID[i], GRID[j]


def rejector_arm(dev, ev, seed):
    """Learned rejector under the same split, same noisy feedback, same UCB rule."""
    h = int(len(dev) * 0.6)
    tr, th = dev.iloc[:h], dev.iloc[h:]
    y = 1 - feedback(tr["correct"].values, np.random.default_rng(seed + 300_000).random(len(tr)))
    cer = pd.Series(y, index=tr.index).groupby(tr["pred_category"]).mean().to_dict()
    g = y.mean()
    X = lambda d: np.c_[d["raw_confidence"].values, d["pred_category"].map(cer).fillna(g).values]
    model = LogisticRegression(max_iter=1000, random_state=seed).fit(X(tr), y)
    p_dev, p_ev = model.predict_proba(X(th))[:, 1], model.predict_proba(X(ev))[:, 1]
    wr = th["correct"].values == 0
    best_e, best_t = np.inf, 0.0
    for t in np.unique(np.r_[p_dev, 1.0 + 1e-9]):
        a = p_dev < t
        n_a, x = a.sum(), (a & wr).sum()
        u = stats.beta.ppf(UCB_LEVEL, x + 1, max(n_a - x, 1)) if n_a else 0.0
        if u <= DELTA and 1 - a.mean() < best_e:
            best_e, best_t = 1 - a.mean(), t
    a = p_ev < best_t
    return float(((ev["correct"].values == 0) & a).sum() / max(a.sum(), 1)), float(1 - a.mean())


def run_one_seed(preds, seed, K):
    sh = preds.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    half = len(sh) // 2
    dev, ev = sh.iloc[:half].reset_index(drop=True), sh.iloc[half:].reset_index(drop=True)
    cats = sorted(preds["pred_category"].unique())
    cidx = {c: i for i, c in enumerate(cats)}
    d_cat, e_cat = dev["pred_category"].map(cidx).values, ev["pred_category"].map(cidx).values
    d_conf, e_conf = dev["raw_confidence"].values, ev["raw_confidence"].values
    d_cor, e_wrong = dev["correct"].values, ev["correct"].values == 0
    zeros_d, zeros_e = np.zeros(len(dev), int), np.zeros(len(ev), int)

    # Dev pass, then locked-eval pass continuing from the dev-final state (online).
    fb_d = feedback(d_cor, np.random.default_rng(seed).random(len(dev)))
    fb_e = feedback(ev["correct"].values, np.random.default_rng(seed + 100_000).random(len(ev)))
    _, A, B = online_trust(d_cat, fb_d, len(cats))
    adapt_e, _, _ = online_trust(e_cat, fb_e, len(cats), A, B)
    _, As, Bs = online_trust(zeros_d, fb_d, 1)
    static_e, _, _ = online_trust(zeros_e, fb_e, 1, As, Bs)

    rng = np.random.default_rng(10_000 + seed)
    out = {"seed": seed}

    def record(name, th, ch, trust):
        s, e = evaluate(th, ch, trust, e_conf, e_wrong)
        out.update({f"{name}_tau_h": float(th), f"{name}_gamma_h": float(ch),
                    f"{name}_silent_failure_rate": s, f"{name}_escalation_rate": e})

    record("static_fixed", *FIXED_THRESHOLDS, static_e)
    record("adaptive_fixed", *FIXED_THRESHOLDS, adapt_e)
    record("static_tuned", *select_chance_constrained(zeros_d, d_conf, d_cor, 1, K, rng), static_e)
    record("adaptive_tuned", *select_chance_constrained(d_cat, d_conf, d_cor, len(cats), K, rng), adapt_e)

    # Confidence-only: tau_h fixed at 0, gamma_h chosen with the same UCB rule.
    S, E, N = grid_rates(np.ones(len(dev)), d_conf, d_cor == 0)
    j = int(np.argmin(np.where(ucb(S, N)[0] <= DELTA, E[0], np.inf)))
    record("confidence_only", 0.0, GRID[j], np.ones(len(ev)))

    s, e = rejector_arm(dev, ev, seed)
    out.update({"rejector_silent_failure_rate": s, "rejector_escalation_rate": e})
    return out


ARMS = ["static_fixed", "adaptive_fixed", "static_tuned", "adaptive_tuned", "confidence_only", "rejector"]


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    path = Path(sys.argv[1])
    prefix = sys.argv[2] if len(sys.argv) > 2 else path.stem
    K = int(sys.argv[3]) if len(sys.argv) > 3 else 20
    preds = pd.read_csv(path)
    print(f"Loaded {len(preds)} predictions; {N_SEEDS} seeds, K={K} replicas, delta={DELTA}, UCB {UCB_LEVEL}")

    t0, rows = time.time(), []
    for seed in range(N_SEEDS):
        r = run_one_seed(preds, seed, K)
        rows.append(r)
        print(f"  seed {seed}: tuned tau_h={r['adaptive_tuned_tau_h']:.3f} "
              f"s={r['adaptive_tuned_silent_failure_rate']:.4f} e={r['adaptive_tuned_escalation_rate']:.4f} | "
              f"conf-only e={r['confidence_only_escalation_rate']:.4f} | rejector e={r['rejector_escalation_rate']:.4f}")
    df = pd.DataFrame(rows)
    print(f"\nTotal {time.time() - t0:.0f}s\n{'=' * 78}")

    summary = {"prefix": prefix, "K": K, "delta": DELTA, "ucb_level": UCB_LEVEL, "arms": {}, "tests": {}}
    for arm in ARMS:
        s, e = df[f"{arm}_silent_failure_rate"], df[f"{arm}_escalation_rate"]
        summary["arms"][arm] = {"silent_mean": s.mean(), "silent_sd": s.std(), "silent_max": s.max(),
                                "violations": int((s > DELTA).sum()),
                                "escalation_mean": e.mean(), "escalation_sd": e.std()}
        print(f"{arm:>16}: s={s.mean():.4f}±{s.std():.4f} (max {s.max():.4f}, >δ {int((s > DELTA).sum())}/10)  "
              f"e={e.mean():.4f}±{e.std():.4f}")
    print("\nPaired Wilcoxon, escalation (adaptive_tuned vs ...):")
    for other in ["confidence_only", "rejector", "static_tuned"]:
        x, y = df["adaptive_tuned_escalation_rate"], df[f"{other}_escalation_rate"]
        p = None if np.allclose(x, y) else float(stats.wilcoxon(x, y).pvalue)
        summary["tests"][f"adaptive_tuned_vs_{other}"] = {"mean_diff_pp": float((x - y).mean() * 100), "p": p}
        print(f"  vs {other:>15}: diff={(x - y).mean() * 100:+.2f} pp  p={p}")

    df.to_csv(OUT_DIR / f"{prefix}_locked_eval_robust.csv", index=False)
    with open(OUT_DIR / f"{prefix}_locked_eval_robust.json", "w") as f:
        json.dump({**summary, "per_seed": rows}, f, indent=2, default=float)
    print(f"\nSaved {prefix}_locked_eval_robust.csv/.json")


if __name__ == "__main__":
    main()
