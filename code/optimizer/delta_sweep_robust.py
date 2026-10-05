"""
Risk-workload curve with robust selection (replaces pareto_sweep.py).

For each silent-failure ceiling delta, every tuned arm is selected exactly as in
locked_eval_robust.py (exhaustive grid, online trust, K=20 dev replicas,
95% Clopper-Pearson upper bound <= delta on all replicas) and evaluated once
on the locked half. Arms: adaptive trust (tuned), confidence-only (tuned; equal
to tuned static trust), learned rejector.

Usage:
  python3 delta_sweep_robust.py <predictions_csv> <prefix> [K]   # one configuration
  python3 delta_sweep_robust.py --plot                           # figure from all saved sweeps
"""
from pathlib import Path
import json, sys, time
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import locked_eval_robust as ler

OUT_DIR = ler.OUT_DIR
DELTAS = [0.01, 0.02, 0.03, 0.05, 0.08, 0.10, 0.15, 0.20]
ARMS = ["adaptive_tuned", "confidence_only", "rejector"]
LABELS = {"adaptive_tuned": "Adaptive trust (tuned)", "confidence_only": "Confidence-only",
          "rejector": "Learned rejector"}
ORDER = ["nslkdd_rf", "nslkdd_xgb", "unsw_rf", "unsw_xgb", "cicids_rf", "cicids_xgb"]
TITLES = {"nslkdd_rf": "NSL-KDD / RF", "nslkdd_xgb": "NSL-KDD / XGB", "unsw_rf": "UNSW-NB15 / RF",
          "unsw_xgb": "UNSW-NB15 / XGB", "cicids_rf": "CICIDS2017 / RF", "cicids_xgb": "CICIDS2017 / XGB"}


def one_seed(preds, seed, K):
    sh = preds.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    half = len(sh) // 2
    dev, ev = sh.iloc[:half].reset_index(drop=True), sh.iloc[half:].reset_index(drop=True)
    cats = sorted(preds["pred_category"].unique())
    cidx = {c: i for i, c in enumerate(cats)}
    d_cat, e_cat = dev["pred_category"].map(cidx).values, ev["pred_category"].map(cidx).values
    d_conf, e_conf = dev["raw_confidence"].values, ev["raw_confidence"].values
    d_cor, e_wrong = dev["correct"].values, ev["correct"].values == 0

    fb_d = ler.feedback(d_cor, np.random.default_rng(seed).random(len(dev)))
    fb_e = ler.feedback(ev["correct"].values, np.random.default_rng(seed + 100_000).random(len(ev)))
    _, A, B = ler.online_trust(d_cat, fb_d, len(cats))
    adapt_e, _, _ = ler.online_trust(e_cat, fb_e, len(cats), A, B)

    # Replica grids are computed once and reused for every delta (identical to
    # locked_eval_robust at delta = 0.05, which draws the same replica stream).
    rng = np.random.default_rng(10_000 + seed)
    reps = []
    # locked_eval_robust selects the static arm first (K replicas) before the adaptive arm;
    # replay those draws so the adaptive replicas match it exactly.
    for _ in range(K):
        perm = rng.permutation(len(dev)); rng.random(len(perm))
    for _ in range(K):
        perm = rng.permutation(len(dev))
        fb = ler.feedback(d_cor[perm], rng.random(len(perm)))
        pre, _, _ = ler.online_trust(d_cat[perm], fb, len(cats))
        S, E, N = ler.grid_rates(pre, d_conf[perm], d_cor[perm] == 0)
        reps.append((ler.ucb(S, N), E))
    U = np.array([r[0] for r in reps]); Em = np.mean([r[1] for r in reps], axis=0)
    Sc, Ec, Nc = ler.grid_rates(np.ones(len(dev)), d_conf, d_cor == 0)
    Uc = ler.ucb(Sc, Nc)[0]

    rows = []
    for delta in DELTAS:
        feas = (U <= delta).all(0)
        if feas.any():
            i, j = np.unravel_index(np.argmin(np.where(feas, Em, np.inf)), Em.shape)
            s, e = ler.evaluate(ler.GRID[i], ler.GRID[j], adapt_e, e_conf, e_wrong)
        else:
            s, e = 0.0, 1.0
        rows.append(dict(seed=seed, delta=delta, arm="adaptive_tuned", silent=s, escalation=e))
        okc = Uc <= delta
        if okc.any():
            j = int(np.argmin(np.where(okc, Ec[0], np.inf)))
            s, e = ler.evaluate(0.0, ler.GRID[j], np.ones(len(ev)), e_conf, e_wrong)
        else:
            s, e = 0.0, 1.0   # no feasible confidence threshold: escalate everything
        rows.append(dict(seed=seed, delta=delta, arm="confidence_only", silent=s, escalation=e))
        ler.DELTA = delta
        s, e = ler.rejector_arm(dev, ev, seed)
        rows.append(dict(seed=seed, delta=delta, arm="rejector", silent=s, escalation=e))
    ler.DELTA = 0.05
    return rows


def run(path, prefix, K):
    preds = pd.read_csv(path)
    print(f"Loaded {len(preds)} predictions; deltas {DELTAS}; K={K}")
    t0, rows = time.time(), []
    for seed in range(ler.N_SEEDS):
        rows += one_seed(preds, seed, K)
        print(f"  seed {seed} done ({time.time() - t0:.0f}s)")
    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / f"{prefix}_delta_sweep_robust.csv", index=False)
    g = df.groupby(["delta", "arm"]).agg(silent=("silent", "mean"), silent_max=("silent", "max"),
                                         esc=("escalation", "mean"), esc_sd=("escalation", "std")).reset_index()
    print(f"\n{'delta':>6} " + "".join(f"{LABELS[a]:>32}" for a in ARMS))
    for d in DELTAS:
        cells = []
        for a in ARMS:
            r = g[(g.delta == d) & (g.arm == a)].iloc[0]
            cells.append(f"e={r.esc:.3f} s={r.silent:.3f} (max {r.silent_max:.3f})")
        print(f"{d * 100:>5.0f}% " + "".join(f"{c:>32}" for c in cells))
    print(f"\nSaved {prefix}_delta_sweep_robust.csv")


def plot():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    have = [p for p in ORDER if (OUT_DIR / f"{p}_delta_sweep_robust.csv").exists()]
    fig, axes = plt.subplots(2, 3, figsize=(13, 7.5), sharex=True)
    style = {"adaptive_tuned": ("o", "-"), "confidence_only": ("s", "--"), "rejector": ("^", ":")}
    for ax, p in zip(axes.flat, ORDER):
        ax.set_title(TITLES[p], fontsize=11)
        ax.grid(alpha=0.3)
        if p not in have:
            ax.text(0.5, 0.5, "not run", ha="center", transform=ax.transAxes)
            continue
        df = pd.read_csv(OUT_DIR / f"{p}_delta_sweep_robust.csv")
        for a in ARMS:
            g = df[df.arm == a].groupby("delta").escalation.agg(["mean", "std"])
            m, ls = style[a]
            ax.errorbar(g.index * 100, g["mean"], yerr=g["std"], marker=m, linestyle=ls, capsize=3,
                        color="black", markersize=5, label=LABELS[a])
        ax.set_ylim(-0.03, 1.03)
    for ax in axes[1]:
        ax.set_xlabel("Silent-failure ceiling δ (%)")
    for ax in axes[:, 0]:
        ax.set_ylabel("Locked escalation rate")
    axes[0, 0].legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    out = OUT_DIR / "delta_sweep_robust.png"
    fig.savefig(out, dpi=300)
    print(f"Saved {out}")

    summary = {}
    for p in have:
        df = pd.read_csv(OUT_DIR / f"{p}_delta_sweep_robust.csv")
        g = df.groupby(["arm", "delta"]).agg(esc=("escalation", "mean"), silent=("silent", "mean"),
                                             silent_max=("silent", "max"),
                                             over=("silent", lambda s: 0)).reset_index()
        summary[p] = {a: {f"{d:.2f}": {"escalation": float(r.esc), "silent_mean": float(r.silent),
                                         "silent_max": float(r.silent_max),
                                         "seeds_over_delta": int((df[(df.arm == a) & (df.delta == d)].silent > d).sum())}
                          for d, r in g[g.arm == a].set_index("delta").iterrows()} for a in ARMS}
    with open(OUT_DIR / "delta_sweep_robust_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved delta_sweep_robust_summary.json")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--plot":
        plot()
    elif len(sys.argv) >= 3:
        run(sys.argv[1], sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 20)
    else:
        print(__doc__)
