"""
M2: Beta-prior sensitivity on a configuration where the tuned policy actually uses trust
(UNSW-NB15/RandomForest; tuned tau_h ~0.5), with online trust and the same robust selection as
locked_eval_robust.py (K=20 replicas, 95% upper bound), seeds 0-9.
(The original sweep used NSL-KDD/RandomForest, whose tuned policy has tau_h = 0 and ignores trust.)

Run from code/optimizer:   python3 prior_sensitivity_online.py
Writes ../outputs/unsw_rf_prior_sensitivity_online.json
"""
import json
import numpy as np, pandas as pd
import locked_eval_robust as ler

PRIORS = [(1.0, 1.0), (0.5, 0.5), (2.0, 2.0), (5.0, 5.0)]
preds = pd.read_csv(ler.OUT_DIR / "unsw_baseline_predictions.csv")
cats = sorted(preds["pred_category"].unique()); cidx = {c: i for i, c in enumerate(cats)}; n = len(cats)
out = {}
for a0, b0 in PRIORS:
    rows = []
    for seed in range(ler.N_SEEDS):
        sh = preds.sample(frac=1.0, random_state=seed).reset_index(drop=True); h = len(sh) // 2
        dev, ev = sh.iloc[:h].reset_index(drop=True), sh.iloc[h:].reset_index(drop=True)
        dc, ec = dev["pred_category"].map(cidx).values, ev["pred_category"].map(cidx).values
        dconf, dcor = dev["raw_confidence"].values, dev["correct"].values
        P0 = (np.full(n, a0), np.full(n, b0))
        fb_d = ler.feedback(dcor, np.random.default_rng(seed).random(len(dev)))
        fb_e = ler.feedback(ev["correct"].values, np.random.default_rng(seed + 100_000).random(len(ev)))
        _, A, B = ler.online_trust(dc, fb_d, n, *P0)
        pre_e, _, _ = ler.online_trust(ec, fb_e, n, A, B)
        rng = np.random.default_rng(10_000 + seed); feas = None; esc = 0.0
        for _ in range(20):
            perm = rng.permutation(len(dev)); fb = ler.feedback(dcor[perm], rng.random(len(perm)))
            pre, _, _ = ler.online_trust(dc[perm], fb, n, *P0)
            S, E, N = ler.grid_rates(pre, dconf[perm], dcor[perm] == 0)
            ok = ler.ucb(S, N) <= ler.DELTA; feas = ok if feas is None else feas & ok; esc = esc + E / 20
        i, j = np.unravel_index(np.argmin(np.where(feas, esc, np.inf)), esc.shape)
        s, e = ler.evaluate(ler.GRID[i], ler.GRID[j], pre_e, ev["raw_confidence"].values, ev["correct"].values == 0)
        rows.append((s, e, ler.GRID[i]))
    r = np.array(rows)
    key = f"Beta({a0:g},{b0:g})"
    out[key] = {"escalation_mean": float(r[:, 1].mean()), "silent_mean": float(r[:, 0].mean()), "tau_h_mean": float(r[:, 2].mean())}
    print(f"{key:<12} escalation {r[:,1].mean():.4f}  silent {r[:,0].mean():.4f}  tau_h {r[:,2].mean():.3f}")
json.dump(out, open(ler.OUT_DIR / "unsw_rf_prior_sensitivity_online.json", "w"), indent=2)
print("Saved unsw_rf_prior_sensitivity_online.json")
