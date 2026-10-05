"""
M1: point-estimate threshold selection evaluated under the SAME online protocol as the
robust selection (locked_eval_robust.py), so both columns of Table 9 come from one protocol.

Point-estimate rule: on the development half's own online trust stream, pick the grid policy
with minimum escalation whose POINT ESTIMATE of silent failure is <= delta (no replicas, no
upper bound). Evaluated once on the locked half with online trust, seeds 0-9.

Run from code/optimizer:   python3 point_estimate_online.py
Writes ../outputs/point_estimate_online.json
"""
import json, sys
import numpy as np, pandas as pd
import locked_eval_robust as ler

CONFIGS = [("nslkdd_rf", "baseline_predictions.csv"), ("nslkdd_xgb", "xgb_baseline_predictions.csv"),
           ("unsw_rf", "unsw_baseline_predictions.csv"), ("unsw_xgb", "unsw_xgb_baseline_predictions.csv"),
           ("cicids_rf", "cicids_baseline_predictions.csv"), ("cicids_xgb", "cicids_xgb_baseline_predictions.csv")]
out = {}
for name, f in CONFIGS:
    preds = pd.read_csv(ler.OUT_DIR / f); rows = []
    for seed in range(ler.N_SEEDS):
        sh = preds.sample(frac=1.0, random_state=seed).reset_index(drop=True); h = len(sh) // 2
        dev, ev = sh.iloc[:h].reset_index(drop=True), sh.iloc[h:].reset_index(drop=True)
        cats = sorted(preds["pred_category"].unique()); cidx = {c: i for i, c in enumerate(cats)}
        dc, ec = dev["pred_category"].map(cidx).values, ev["pred_category"].map(cidx).values
        fb_d = ler.feedback(dev["correct"].values, np.random.default_rng(seed).random(len(dev)))
        fb_e = ler.feedback(ev["correct"].values, np.random.default_rng(seed + 100_000).random(len(ev)))
        pre_d, A, B = ler.online_trust(dc, fb_d, len(cats))
        pre_e, _, _ = ler.online_trust(ec, fb_e, len(cats), A, B)
        S, E, N = ler.grid_rates(pre_d, dev["raw_confidence"].values, dev["correct"].values == 0)
        feas = S <= ler.DELTA
        i, j = np.unravel_index(np.argmin(np.where(feas, E, np.inf)), E.shape)
        s, e = ler.evaluate(ler.GRID[i], ler.GRID[j], pre_e, ev["raw_confidence"].values, ev["correct"].values == 0)
        rows.append((s, e))
    r = np.array(rows)
    out[name] = {"escalation_mean": float(r[:, 1].mean()), "silent_mean": float(r[:, 0].mean()),
                 "silent_max": float(r[:, 0].max()), "seeds_over_delta": int((r[:, 0] > ler.DELTA).sum())}
    print(f"{name:<11} escalation {r[:,1].mean():.3f}  silent mean {r[:,0].mean():.3f}  max {r[:,0].max():.3f}  >δ {int((r[:,0]>ler.DELTA).sum())}/10")
json.dump(out, open(ler.OUT_DIR / "point_estimate_online.json", "w"), indent=2)
print("Saved point_estimate_online.json")
