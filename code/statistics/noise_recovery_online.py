"""
Online, multi-seed re-run of the Section 5.6 diagnostics (replaces sensitivity_analysis_v2.py).

A. Feedback-accuracy sweep (60-100%): each alert is arbitrated with the trust held BEFORE its
   own feedback (online), under the original fixed thresholds (0.75, 0.80) and under the
   seed-matched tuned NSL-KDD/RF policy from locked_eval_robust.py. (v2 used the end-of-stream
   trust snapshot.)
B. Burst recovery: 100 forced-incorrect feedback events for 'dos' mid-stream; trust before the
   burst, minimum, and good-feedback events needed to return within 90% of the pre-burst value.
   (Same method as v2, now 10 seeds.)
C. Decay sweep, stationary stream: NO drift; second-half std of probe trust (stability).
   (v2 accidentally injected the drift here as well.)
D. Decay sweep with injected mid-stream drift on 'probe' (85% forced errors from the midpoint);
   events needed to move halfway from the trust held JUST BEFORE the drift to the final trust.
   (v2 used the trust after the first probe event, ~0.667, as the pre-drift value.)

Seeds 0-9; each seed reshuffles the NSL-KDD/RF prediction stream and the feedback noise.
Run from the project's code/ folder:   python3 statistics/noise_recovery_online.py
Writes outputs/noise_recovery_online.json
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd

CODE = Path(__file__).resolve().parent.parent
OUT = CODE / "outputs"
SEEDS = range(10)
LAMBDA = 0.995
FIXED = (0.75, 0.80)
DELTA = 0.05


class Trust:
    """Per-category Beta trust; decay applied to that category on each update (Section 3.3)."""
    def __init__(self, lam=LAMBDA):
        self.lam, self.s = lam, {}

    def get(self, c):
        a, b = self.s.get(c, (1.0, 1.0))
        return a / (a + b)

    def update(self, c, ok):
        a, b = self.s.get(c, (1.0, 1.0))
        a, b = a * self.lam, b * self.lam
        self.s[c] = (a + 1, b) if ok else (a, b + 1)


def fb(correct, err, rng):
    return (not correct) if rng.random() < err else bool(correct)


def stream(preds, seed):
    return preds.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def evaluate(th, ch, trust, conf, wrong):
    auto = (trust >= th) & (conf >= ch)
    return float((auto & wrong).sum() / max(auto.sum(), 1)), float(1 - auto.mean())


def ms(v):
    v = np.array([x for x in v if x is not None], float)
    return {"mean": float(v.mean()), "sd": float(v.std(ddof=1)) if len(v) > 1 else 0.0, "n": int(len(v))}


def main():
    preds = pd.read_csv(OUT / "baseline_predictions.csv")
    tuned = {r["seed"]: (r["adaptive_tuned_tau_h"], r["adaptive_tuned_gamma_h"])
             for r in json.load(open(OUT / "nslkdd_rf_locked_eval_robust.json"))["per_seed"]}
    res = {}

    # ---------------- A ----------------
    print("A. Feedback-accuracy sweep (online trust, 10 seeds)")
    print(f"{'accuracy':>9} {'mean trust':>11} {'FIXED esc':>10} {'FIXED sf':>9} {'>δ':>5} {'TUNED esc':>10} {'TUNED sf':>9} {'>δ':>5}")
    res["A_feedback_accuracy"] = {}
    for acc in (0.6, 0.7, 0.8, 0.9, 1.0):
        rows = []
        for s in SEEDS:
            st = stream(preds, s); rng = np.random.default_rng(s); T = Trust()
            cats, cor = st["pred_category"].values, st["correct"].values
            pre = np.empty(len(st))
            for i in range(len(st)):
                pre[i] = T.get(cats[i])
                T.update(cats[i], fb(cor[i], 1 - acc, rng))
            conf, wrong = st["raw_confidence"].values, cor == 0
            f = evaluate(*FIXED, pre, conf, wrong); t = evaluate(*tuned[s], pre, conf, wrong)
            rows.append((float(np.mean(pre)), *f, *t))
        r = np.array(rows)
        res["A_feedback_accuracy"][str(acc)] = {
            "mean_trust": ms(r[:, 0]), "fixed_silent": ms(r[:, 1]), "fixed_escalation": ms(r[:, 2]),
            "fixed_over_budget": int((r[:, 1] > DELTA).sum()),
            "tuned_silent": ms(r[:, 3]), "tuned_escalation": ms(r[:, 4]), "tuned_over_budget": int((r[:, 3] > DELTA).sum())}
        print(f"{acc:>9.0%} {r[:,0].mean():>11.3f} {r[:,2].mean():>10.3f} {r[:,1].mean():>9.4f} {int((r[:,1]>DELTA).sum()):>3}/10"
              f" {r[:,4].mean():>10.3f} {r[:,3].mean():>9.4f} {int((r[:,3]>DELTA).sum()):>3}/10")

    # ---------------- B ----------------
    print("\nB. Burst recovery ('dos', 100 forced-incorrect events, λ=0.995, 10 seeds)")
    before, low, rec = [], [], []
    for s in SEEDS:
        st = stream(preds, s); rng = np.random.default_rng(s); T = Trust()
        start, burst, b0, trace = len(st) // 2, 0, None, []
        for i, (c, ok) in enumerate(zip(st["pred_category"], st["correct"])):
            if c == "dos" and i >= start and burst < 100:
                if burst == 0:
                    b0 = T.get("dos")
                T.update(c, False); burst += 1
            else:
                T.update(c, fb(ok, 0.08, rng))
            if c == "dos" and i >= start:
                trace.append(T.get("dos"))
        after = trace[100:]
        k = next((j for j, v in enumerate(after) if v >= 0.9 * b0), None)
        before.append(b0); low.append(min(trace)); rec.append(k)
    res["B_burst_recovery"] = {"trust_before": ms(before), "trust_min": ms(low), "events_to_recover_90pct": ms(rec),
                               "recovered_seeds": int(sum(x is not None for x in rec))}
    print(f"  trust before {np.mean(before):.3f}±{np.std(before, ddof=1):.3f}  minimum {np.mean(low):.3f}±{np.std(low, ddof=1):.3f}"
          f"  events to recover {np.mean([x for x in rec if x is not None]):.0f}±{np.std([x for x in rec if x is not None], ddof=1):.0f}"
          f"  (recovered on {sum(x is not None for x in rec)}/10 seeds)")

    # ---------------- C & D ----------------
    def probe_trace(st, lam, drift, rng):
        T, mid, tr = Trust(lam), len(st) // 2, []
        for i, (c, ok) in enumerate(zip(st["pred_category"], st["correct"])):
            if drift and c == "probe" and i >= mid:
                ok = rng.random() > 0.85
            T.update(c, fb(ok, 0.08, rng))
            if c == "probe":
                tr.append((i, T.get("probe")))
        return tr, mid

    print("\nC. Decay sweep, stationary (no drift): second-half std of probe trust (lower = more stable)")
    print("D. Decay sweep with drift: events from the pre-drift trust to halfway adaptation (lower = faster)")
    res["C_stationary"], res["D_drift"] = {}, {}
    for lam in (0.90, 0.95, 0.99, 0.995, 0.999, 1.0):
        sd, half = [], []
        for s in SEEDS:
            st = stream(preds, s)
            tr, _ = probe_trace(st, lam, False, np.random.default_rng(s))
            v = [t for _, t in tr]; sd.append(float(np.std(v[len(v) // 2:])))
            tr, mid = probe_trace(st, lam, True, np.random.default_rng(s))
            pre = [t for i, t in tr if i < mid][-1]
            post = [t for i, t in tr if i >= mid]
            target = pre - 0.5 * (pre - post[-1])
            half.append(next((j for j, t in enumerate(post) if t <= target), None))
        res["C_stationary"][str(lam)] = ms(sd)
        res["D_drift"][str(lam)] = ms(half)
        print(f"  λ={lam:<6} C: std {np.mean(sd):.4f}±{np.std(sd, ddof=1):.4f}   D: {np.mean(half):.0f}±{np.std(half, ddof=1):.0f} events")

    with open(OUT / "noise_recovery_online.json", "w") as f:
        json.dump(res, f, indent=2)
    print("\nSaved outputs/noise_recovery_online.json")


if __name__ == "__main__":
    main()
