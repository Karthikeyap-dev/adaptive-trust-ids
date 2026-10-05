"""
Table 7: paired escalation differences (percentage points) and matched-pairs rank-biserial
correlation for tuned adaptive trust vs. fixed thresholds, confidence-only and learned rejector.
Reads outputs/<config>_locked_eval_robust.csv (written by optimizer/locked_eval_robust.py).

Run from code/:   python3 statistics/effect_sizes_robust.py
"""
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats

OUT = Path(__file__).resolve().parent.parent / "outputs"
CONFIGS = ["nslkdd_rf", "nslkdd_xgb", "unsw_rf", "unsw_xgb", "cicids_rf", "cicids_xgb"]
ARMS = [("adaptive_fixed", "Tuned - fixed"), ("confidence_only", "Tuned - confidence-only"), ("rejector", "Tuned - learned rejector")]


def rank_biserial(x, y):
    d = np.asarray(x) - np.asarray(y); d = d[d != 0]
    if len(d) == 0:
        return None
    r = stats.rankdata(np.abs(d))
    return (r[d > 0].sum() - r[d < 0].sum()) / r.sum()


rows = []
for c in CONFIGS:
    df = pd.read_csv(OUT / f"{c}_locked_eval_robust.csv")
    tuned = df["adaptive_tuned_escalation_rate"]
    row = {"configuration": c}
    for arm, label in ARMS:
        other = df[f"{arm}_escalation_rate"]
        diff = (tuned - other).mean() * 100
        rb = rank_biserial(tuned, other)
        row[label] = "0.00 (identical)" if rb is None else f"{diff:+.2f} ({rb:.2f})"
    rows.append(row)
res = pd.DataFrame(rows)
print(res.to_string(index=False))
res.to_csv(OUT / "effect_sizes_robust.csv", index=False)
print("Saved outputs/effect_sizes_robust.csv")
