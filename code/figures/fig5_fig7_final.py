"""
Figures 5 and 7 (final layout): panel labels placed BELOW each panel; Figure 5's colour bar
matches the height of the confusion matrices.

Run from code/:   python3 figures/fig5_fig7_final.py
Writes code/figures/fig5_confusion_reds.(png|pdf) and code/figures/fig7_risk_workload.(png|pdf)
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CODE = Path(__file__).resolve().parent.parent
OUT, FIG = CODE / "outputs", CODE / "figures"
plt.rcParams.update({"font.family": "serif",
                     "font.serif": ["Times New Roman", "Liberation Serif", "DejaVu Serif"], "font.size": 9})


def save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(FIG / f"{name}.{ext}", dpi=300, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


def labels_below(fig, axes, labels, gap=0.012, size=9):
    """Place one label under each axis, all on a common baseline below the lowest tick labels."""
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    inv = fig.transFigure.inverted()
    bottom = min(inv.transform((0, ax.get_tightbbox(r).y0))[1] for ax in axes)
    for ax, lab in zip(axes, labels):
        p = ax.get_position()
        fig.text((p.x0 + p.x1) / 2, bottom - gap, lab, ha="center", va="top", fontsize=size)


# ---------------- Figure 5: confusion matrices ----------------
def confusion(fname, order=None):
    d = pd.read_csv(OUT / fname)
    labs = order or sorted(d.true_category.unique())
    m = pd.crosstab(d.true_category, d.pred_category).reindex(index=labs, columns=labs, fill_value=0).values.astype(float)
    return m / m.sum(1, keepdims=True), labs


panels = [("baseline_predictions.csv", "(a) NSL-KDD", ["dos", "normal", "probe", "r2l", "u2r"]),
          ("cicids_baseline_predictions.csv", "(b) CICIDS2017", None),
          ("unsw_baseline_predictions.csv", "(c) UNSW-NB15", None)]
fig, axes = plt.subplots(1, 3, figsize=(7.2, 3.0), gridspec_kw={"wspace": 0.6})
for ax, (f, _, order) in zip(axes, panels):
    m, labs = confusion(f, order); k = len(labs)
    im = ax.imshow(m, cmap="Reds", vmin=0, vmax=1)
    fs = 6.5 if k <= 6 else 4.6
    for i in range(k):
        for j in range(k):
            if m[i, j] >= 0.005:
                ax.text(j, i, f"{m[i, j]:.2f}", ha="center", va="center", fontsize=fs,
                        color="white" if m[i, j] > 0.6 else "black")
    short = [l.replace("Reconnaissance", "Recon.").replace("brute_force", "brute force").replace("web_attack", "web attack") for l in labs]
    tfs = 6.5 if k <= 6 else 5.5
    ax.set_xticks(range(k)); ax.set_xticklabels(short, rotation=55, ha="right", fontsize=tfs)
    ax.set_yticks(range(k)); ax.set_yticklabels(short, fontsize=tfs)
    ax.tick_params(length=0, pad=1.5)
    ax.set_xlabel("Predicted", fontsize=7.5, labelpad=1)
axes[0].set_ylabel("True", fontsize=7.5)
fig.canvas.draw()
p = axes[2].get_position()                              # colour bar exactly as tall as the matrices
cax = fig.add_axes([p.x1 + 0.015, p.y0, 0.012, p.height])
cb = fig.colorbar(im, cax=cax); cb.outline.set_visible(False); cb.ax.tick_params(labelsize=6.5)
labels_below(fig, list(axes), [t for _, t, _ in panels])
save(fig, "fig5_confusion_reds")

# ---------------- Figure 7: risk-workload curves ----------------
ORDER = [("nslkdd_rf", "(a) NSL-KDD / RandomForest"), ("nslkdd_xgb", "(b) NSL-KDD / XGBoost"),
         ("unsw_rf", "(c) UNSW-NB15 / RandomForest"), ("unsw_xgb", "(d) UNSW-NB15 / XGBoost")]
ARMS = [("adaptive_tuned", "Adaptive trust (tuned)", "o", "-"), ("confidence_only", "Confidence-only", "s", "--"),
        ("rejector", "Learned rejector", "^", ":")]
fig, axes = plt.subplots(2, 2, figsize=(7.2, 6.0), sharey=True, gridspec_kw={"hspace": 0.62, "wspace": 0.12})
for ax, (cfg, _) in zip(axes.flat, ORDER):
    df = pd.read_csv(OUT / f"{cfg}_delta_sweep_robust.csv")
    for a, l, mk, ls in ARMS:
        g = df[df.arm == a].groupby("delta").escalation.agg(["mean", "std"])
        ax.errorbar(g.index * 100, g["mean"], yerr=g["std"], marker=mk, linestyle=ls, color="black",
                    ms=4.5, capsize=2.5, lw=1.2, label=l)
    ax.grid(alpha=0.3); ax.set_ylim(-0.03, 1.03)
    ax.set_xlabel("Silent-failure ceiling δ (%)")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
for ax in axes[:, 0]:
    ax.set_ylabel("Locked escalation rate")
axes[0, 0].legend(fontsize=8.5, frameon=False, loc="upper right")
for row in range(2):                                    # labels under each row of panels
    labels_below(fig, list(axes[row]), [t for _, t in ORDER[2 * row: 2 * row + 2]], gap=0.008, size=9.5)
save(fig, "fig7_risk_workload")
print("Saved fig5_confusion_reds and fig7_risk_workload (png, pdf) to", FIG)
