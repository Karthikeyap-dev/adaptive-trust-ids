"""
Figures 4 and 5 (red palette).

Figure 4: per-category metrics heatmap, NSL-KDD (RandomForest).
Figure 5: row-normalized confusion matrices for all three datasets,
          three equal-size panels in one row.

Palette: sequential "Reds" (0 = white, 1 = dark red). Unlike the old
red-yellow-blue map, zero cells are white instead of bright red, so the
eye goes to the large values, and the figure still reads in greyscale.

Run from the project's code/ folder:
    python3 figures/fig4_fig5_reds.py
Outputs (PNG at 300 dpi + vector PDF) are written to code/figures/.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CODE = Path(__file__).resolve().parent.parent
OUT = CODE / "outputs"
FIG = CODE / "figures"
CMAP = "Reds"
plt.rcParams.update({"font.family": "serif",
                     "font.serif": ["Times New Roman", "Liberation Serif", "DejaVu Serif"],
                     "font.size": 9})


def text_color(v):
    return "white" if v > 0.6 else "black"


# ---------------- Figure 4 ----------------
d = pd.read_csv(OUT / "baseline_predictions.csv")
n = len(d)
cats = ["normal", "dos", "probe", "r2l", "u2r"]
mets = ["Precision", "Recall", "F1", "Specificity", "MCC"]
M, sup = [], []
for c in cats:
    tp = ((d.pred_category == c) & (d.true_category == c)).sum()
    fp = ((d.pred_category == c) & (d.true_category != c)).sum()
    fn = ((d.pred_category != c) & (d.true_category == c)).sum()
    tn = n - tp - fp - fn
    p, r = tp / (tp + fp), tp / (tp + fn)
    f = 2 * p * r / (p + r)
    sp = tn / (tn + fp)
    mcc = (tp * tn - fp * fn) / np.sqrt(float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))
    M.append([p, r, f, sp, mcc]); sup.append(tp + fn)
M = np.array(M)
fig, ax = plt.subplots(figsize=(6.0, 3.0))
im = ax.imshow(M, cmap=CMAP, vmin=0, vmax=1, aspect="auto")
for i in range(5):
    for j in range(5):
        v = M[i, j]
        ax.text(j, i, f"{v:.3f}" if v < 0.1 else f"{v:.2f}", ha="center", va="center",
                fontsize=9, color=text_color(v))
ax.set_xticks(range(5)); ax.set_xticklabels(mets)
ax.set_yticks(range(5)); ax.set_yticklabels([f"{c} (n={s})" for c, s in zip(cats, sup)])
ax.tick_params(length=0)
for s in ax.spines.values():
    s.set_visible(False)
cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02); cb.outline.set_visible(False)
fig.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(FIG / f"fig4_per_category_reds.{ext}", dpi=300, bbox_inches="tight", pad_inches=0.03)
plt.close(fig)


# ---------------- Figure 5 ----------------
def confusion(fname, order=None):
    d = pd.read_csv(OUT / fname)
    labs = order or sorted(d.true_category.unique())
    m = pd.crosstab(d.true_category, d.pred_category).reindex(index=labs, columns=labs, fill_value=0).values.astype(float)
    return m / m.sum(1, keepdims=True), labs


panels = [("baseline_predictions.csv", "(a) NSL-KDD", ["dos", "normal", "probe", "r2l", "u2r"]),
          ("cicids_baseline_predictions.csv", "(b) CICIDS2017", None),
          ("unsw_baseline_predictions.csv", "(c) UNSW-NB15", None)]
fig, axes = plt.subplots(1, 3, figsize=(7.2, 3.15), gridspec_kw={"wspace": 0.55})
for ax, (f, title, order) in zip(axes, panels):
    m, labs = confusion(f, order)
    k = len(labs)
    im = ax.imshow(m, cmap=CMAP, vmin=0, vmax=1)          # every panel is the same square size
    fs = 6.5 if k <= 6 else 4.6
    for i in range(k):
        for j in range(k):
            if m[i, j] >= 0.005:
                ax.text(j, i, f"{m[i, j]:.2f}", ha="center", va="center", fontsize=fs, color=text_color(m[i, j]))
    short = [l.replace("Reconnaissance", "Recon.").replace("brute_force", "brute force").replace("web_attack", "web attack")
             for l in labs]
    tfs = 6.5 if k <= 6 else 5.5
    ax.set_xticks(range(k)); ax.set_xticklabels(short, rotation=55, ha="right", fontsize=tfs)
    ax.set_yticks(range(k)); ax.set_yticklabels(short, fontsize=tfs)
    ax.tick_params(length=0, pad=1.5)
    ax.set_title(title, fontsize=9, pad=4)
    ax.set_xlabel("Predicted", fontsize=7.5, labelpad=1)
axes[0].set_ylabel("True", fontsize=7.5)
cax = fig.add_axes([0.925, 0.25, 0.012, 0.55])
cb = fig.colorbar(im, cax=cax); cb.outline.set_visible(False); cb.ax.tick_params(labelsize=6.5)
for ext in ("png", "pdf"):
    fig.savefig(FIG / f"fig5_confusion_reds.{ext}", dpi=300, bbox_inches="tight", pad_inches=0.03)
plt.close(fig)
print("Saved fig4_per_category_reds.png/.pdf and fig5_confusion_reds.png/.pdf to", FIG)
