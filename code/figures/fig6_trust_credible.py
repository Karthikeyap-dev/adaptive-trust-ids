"""
Figure 6: per-category adaptive trust, NSL-KDD (RandomForest).

Shows each category's posterior mean trust (dot) with its 95% Beta credible
interval (line), and the effective evidence alpha + beta behind it (right).
Same values as fig2_trust_uncertainty.py; the Beta parameters are recovered
exactly from each category's posterior mean and variance (Equation 3).
No title inside the image (the journal requires titles in the caption).

Run from the project's code/ folder:
    python3 figures/fig6_trust_credible.py
Outputs code/figures/fig6_trust_credible.png (300 dpi) and .pdf
"""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import beta

FIG = Path(__file__).resolve().parent
plt.rcParams.update({"font.family": "serif",
                     "font.serif": ["Times New Roman", "Liberation Serif", "DejaVu Serif"],
                     "font.size": 9})

# Values from fig2_trust_uncertainty.py (posterior mean and variance per category)
CATEGORIES = ["dos", "probe", "r2l", "normal", "u2r"]
MEAN = np.array([0.8909, 0.7707, 0.6863, 0.5983, 0.5710])
VAR = np.array([0.000483, 0.000879, 0.011228, 0.001196, 0.031004])

# Beta(alpha, beta) from mean m and variance v: alpha + beta = m(1 - m) / v - 1
total = MEAN * (1 - MEAN) / VAR - 1
a, b = MEAN * total, (1 - MEAN) * total
lo, hi = beta.ppf(0.025, a, b), beta.ppf(0.975, a, b)

order = np.argsort(MEAN)                 # lowest trust at the bottom
y = np.arange(len(CATEGORIES))
DARK, LIGHT = "#a50f15", "#fcbba1"       # same red family as Figures 4 and 5

fig, ax = plt.subplots(figsize=(3.6, 2.3))
for yi, k in zip(y, order):
    ax.plot([lo[k], hi[k]], [yi, yi], color=LIGHT, lw=6, solid_capstyle="round", zorder=1)
    ax.plot([lo[k], hi[k]], [yi, yi], color=DARK, lw=1.0, zorder=2)
    ax.scatter(MEAN[k], yi, s=28, color=DARK, zorder=3)
    ax.text(1.03, yi, f"{total[k]:.0f}", va="center", ha="left", fontsize=8, transform=ax.get_yaxis_transform())
ax.text(1.03, len(y) - 0.35, "α+β", va="bottom", ha="left", fontsize=8, style="italic", transform=ax.get_yaxis_transform())
ax.set_yticks(y); ax.set_yticklabels([CATEGORIES[k] for k in order])
ax.set_xlim(0, 1); ax.set_ylim(-0.6, len(y) - 0.4)
ax.set_xlabel("Trust (posterior mean, 95% credible interval)")
ax.grid(axis="x", alpha=0.3); ax.set_axisbelow(True)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
fig.tight_layout(pad=0.3)
for ext in ("png", "pdf"):
    fig.savefig(FIG / f"fig6_trust_credible.{ext}", dpi=300, bbox_inches="tight", pad_inches=0.03)
print("Saved fig6_trust_credible.png/.pdf to", FIG)
for k in range(len(CATEGORIES)):
    print(f"  {CATEGORIES[k]:>7}: mean {MEAN[k]:.3f}  95% CI [{lo[k]:.3f}, {hi[k]:.3f}]  alpha+beta {total[k]:.0f}")
