"""
fig1b: NSL-KDD RandomForest, r2l and u2r only - separated from the main
per-category figure since their near-zero values were nearly invisible
at the same scale, which could read as a rendering error rather than
genuine extreme class imbalance. Same five metrics, real values.
"""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FIG_DIR = PROJECT_ROOT / "outputs" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "Nimbus Roman", "DejaVu Serif"],
    "font.size": 9,
})

CATEGORIES = ["r2l", "u2r"]
METRICS = {
    "Precision":   [0.7647, 0.6000],
    "Recall":      [0.0047, 0.0150],
    "F1":          [0.0094, 0.0293],
    "Specificity": [0.9998, 0.9999],
    "MCC":         [0.0539, 0.0939],
}
COLORS = ["#4338ca", "#dc2626", "#059669", "#d97706", "#7c3aed"]


def main():
    x = np.arange(len(CATEGORIES))
    n_metrics = len(METRICS)
    width = 0.8 / n_metrics

    fig, ax = plt.subplots(figsize=(5.4, 3.8))
    for i, (metric_name, values) in enumerate(METRICS.items()):
        offset = (i - (n_metrics - 1) / 2) * width
        bars = ax.bar(x + offset, values, width, label=metric_name, color=COLORS[i])
        for bar, v in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.015, f"{v:.3f}",
                    ha="center", fontsize=6, rotation=90)

    ax.set_xticks(x)
    ax.set_xticklabels(CATEGORIES)
    ax.set_ylabel("Score")
    ax.set_title("Baseline classifier performance on rare categories\n(NSL-KDD, RandomForest)")
    ax.legend(loc="lower center", ncol=5, fontsize=7, bbox_to_anchor=(0.5, -0.32))
    ax.set_ylim(0, 1.18)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig1b_rare_categories.pdf", bbox_inches="tight")
    fig.savefig(FIG_DIR / "fig1b_rare_categories.png", bbox_inches="tight", dpi=300)
    plt.close(fig)
    print("Saved fig1b_rare_categories.pdf / .png")


if __name__ == "__main__":
    main()
