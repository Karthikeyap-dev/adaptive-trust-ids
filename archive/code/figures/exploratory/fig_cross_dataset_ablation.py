"""
fig3: Cross-dataset ablation, escalation rate x (dataset x policy).
FIX: log-scale y-axis so CICIDS2017's near-zero bars are visible, plus
direct value annotations on every bar.
"""
from pathlib import Path
import numpy as np
from plot_style import apply_style, COLOR_NSLKDD, COLOR_UNSW, COLOR_CICIDS
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FIG_DIR = PROJECT_ROOT / "outputs" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)
apply_style()

DATA = {
    "NSL-KDD": {"Static": 1.0000, "Adaptive": 0.6934, "Adaptive+\nOptimized": 0.6929},
    "UNSW-NB15": {"Static": 1.0000, "Adaptive": 0.4880, "Adaptive+\nOptimized": 0.3233},
    "CICIDS2017": {"Static": 0.0017, "Adaptive": 0.0017, "Adaptive+\nOptimized": 0.0002},
}
COLORS = {"NSL-KDD": COLOR_NSLKDD, "UNSW-NB15": COLOR_UNSW, "CICIDS2017": COLOR_CICIDS}
POLICIES = ["Static", "Adaptive", "Adaptive+\nOptimized"]


def main():
    datasets = list(DATA.keys())
    x = np.arange(len(POLICIES))
    width = 0.25

    fig, ax = plt.subplots(figsize=(7.16, 4.2))
    for i, name in enumerate(datasets):
        vals = [DATA[name][p] * 100 for p in POLICIES]
        vals_display = [max(v, 0.01) for v in vals]
        bars = ax.bar(x + (i - 1) * width, vals_display, width, label=name, color=COLORS[name])
        for bar, real_v in zip(bars, vals):
            label = f"{real_v:.2f}%" if real_v >= 0.01 else f"{real_v:.3f}%"
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.15, label,
                    ha="center", fontsize=6.5, rotation=90 if real_v < 1 else 0)

    ax.set_yscale("log")
    ax.set_ylim(0.005, 500)
    ax.set_xticks(x)
    ax.set_xticklabels(POLICIES)
    ax.set_ylabel("Escalation rate (%, log scale)")
    ax.set_title("Three-way ablation: escalation rate across datasets")
    ax.legend(loc="upper center", ncol=3, framealpha=0.95, fontsize=8.5, bbox_to_anchor=(0.5, 1.0))

    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_cross_dataset_ablation.pdf", bbox_inches="tight")
    fig.savefig(FIG_DIR / "fig_cross_dataset_ablation.png", bbox_inches="tight", dpi=300)
    plt.close(fig)
    print("Saved fig_cross_dataset_ablation.pdf / .png")


if __name__ == "__main__":
    main()
