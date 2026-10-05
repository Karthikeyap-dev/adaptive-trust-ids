"""fig1: Baseline classifier per-category performance, NSL-KDD RandomForest."""
from pathlib import Path
import numpy as np
from plot_style import apply_style
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FIG_DIR = PROJECT_ROOT / "outputs" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)
apply_style()

CATEGORIES = ["dos", "normal", "probe", "r2l", "u2r"]
PRECISION = [0.9603, 0.6337, 0.8302, 0.7647, 0.6000]
RECALL = [0.7560, 0.9735, 0.5936, 0.0047, 0.0150]
F1 = [0.8460, 0.7677, 0.6922, 0.0094, 0.0293]


def main():
    x = np.arange(len(CATEGORIES))
    width = 0.26
    fig, ax = plt.subplots(figsize=(6.8, 3.6))
    ax.bar(x - width, PRECISION, width, label="Precision", color="#4338ca")
    ax.bar(x, RECALL, width, label="Recall", color="#dc2626")
    ax.bar(x + width, F1, width, label="F1-score", color="#059669")
    ax.set_xticks(x)
    ax.set_xticklabels(CATEGORIES)
    ax.set_ylabel("Score")
    ax.set_title("Baseline classifier per-category performance (NSL-KDD, RandomForest)")
    ax.legend(loc="upper right")
    ax.set_ylim(0, 1.05)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig1_baseline_performance.pdf", bbox_inches="tight")
    fig.savefig(FIG_DIR / "fig1_baseline_performance.png", bbox_inches="tight", dpi=300)
    plt.close(fig)
    print("Saved fig1_baseline_performance.pdf / .png")


if __name__ == "__main__":
    main()
