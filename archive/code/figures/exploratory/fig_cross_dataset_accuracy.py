"""fig_cross_dataset_accuracy: Baseline accuracy comparison, RandomForest, three datasets."""
from pathlib import Path
import numpy as np
from plot_style import apply_style, COLOR_NSLKDD, COLOR_UNSW, COLOR_CICIDS
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FIG_DIR = PROJECT_ROOT / "outputs" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)
apply_style()

DATASETS = ["NSL-KDD", "UNSW-NB15", "CICIDS2017"]
ACCURACY = [0.7340, 0.7497, 0.9989]
CI_LOW = [0.7282, 0.7468, 0.9988]
CI_HIGH = [0.7397, 0.7527, 0.9990]
COLORS = [COLOR_NSLKDD, COLOR_UNSW, COLOR_CICIDS]


def main():
    fig, ax = plt.subplots(figsize=(5.4, 3.6))
    x = np.arange(len(DATASETS))
    errors = [[a - lo for a, lo in zip(ACCURACY, CI_LOW)], [hi - a for a, hi in zip(ACCURACY, CI_HIGH)]]
    bars = ax.bar(x, ACCURACY, yerr=errors, capsize=5, color=COLORS)
    for bar, acc in zip(bars, ACCURACY):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02, f"{acc:.4f}",
                ha="center", fontsize=8.5)
    ax.set_xticks(x)
    ax.set_xticklabels(DATASETS)
    ax.set_ylabel("Accuracy")
    ax.set_title("Baseline classification accuracy across datasets\n(RandomForest, 95% Wilson CI)", fontsize=10)
    ax.set_ylim(0, 1.1)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_cross_dataset_accuracy.pdf", bbox_inches="tight")
    fig.savefig(FIG_DIR / "fig_cross_dataset_accuracy.png", bbox_inches="tight", dpi=300)
    plt.close(fig)
    print("Saved fig_cross_dataset_accuracy.pdf / .png")


if __name__ == "__main__":
    main()
