"""fig6: Calibration reliability diagram, NSL-KDD."""
from pathlib import Path
from plot_style import apply_style
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FIG_DIR = PROJECT_ROOT / "outputs" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)
apply_style()

BEFORE_CONF = [0.387, 0.473, 0.557, 0.649, 0.765, 0.857, 0.991]
BEFORE_ACC = [0.538, 0.300, 0.171, 0.216, 0.519, 0.263, 0.876]
AFTER_CONF = [0.0, 0.333, 0.691, 0.785, 0.897, 0.994]
AFTER_ACC = [0.387, 0.290, 0.171, 0.131, 0.326, 0.796]


def main():
    fig, ax = plt.subplots(figsize=(5.2, 5.0))
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1, label="Perfect calibration")
    ax.plot(BEFORE_CONF, BEFORE_ACC, marker="o", color="#dc2626", linewidth=1.6, label="Before calibration (ECE=0.1895)")
    ax.plot(AFTER_CONF, AFTER_ACC, marker="s", color="#4338ca", linewidth=1.6, label="After isotonic calibration (ECE=0.2269)")
    ax.set_xlabel("Mean predicted confidence")
    ax.set_ylabel("Empirical accuracy")
    ax.set_title("Reliability diagram, NSL-KDD (RandomForest)")
    ax.legend(loc="upper left", fontsize=8)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig6_calibration_reliability.pdf", bbox_inches="tight")
    fig.savefig(FIG_DIR / "fig6_calibration_reliability.png", bbox_inches="tight", dpi=300)
    plt.close(fig)
    print("Saved fig6_calibration_reliability.pdf / .png")


if __name__ == "__main__":
    main()
