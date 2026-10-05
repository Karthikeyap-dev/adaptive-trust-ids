"""fig7: Explanation faithfulness, SHAP-guided vs random deletion."""
from pathlib import Path
from plot_style import apply_style
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FIG_DIR = PROJECT_ROOT / "outputs" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)
apply_style()

MEANS = [0.272, 0.039]
STDS = [0.293, 0.102]
LABELS = ["SHAP-guided\nremoval", "Random\nremoval"]


def main():
    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    ax.bar(LABELS, MEANS, yerr=STDS, capsize=5, color=["#4338ca", "#9ca3af"])
    ax.plot([0, 0, 1, 1], [0.58, 0.60, 0.60, 0.15], color="black", linewidth=0.9, clip_on=False)
    ax.text(0.5, 0.61, "***", ha="center", fontsize=11)
    ax.set_ylabel("Mean confidence drop")
    ax.set_title("Explanation faithfulness (deletion test, n=200)")
    ax.set_ylim(0, 0.65)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig7_faithfulness.pdf", bbox_inches="tight")
    fig.savefig(FIG_DIR / "fig7_faithfulness.png", bbox_inches="tight", dpi=300)
    plt.close(fig)
    print("Saved fig7_faithfulness.pdf / .png")


if __name__ == "__main__":
    main()
