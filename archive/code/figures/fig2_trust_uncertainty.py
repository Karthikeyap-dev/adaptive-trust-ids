"""fig2: Adaptive Trust Engine per-category trust with uncertainty bars, NSL-KDD."""
from pathlib import Path
import numpy as np
from plot_style import apply_style, COLOR_TRUST
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FIG_DIR = PROJECT_ROOT / "outputs" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)
apply_style()

CATEGORIES = ["dos", "probe", "r2l", "normal", "u2r"]
TRUST = [0.8909, 0.7707, 0.6863, 0.5983, 0.5710]
UNCERTAINTY_SD = [np.sqrt(0.000483), np.sqrt(0.000879), np.sqrt(0.011228), np.sqrt(0.001196), np.sqrt(0.031004)]


def main():
    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    x = np.arange(len(CATEGORIES))
    ax.bar(x, TRUST, yerr=UNCERTAINTY_SD, capsize=4, color=COLOR_TRUST)
    ax.set_xticks(x)
    ax.set_xticklabels(CATEGORIES)
    ax.set_ylabel("Trust score")
    ax.set_title("Adaptive Trust Engine: per-category trust ($\\pm$ SD), NSL-KDD")
    ax.set_ylim(0, 1.05)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig2_trust_uncertainty.pdf", bbox_inches="tight")
    fig.savefig(FIG_DIR / "fig2_trust_uncertainty.png", bbox_inches="tight", dpi=300)
    plt.close(fig)
    print("Saved fig2_trust_uncertainty.pdf / .png")


if __name__ == "__main__":
    main()
