"""fig_beta_trust_concept: Illustrative Beta-distribution belief states."""
from pathlib import Path
import numpy as np
from scipy import stats
from plot_style import apply_style
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FIG_DIR = PROJECT_ROOT / "outputs" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)
apply_style()

X = np.linspace(0, 1, 500)
DISTRIBUTIONS = [
    (1, 1, "gray", "Beta(1, 1) -- no evidence yet\n(uniform prior, trust=0.50)"),
    (8, 30, "#dc2626", "Beta(8, 30) -- some evidence,\nlow trust (mean=0.21)"),
    (180, 22, "#059669", "Beta(180, 22) -- high evidence,\nhigh trust (mean=0.89)"),
]


def main():
    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    for a, b, color, label in DISTRIBUTIONS:
        y = stats.beta.pdf(X, a, b)
        ax.plot(X, y, color=color, linewidth=1.8, label=label)
        ax.fill_between(X, y, alpha=0.15, color=color)
    ax.set_xlabel("Trust value")
    ax.set_ylabel("Probability density")
    ax.set_title("Beta-distribution belief states in the Adaptive Trust Engine")
    ax.legend(loc="upper center", fontsize=8)
    ax.set_xlim(0, 1)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_beta_trust_concept.pdf", bbox_inches="tight")
    fig.savefig(FIG_DIR / "fig_beta_trust_concept.png", bbox_inches="tight", dpi=300)
    plt.close(fig)
    print("Saved fig_beta_trust_concept.pdf / .png")


if __name__ == "__main__":
    main()
