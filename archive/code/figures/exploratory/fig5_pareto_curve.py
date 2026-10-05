"""fig5: Risk-workload trade-off (Pareto sweep), NSL-KDD, RandomForest."""
from pathlib import Path
from plot_style import apply_style, COLOR_TRUST
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FIG_DIR = PROJECT_ROOT / "outputs" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)
apply_style()

CEILINGS = [0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.1, 0.15, 0.2, 0.3]
ESCALATION = [0.7732, 0.7366, 0.7133, 0.6929, 0.6746, 0.6746, 0.2815, 0.2133, 0.1152, 0.0246]


def main():
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    x = [c * 100 for c in CEILINGS]
    y = [e * 100 for e in ESCALATION]
    ax.plot(x, y, marker="o", color=COLOR_TRUST, linewidth=2, markersize=6)
    ax.annotate("knee: 8-10%", xy=(9, 45), fontsize=8, ha="center",
                arrowprops=dict(arrowstyle="->", lw=0.8), xytext=(18, 55))
    ax.set_xlabel("Acceptable silent-failure ceiling (%)")
    ax.set_ylabel("Resulting escalation rate (%)")
    ax.set_title("Risk-workload trade-off (NSL-KDD, RandomForest)")

    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig5_pareto_curve.pdf", bbox_inches="tight")
    fig.savefig(FIG_DIR / "fig5_pareto_curve.png", bbox_inches="tight", dpi=300)
    plt.close(fig)
    print("Saved fig5_pareto_curve.pdf / .png")


if __name__ == "__main__":
    main()
