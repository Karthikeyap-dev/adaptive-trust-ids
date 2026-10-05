"""fig3: Trust evolution over feedback events (recovery-from-burst trace)."""
from pathlib import Path
from plot_style import apply_style, COLOR_TRUST
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FIG_DIR = PROJECT_ROOT / "outputs" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)
apply_style()

STEPS = [1, 12, 19, 21, 29, 40, 51, 62, 76, 87, 99, 110, 124, 135, 148, 160, 180, 210, 240, 265]
TRUST = [0.667, 0.751, 0.801, 0.666, 0.714, 0.820, 0.750, 0.801, 0.853, 0.827, 0.860, 0.870, 0.845, 0.843, 0.863, 0.868, 0.872, 0.874, 0.875, 0.876]
PRE_BURST = 0.8912


def main():
    fig, ax = plt.subplots(figsize=(6.8, 3.4))
    ax.plot(STEPS, TRUST, color=COLOR_TRUST, linewidth=1.8, marker="o", markersize=3)
    ax.axhline(PRE_BURST, color="#059669", linestyle="--", linewidth=1.2, label=f"Pre-burst trust ({PRE_BURST:.3f})")
    ax.axhline(PRE_BURST * 0.9, color="#dc2626", linestyle=":", linewidth=1.2, label="90% recovery threshold")
    ax.axvspan(0, 100, alpha=0.08, color="red", label="Bad-feedback burst (100 events)")
    ax.set_xlabel("Feedback event")
    ax.set_ylabel("Trust (dos category)")
    ax.set_title("Trust recovery after a 100-event bad-feedback burst, NSL-KDD")
    ax.legend(loc="lower right", fontsize=8)
    ax.set_ylim(0.5, 0.95)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig3_trust_evolution.pdf", bbox_inches="tight")
    fig.savefig(FIG_DIR / "fig3_trust_evolution.png", bbox_inches="tight", dpi=300)
    plt.close(fig)
    print("Saved fig3_trust_evolution.pdf / .png")


if __name__ == "__main__":
    main()
