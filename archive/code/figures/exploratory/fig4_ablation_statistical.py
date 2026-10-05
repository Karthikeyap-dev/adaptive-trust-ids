"""
fig4: NSL-KDD 10-seed three-way ablation, silent failure + escalation
rate with significance brackets.
FIX: y-axis no longer exceeds the true bound of 1.0 (100%).
"""
from pathlib import Path
from plot_style import apply_style
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FIG_DIR = PROJECT_ROOT / "outputs" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)
apply_style()

SUMMARY = {
    "static_silent_failure_rate": (0.0, 0.0),
    "static_escalation_rate": (1.0, 0.0),
    "adaptive_silent_failure_rate": (0.04786, 0.000916),
    "adaptive_escalation_rate": (0.69605, 0.002520),
    "qpso_silent_failure_rate": (0.04895, 0.001969),
    "qpso_escalation_rate": (0.69300, 0.003960),
}
LABELS = ["Static\ntrust", "Adaptive\ntrust", "Adaptive+\nOptimized"]
COLORS = ["#4338ca", "#dc2626", "#059669"]


def bracket(ax, x1, x2, y, h, text):
    ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], color="black", linewidth=0.8, clip_on=False)
    ax.text((x1 + x2) / 2, y + h * 1.3, text, ha="center", fontsize=7, clip_on=False)


def main():
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 3.0))

    ax = axes[0]
    means = [SUMMARY["static_silent_failure_rate"][0], SUMMARY["adaptive_silent_failure_rate"][0], SUMMARY["qpso_silent_failure_rate"][0]]
    stds = [SUMMARY["static_silent_failure_rate"][1], SUMMARY["adaptive_silent_failure_rate"][1], SUMMARY["qpso_silent_failure_rate"][1]]
    ax.bar(LABELS, means, yerr=stds, capsize=3, color=COLORS)
    ax.set_title("Silent failure rate", fontsize=9)
    ax.set_ylabel("Rate")
    ax.set_ylim(0, 0.075)
    bracket(ax, 0, 1, 0.056, 0.004, "**")
    bracket(ax, 1, 2, 0.050, 0.004, "n.s.")

    ax = axes[1]
    means = [SUMMARY["static_escalation_rate"][0], SUMMARY["adaptive_escalation_rate"][0], SUMMARY["qpso_escalation_rate"][0]]
    stds = [SUMMARY["static_escalation_rate"][1], SUMMARY["adaptive_escalation_rate"][1], SUMMARY["qpso_escalation_rate"][1]]
    ax.bar(LABELS, means, yerr=stds, capsize=3, color=COLORS)
    ax.set_title("Escalation rate", fontsize=9)
    ax.set_ylabel("Rate")
    ax.set_ylim(0, 1.0)
    bracket(ax, 0, 1, 0.90, 0.03, "**")
    bracket(ax, 1, 2, 0.80, 0.03, "*")

    fig.suptitle("Three-way ablation (mean $\\pm$ SD, n=10 seeds)", fontsize=10, y=1.03)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig4_ablation_statistical.pdf", bbox_inches="tight")
    fig.savefig(FIG_DIR / "fig4_ablation_statistical.png", bbox_inches="tight", dpi=300)
    plt.close(fig)
    print("Saved fig4_ablation_statistical.pdf / .png")


if __name__ == "__main__":
    main()
