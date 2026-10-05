"""
fig9: Archetype stress test, silent failure rate x (dataset x archetype).
FIX: Novice bars (all exactly 0.00%) are drawn as a thin visible sliver
with an explicit "0.00%" label rather than omitted.
"""
from pathlib import Path
import numpy as np
from plot_style import apply_style, COLOR_NOVICE, COLOR_EXPERT, COLOR_COMPLACENT
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FIG_DIR = PROJECT_ROOT / "outputs" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)
apply_style()

DATA = {
    "NSL-KDD":    {"Novice": 0.0000, "Expert": 0.0504, "Complacent": 0.1927},
    "UNSW-NB15":  {"Novice": 0.0000, "Expert": 0.0012, "Complacent": 0.0045},
    "CICIDS2017": {"Novice": 0.0000, "Expert": 0.0002, "Complacent": 0.0002},
}
SAFETY_CEILING = 0.05
ARCHETYPES = ["Novice", "Expert", "Complacent"]
COLORS = {"Novice": COLOR_NOVICE, "Expert": COLOR_EXPERT, "Complacent": COLOR_COMPLACENT}
MIN_VISIBLE_HEIGHT = 0.35


def main():
    datasets = list(DATA.keys())
    x = np.arange(len(datasets))
    width = 0.25

    fig, ax = plt.subplots(figsize=(7.16, 3.8))
    for i, archetype in enumerate(ARCHETYPES):
        real_vals = [DATA[d][archetype] * 100 for d in datasets]
        display_vals = [max(v, MIN_VISIBLE_HEIGHT) if v == 0 else v for v in real_vals]
        bars = ax.bar(x + (i - 1) * width, display_vals, width, label=archetype, color=COLORS[archetype],
                       edgecolor="black", linewidth=0.3)
        for bar, real_v in zip(bars, real_vals):
            label = f"{real_v:.2f}%"
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3, label,
                    ha="center", fontsize=7)

    ax.axhline(SAFETY_CEILING * 100, color="black", linestyle="--", linewidth=1.2, label="Safety ceiling (5%)")
    ax.set_xticks(x)
    ax.set_xticklabels(datasets)
    ax.set_ylabel("Silent failure rate (%)")
    ax.set_title("Archetype stress test: silent failure rate vs. safety ceiling\n(all archetypes evaluated on all datasets; Novice is exactly 0.00% throughout)", fontsize=9.5)
    ax.legend(loc="upper right", ncol=2, fontsize=8)
    ax.set_ylim(0, 22)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig9_archetype_stress_test.pdf", bbox_inches="tight")
    fig.savefig(FIG_DIR / "fig9_archetype_stress_test.png", bbox_inches="tight", dpi=300)
    plt.close(fig)
    print("Saved fig9_archetype_stress_test.pdf / .png")


if __name__ == "__main__":
    main()
