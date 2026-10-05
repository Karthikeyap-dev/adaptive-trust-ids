from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = PROJECT_ROOT / "outputs"
FIG_DIR = OUT_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "Nimbus Roman", "DejaVu Serif"],
    "font.size": 9,
})

# (result filename, output figure name, display title)
DATASETS = [
    ("baseline_results.json", "fig8_confusion_matrix_v2", "NSL-KDD"),
    ("unsw_baseline_results.json", "fig_unsw_confusion_matrix", "UNSW-NB15"),
    ("cicids_baseline_results.json", "fig_cicids_confusion_matrix", "CICIDS2017"),
]


def plot_confusion(result_file, out_name, title):
    path = OUT_DIR / result_file
    if not path.exists():
        print(f"  [skip] {title} - {result_file} not found")
        return
    results = json.load(open(path))
    cm = np.array(results["confusion_matrix"])
    cats = results["category_order"]
    cm_norm = cm / cm.sum(axis=1, keepdims=True)
    n = len(cats)

    # Scale figure size and font sizes with category count so larger
    # matrices (e.g. UNSW-NB15's 10x10) get proportionally more room
    # instead of being crammed into the same fixed size as a 5x5 or 6x6.
    base_size = 3.6
    size = base_size + max(0, n - 6) * 0.45
    tick_fontsize = 9 if n <= 6 else max(6, 9 - (n - 6) * 0.4)
    annot_fontsize = 6.5 if n <= 6 else max(4.5, 6.5 - (n - 6) * 0.35)

    fig, ax = plt.subplots(figsize=(size, size * 0.89))
    # RdYlBu: red (low) -> yellow (mid) -> blue (high). Diagonal (correct
    # predictions) should read as blue/strong, off-diagonal errors as red.
    im = ax.imshow(cm_norm, cmap="RdYlBu", vmin=0, vmax=1)
    ax.set_xticks(range(n))
    ax.set_xticklabels(cats, rotation=45, ha="right", fontsize=tick_fontsize)
    ax.set_yticks(range(n))
    ax.set_yticklabels(cats, fontsize=tick_fontsize)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    for i in range(n):
        for j in range(n):
            val = cm_norm[i, j]
            color = "white" if (val > 0.75 or val < 0.15) else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=annot_fontsize, color=color)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title(f"Confusion matrix ({title}, row-normalized)", fontsize=9)

    fig.tight_layout()
    fig.savefig(FIG_DIR / f"{out_name}.pdf", bbox_inches="tight")
    fig.savefig(FIG_DIR / f"{out_name}.png", bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out_name}.pdf / .png")


def main():
    for result_file, out_name, title in DATASETS:
        plot_confusion(result_file, out_name, title)


if __name__ == "__main__":
    main()
