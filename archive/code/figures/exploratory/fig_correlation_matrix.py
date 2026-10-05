"""
Correlation matrix of NSL-KDD's numeric flow-level features.

Uses a diverging red-blue colormap centered at 0 - this is the ONE case
where a diverging map is statistically, not just aesthetically, correct:
correlation genuinely ranges from -1 to +1 with 0 as a meaningful neutral
point (no relationship), unlike the confusion matrix's 0-1 scale.
"""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from data_loader import load_raw

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FIG_DIR = PROJECT_ROOT / "outputs" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "Nimbus Roman", "DejaVu Serif"],
    "font.size": 8,
})

NON_NUMERIC_COLS = ["protocol_type", "service", "flag", "label", "category", "is_attack"]


def main():
    print("Loading NSL-KDD training data...")
    train_df = load_raw("train")
    numeric_df = train_df.drop(columns=[c for c in NON_NUMERIC_COLS if c in train_df.columns])

    print(f"Computing correlation matrix over {numeric_df.shape[1]} numeric features...")
    corr = numeric_df.corr()

    fig, ax = plt.subplots(figsize=(7.16, 6.4))
    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=90, fontsize=5.5)
    ax.set_yticks(range(len(corr.columns)))
    ax.set_yticklabels(corr.columns, fontsize=5.5)
    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("Pearson correlation", fontsize=8)
    ax.set_title("Feature correlation matrix (NSL-KDD, numeric features)", fontsize=10)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_correlation_matrix.pdf", bbox_inches="tight")
    fig.savefig(FIG_DIR / "fig_correlation_matrix.png", bbox_inches="tight")
    plt.close(fig)
    print("Saved fig_correlation_matrix.pdf / .png")


if __name__ == "__main__":
    main()
