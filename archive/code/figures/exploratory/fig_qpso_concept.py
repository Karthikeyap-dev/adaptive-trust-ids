"""
Conceptual illustration: QPSO's quantum-inspired search behavior.

Shows a 2D toy fitness landscape with a swarm of particles at an early
iteration and a later iteration, illustrating convergence toward the
optimum. A third panel contrasts classical PSO's velocity-vector movement
with QPSO's probability-cloud-based position sampling (the actual source
of the "quantum-inspired" name), to make the distinction concrete rather
than just asserted in text.
"""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FIG_DIR = PROJECT_ROOT / "outputs" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "Nimbus Roman", "DejaVu Serif"],
    "font.size": 9,
})


def toy_fitness(x, y):
    return -((x - 0.7) ** 2 + (y - 0.6) ** 2) + 0.15 * np.sin(8 * x) * np.cos(8 * y)


def main():
    rng = np.random.default_rng(7)
    xg, yg = np.meshgrid(np.linspace(0, 1, 200), np.linspace(0, 1, 200))
    zg = toy_fitness(xg, yg)
    optimum = (0.7, 0.6)

    fig, axes = plt.subplots(1, 3, figsize=(7.16, 2.6))

    ax = axes[0]
    ax.contourf(xg, yg, zg, levels=20, cmap="Purples")
    early = rng.uniform(0.05, 0.95, size=(18, 2))
    ax.scatter(early[:, 0], early[:, 1], c="white", edgecolors="black", s=22, zorder=3)
    ax.scatter(*optimum, marker="*", c="#facc15", edgecolors="black", s=140, zorder=4)
    ax.set_title("Early iteration:\nswarm scattered", fontsize=8.5)
    ax.set_xticks([]); ax.set_yticks([])

    ax = axes[1]
    ax.contourf(xg, yg, zg, levels=20, cmap="Purples")
    late = np.array(optimum) + rng.normal(0, 0.05, size=(18, 2))
    late = np.clip(late, 0, 1)
    ax.scatter(late[:, 0], late[:, 1], c="white", edgecolors="black", s=22, zorder=3)
    ax.scatter(*optimum, marker="*", c="#facc15", edgecolors="black", s=140, zorder=4)
    ax.set_title("Later iteration:\nswarm converged near optimum", fontsize=8.5)
    ax.set_xticks([]); ax.set_yticks([])

    ax = axes[2]
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    current = np.array([0.35, 0.4])
    classical_new = np.array([0.55, 0.55])
    ax.scatter(*current, c="#374151", s=50, zorder=3)
    ax.annotate("", xy=classical_new, xytext=current,
                arrowprops=dict(arrowstyle="->", color="#2563eb", lw=1.6))
    ax.text(classical_new[0] + 0.02, classical_new[1], "Classical PSO:\nfixed velocity step",
            fontsize=6.5, color="#2563eb")

    cloud_pts = current + rng.normal(0, 0.13, size=(60, 2))
    cloud_pts = np.clip(cloud_pts, 0, 1)
    ax.scatter(cloud_pts[:, 0], cloud_pts[:, 1], c="#7c3aed", s=6, alpha=0.35, zorder=2)
    ax.text(0.05, 0.85, "QPSO: sampled from\nprobability cloud", fontsize=6.5, color="#7c3aed")
    ax.set_title("Position update mechanism", fontsize=8.5)
    ax.set_xticks([]); ax.set_yticks([])

    fig.suptitle("Quantum-inspired Particle Swarm Optimization (QPSO): conceptual illustration", fontsize=9.5, y=1.06)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_qpso_concept.pdf", bbox_inches="tight")
    fig.savefig(FIG_DIR / "fig_qpso_concept.png", bbox_inches="tight")
    plt.close(fig)
    print("Saved fig_qpso_concept.pdf / .png")


if __name__ == "__main__":
    main()
