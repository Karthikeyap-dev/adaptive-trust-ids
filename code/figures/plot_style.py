"""
Shared style module - import this in every figure script for consistent
fonts, colors, and sizing across the whole figure set.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def apply_style():
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Nimbus Roman", "DejaVu Serif"],
        "font.size": 10,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "axes.axisbelow": True,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })

COLOR_STATIC = "#4338ca"
COLOR_ADAPTIVE = "#dc2626"
COLOR_OPTIMIZED = "#059669"
COLOR_NSLKDD = "#4338ca"
COLOR_UNSW = "#dc2626"
COLOR_CICIDS = "#059669"
COLOR_NOVICE = "#9ca3af"
COLOR_EXPERT = "#2563eb"
COLOR_COMPLACENT = "#dc2626"
COLOR_TRUST = "#7c3aed"
