import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.lines import Line2D
import numpy as np

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif']
plt.rcParams['mathtext.fontset'] = 'stix'

comparisons = [
    'UNSW-NB15/RF', 'CICIDS2017/RF', 'CICIDS2017/XGB',
    'NSL-KDD/XGB', 'UNSW-NB15/XGB', 'NSL-KDD/RF'
]
p_values = [0.002, 0.002, 0.002, 0.0098, 0.037, 0.25]
n = len(p_values)
alpha = 0.05

ranks = np.arange(1, n + 1)
thresholds = alpha / (n - ranks + 1)
survives = [p < t for p, t in zip(p_values, thresholds)]

fig, ax = plt.subplots(figsize=(7.5, 5), dpi=600)
x = np.arange(n)

colors = ['#1a7a3c' if s else '#b3261e' for s in survives]
markers = ['o' if s else 'X' for s in survives]

for i in range(n):
    ax.scatter(x[i], p_values[i], s=160, color=colors[i], marker=markers[i],
               zorder=3, edgecolor='black', linewidth=0.9)
    va = 'bottom' if p_values[i] < 0.15 else 'top'
    offset = 12 if va == 'bottom' else -14
    ax.annotate(f'{p_values[i]:.4f}', (x[i], p_values[i]),
                textcoords="offset points", xytext=(0, offset),
                ha='center', va=va, fontsize=10)

ax.plot(x, thresholds, linestyle='--', color='#6e6e6e', marker='s',
        markersize=6, linewidth=1.3, zorder=2)
ax.axhline(alpha, color='black', linewidth=0.8, linestyle=':', zorder=1)

ax.set_yscale('log')
ax.set_ylim(0.0012, 0.6)          # headroom so the top marker/label never clips
ax.set_xlim(-0.5, n - 0.5)
ax.set_xticks(x)
ax.set_xticklabels(comparisons, rotation=20, ha='right', fontsize=11)
ax.set_ylabel('p-value (log scale)', fontsize=12)

ax.grid(axis='y', which='major', linestyle='-', linewidth=0.4, alpha=0.35, zorder=0)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.tick_params(axis='both', labelsize=10)

legend_elements = [
    Line2D([0], [0], marker='o', color='w', markerfacecolor='#1a7a3c',
           markeredgecolor='black', markersize=11, label='Survives correction'),
    Line2D([0], [0], marker='X', color='w', markerfacecolor='#b3261e',
           markeredgecolor='black', markersize=11, label='Does not survive'),
    Line2D([0], [0], linestyle='--', color='#6e6e6e', marker='s', markersize=6,
           label='Holm-Bonferroni threshold'),
    Line2D([0], [0], linestyle=':', color='black', label='Uncorrected α = 0.05'),
]
ax.legend(handles=legend_elements, loc='upper left', fontsize=9.5,
          frameon=True, framealpha=0.95, edgecolor='#cccccc')

plt.tight_layout()
plt.savefig('fig_holm_bonferroni.png', dpi=600, bbox_inches='tight')
plt.savefig('fig_holm_bonferroni.pdf', bbox_inches='tight')