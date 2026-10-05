"""
Generates IEEE-standard publication figures (vector PDF + PNG, 300 DPI,
serif font, single/double-column sizing) and LaTeX-ready tables from every
result file the pipeline has produced.

Gracefully skips any figure/table whose underlying result file doesn't
exist yet (e.g. run multi_seed_evaluation.py first for Fig. 4).

Run LAST, after all other analysis scripts have been run at least once:
  train_baseline.py, simulate_feedback.py, calibrate_confidence.py,
  static_trust_baseline.py, quantum_optimizer.py, pareto_sweep.py,
  multi_seed_evaluation.py, explanation_faithfulness.py, measure_real_latency.py
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = PROJECT_ROOT / "outputs"
FIG_DIR = PROJECT_ROOT / "outputs" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

IEEE_SINGLE_COL = (3.45, 2.6)
IEEE_DOUBLE_COL = (7.16, 3.2)
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "Nimbus Roman", "DejaVu Serif"],
    "font.size": 9, "axes.titlesize": 9, "axes.labelsize": 9,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 7.5,
    "figure.dpi": 300, "savefig.dpi": 300, "axes.grid": True, "grid.alpha": 0.3,
    "axes.axisbelow": True,
})
COLORS = ["#4338ca", "#dc2626", "#059669", "#d97706", "#7c3aed"]


def load_json(name):
    path = OUT_DIR / name
    return json.load(open(path)) if path.exists() else None


def load_csv(name):
    path = OUT_DIR / name
    return pd.read_csv(path) if path.exists() else None


def savefig(fig, name):
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(FIG_DIR / f"{name}.png", bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {name}.pdf / .png")


def fig1_baseline_performance():
    baseline = load_json("baseline_results.json")
    if not baseline:
        print("  [skip] Fig 1 - baseline_results.json not found"); return
    report = baseline["per_category_report"]
    cats = [c for c in ["normal", "dos", "probe", "r2l", "u2r"] if c in report]
    metrics = ["precision", "recall", "f1-score"]

    fig, ax = plt.subplots(figsize=IEEE_SINGLE_COL)
    x = np.arange(len(cats))
    width = 0.25
    for i, m in enumerate(metrics):
        vals = [report[c][m] for c in cats]
        ax.bar(x + (i - 1) * width, vals, width, label=m.capitalize(), color=COLORS[i])
    ax.set_xticks(x); ax.set_xticklabels(cats)
    ax.set_ylabel("Score"); ax.set_ylim(0, 1.05)
    ax.set_title("Baseline classifier per-category performance")
    ax.legend(loc="upper right", ncol=1)
    savefig(fig, "fig1_baseline_performance")


def fig2_trust_with_uncertainty():
    snapshot = load_json("trust_snapshot.json")
    if not snapshot:
        print("  [skip] Fig 2 - trust_snapshot.json not found"); return
    df = pd.DataFrame(snapshot).sort_values("trust", ascending=False)
    fig, ax = plt.subplots(figsize=IEEE_SINGLE_COL)
    errs = np.sqrt(df["uncertainty"].values)
    ax.bar(df["category"], df["trust"], yerr=errs, capsize=3, color=COLORS[0], alpha=0.85)
    ax.set_ylabel("Trust score"); ax.set_ylim(0, 1.05)
    ax.set_title("Adaptive Trust Engine: per-category trust ($\\pm$ SD)")
    savefig(fig, "fig2_trust_uncertainty")


def fig3_trust_evolution():
    evolution = load_csv("trust_evolution.csv")
    if evolution is None:
        print("  [skip] Fig 3 - trust_evolution.csv not found"); return
    fig, ax = plt.subplots(figsize=IEEE_DOUBLE_COL)
    for i, cat in enumerate(sorted(evolution["category"].unique())):
        subset = evolution[evolution["category"] == cat]
        ax.plot(subset["step"], subset["trust"], label=cat, color=COLORS[i % len(COLORS)], linewidth=1.2)
    ax.set_xlabel("Feedback step"); ax.set_ylabel("Trust score")
    ax.set_title("Trust convergence over simulated feedback stream")
    ax.legend(loc="lower right", ncol=5, fontsize=7)
    savefig(fig, "fig3_trust_evolution")


def fig4_ablation_with_stats():
    multiseed = load_json("multi_seed_summary.json")
    if not multiseed:
        print("  [skip] Fig 4 - multi_seed_summary.json not found. Run multi_seed_evaluation.py first."); return

    summary = multiseed["summary"]
    tests = multiseed["statistical_tests"]
    arms = ["static", "adaptive", "qpso"]
    arm_labels = ["Static\ntrust", "Adaptive\ntrust", "Adaptive+\nQPSO"]

    fig, axes = plt.subplots(1, 2, figsize=IEEE_DOUBLE_COL)
    for ax, metric, title in zip(axes, ["silent_failure_rate", "escalation_rate"],
                                  ["Silent failure rate", "Escalation rate"]):
        means = [summary[f"{a}_{metric}"]["mean"] for a in arms]
        stds = [summary[f"{a}_{metric}"]["std"] for a in arms]
        ax.bar(arm_labels, means, yerr=stds, capsize=4, color=COLORS[:3], alpha=0.85)
        ax.set_title(title, pad=28)

        top_val = max(m + s for m, s in zip(means, stds))
        headroom = max(top_val * 0.35, 0.02)
        ax.set_ylim(0, top_val + headroom * 2.6)
        ax.set_ylabel("Rate")

        p1 = tests.get(f"{metric}__static_vs_adaptive", {}).get("p_value", 1.0)
        p2 = tests.get(f"{metric}__adaptive_vs_qpso", {}).get("p_value", 1.0)
        # Stagger bracket heights so the two annotations never overlap each other
        bracket_ys = [top_val + headroom * 0.9, top_val + headroom * 1.9]
        for (i, j), p, y in [((0, 1), p1, bracket_ys[0]), ((1, 2), p2, bracket_ys[1])]:
            star = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."
            ax.plot([i, j], [y, y], color="black", linewidth=0.8)
            ax.text((i + j) / 2, y + headroom * 0.15, star, ha="center", fontsize=7)
    fig.suptitle(f"Three-way ablation (mean $\\pm$ SD, n={multiseed['n_seeds']} seeds)", y=1.08)
    savefig(fig, "fig4_ablation_statistical")


def fig5_pareto_curve():
    sweep = load_json("pareto_sweep.json")
    if not sweep:
        print("  [skip] Fig 5 - pareto_sweep.json not found"); return
    df = pd.DataFrame(sweep)
    fig, ax = plt.subplots(figsize=IEEE_SINGLE_COL)
    ax.plot(df["ceiling"] * 100, df["escalation_rate"] * 100, marker="o", markersize=3,
            color=COLORS[4], linewidth=1.3)
    ax.set_xlabel("Acceptable silent-failure ceiling (%)")
    ax.set_ylabel("Resulting escalation rate (%)")
    ax.set_title("QPSO risk-vs-workload trade-off")
    savefig(fig, "fig5_pareto_curve")


def fig6_calibration():
    calib = load_json("calibration_results.json")
    if not calib:
        print("  [skip] Fig 6 - calibration_results.json not found"); return
    fig, ax = plt.subplots(figsize=IEEE_SINGLE_COL)
    for bins, label, color in [(calib["bins_before"], "Before calibration", COLORS[1]),
                                (calib["bins_after"], "After calibration", COLORS[0])]:
        confs = [b["avg_confidence"] for b in bins]
        accs = [b["avg_accuracy"] for b in bins]
        ax.plot(confs, accs, marker="o", markersize=3, label=label, color=color, linewidth=1.2)
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=0.8, label="Perfect calibration")
    ax.set_xlabel("Mean predicted confidence"); ax.set_ylabel("Empirical accuracy")
    ax.set_title("Reliability diagram")
    ax.legend(loc="upper left")
    savefig(fig, "fig6_calibration_reliability")


def fig7_faithfulness():
    faith = load_json("faithfulness_results.json")
    if not faith:
        print("  [skip] Fig 7 - faithfulness_results.json not found. Run explanation_faithfulness.py first."); return
    fig, ax = plt.subplots(figsize=IEEE_SINGLE_COL)
    labels = ["SHAP-guided\nremoval", "Random\nremoval"]
    means = [faith["mean_drop_shap"], faith["mean_drop_random"]]
    stds = [faith["std_drop_shap"], faith["std_drop_random"]]
    ax.bar(labels, means, yerr=stds, capsize=4, color=[COLORS[0], "#9ca3af"], alpha=0.85)
    ax.set_ylabel("Mean confidence drop")
    p = faith["wilcoxon_p_value"]
    star = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."
    y = max(means[0] + stds[0], means[1] + stds[1]) + 0.03
    ax.plot([0, 1], [y, y], color="black", linewidth=0.8)
    ax.text(0.5, y + 0.01, star, ha="center", fontsize=8)
    ax.set_title(f"Explanation faithfulness (deletion test, n={faith['n_samples']})")
    savefig(fig, "fig7_faithfulness")


def fig8_confusion_matrix():
    baseline = load_json("baseline_results.json")
    if not baseline:
        print("  [skip] Fig 8 - baseline_results.json not found"); return
    cm = np.array(baseline["confusion_matrix"])
    cats = baseline["category_order"]
    cm_norm = cm / cm.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=IEEE_SINGLE_COL)
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(cats))); ax.set_xticklabels(cats, rotation=45, ha="right")
    ax.set_yticks(range(len(cats))); ax.set_yticklabels(cats)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    for i in range(len(cats)):
        for j in range(len(cats)):
            ax.text(j, i, f"{cm_norm[i,j]:.2f}", ha="center", va="center",
                     fontsize=6.5, color="white" if cm_norm[i, j] > 0.5 else "black")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title("Confusion matrix (row-normalized)")
    savefig(fig, "fig8_confusion_matrix")


def generate_latex_tables():
    lines = []

    baseline = load_json("baseline_results.json")
    if baseline:
        lines.append("% Table: Baseline classifier per-category performance")
        lines.append("\\begin{table}[t]")
        lines.append("\\centering")
        lines.append("\\caption{Baseline classifier per-category performance on the official NSL-KDD test split.}")
        lines.append("\\label{tab:baseline}")
        lines.append("\\begin{tabular}{lcccc}")
        lines.append("\\hline")
        lines.append("Category & Precision & Recall & F1 & Support \\\\")
        lines.append("\\hline")
        for c in ["normal", "dos", "probe", "r2l", "u2r"]:
            r = baseline["per_category_report"].get(c, {})
            lines.append(f"{c} & {r.get('precision',0):.3f} & {r.get('recall',0):.3f} & "
                         f"{r.get('f1-score',0):.3f} & {int(r.get('support',0))} \\\\")
        lines.append("\\hline")
        lines.append(f"Overall accuracy & \\multicolumn{{4}}{{c}}{{{baseline['overall_accuracy']:.3f}}} \\\\")
        lines.append("\\hline")
        lines.append("\\end{tabular}")
        lines.append("\\end{table}\n")

    multiseed = load_json("multi_seed_summary.json")
    if multiseed:
        s = multiseed["summary"]
        t = multiseed["statistical_tests"]
        lines.append("% Table: Three-way ablation, multi-seed statistics")
        lines.append("\\begin{table}[t]")
        lines.append("\\centering")
        lines.append(f"\\caption{{Three-way ablation, mean $\\pm$ SD across {multiseed['n_seeds']} seeds. "
                     "Wilcoxon signed-rank $p$-values shown for adjacent-arm comparisons.}")
        lines.append("\\label{tab:ablation}")
        lines.append("\\begin{tabular}{lccc}")
        lines.append("\\hline")
        lines.append("System & Silent failure rate & Escalation rate & $p$ (vs.\\ previous) \\\\")
        lines.append("\\hline")
        arms = [("static", "Static trust", "--"),
                ("adaptive", "Adaptive trust", f"{t['silent_failure_rate__static_vs_adaptive']['p_value']:.4f}"),
                ("qpso", "Adaptive + QPSO", f"{t['silent_failure_rate__adaptive_vs_qpso']['p_value']:.4f}")]
        for key, label, p in arms:
            sf = s[f"{key}_silent_failure_rate"]
            esc = s[f"{key}_escalation_rate"]
            lines.append(f"{label} & {sf['mean']:.3f} $\\pm$ {sf['std']:.3f} & "
                         f"{esc['mean']:.3f} $\\pm$ {esc['std']:.3f} & {p} \\\\")
        lines.append("\\hline")
        lines.append("\\end{tabular}")
        lines.append("\\end{table}\n")

    faith = load_json("faithfulness_results.json")
    if faith:
        lines.append("% Table: Explanation faithfulness")
        lines.append("\\begin{table}[t]")
        lines.append("\\centering")
        lines.append(f"\\caption{{Explanation faithfulness (deletion test, $n={faith['n_samples']}$, top-$k={faith['top_k']}$).}}")
        lines.append("\\label{tab:faithfulness}")
        lines.append("\\begin{tabular}{lcc}")
        lines.append("\\hline")
        lines.append("Removal strategy & Mean confidence drop & SD \\\\")
        lines.append("\\hline")
        lines.append(f"SHAP-guided & {faith['mean_drop_shap']:.3f} & {faith['std_drop_shap']:.3f} \\\\")
        lines.append(f"Random (control) & {faith['mean_drop_random']:.3f} & {faith['std_drop_random']:.3f} \\\\")
        lines.append("\\hline")
        lines.append(f"\\multicolumn{{3}}{{l}}{{Wilcoxon $p={faith['wilcoxon_p_value']:.2e}$}} \\\\")
        lines.append("\\hline")
        lines.append("\\end{tabular}")
        lines.append("\\end{table}\n")

    latency = load_json("measured_latency.json")
    if latency and latency.get("classifier_inference"):
        ci = latency["classifier_inference"]
        lines.append("% Table: Measured system latency")
        lines.append("\\begin{table}[t]")
        lines.append("\\centering")
        lines.append("\\caption{Measured computational latency (real wall-clock, not an assumed cost model). "
                     "Human review time was not measured computationally; see pilot study results.}")
        lines.append("\\label{tab:latency}")
        lines.append("\\begin{tabular}{lc}")
        lines.append("\\hline")
        lines.append("Measurement & Value \\\\")
        lines.append("\\hline")
        lines.append(f"ML agent inference (single alert) & {ci['single_row_inference_ms_mean']:.2f} ms $\\pm$ {ci['single_row_inference_ms_std']:.2f} \\\\")
        lines.append(f"ML agent inference (batch-amortized) & {ci['per_row_amortized_batch_ms']:.4f} ms \\\\")
        if latency.get("qpso_convergence"):
            qc = latency["qpso_convergence"]
            lines.append(f"QPSO policy convergence (offline, periodic) & {qc['qpso_convergence_seconds_mean']:.2f} s $\\pm$ {qc['qpso_convergence_seconds_std']:.2f} \\\\")
        lines.append("\\hline")
        lines.append("\\end{tabular}")
        lines.append("\\end{table}\n")

    out_path = OUT_DIR / "ieee_tables.tex"
    with open(out_path, "w") as f:
        f.write("\n".join(lines))
    print(f"  saved {out_path.name}")


def main():
    print("Generating IEEE-standard figures...")
    fig1_baseline_performance()
    fig2_trust_with_uncertainty()
    fig3_trust_evolution()
    fig4_ablation_with_stats()
    fig5_pareto_curve()
    fig6_calibration()
    fig7_faithfulness()
    fig8_confusion_matrix()
    print("\nGenerating LaTeX tables...")
    generate_latex_tables()
    print(f"\nAll figures saved to {FIG_DIR}/")
    print(f"LaTeX tables saved to {OUT_DIR / 'ieee_tables.tex'}")


if __name__ == "__main__":
    main()
