"""
Consolidated cross-dataset figures and tables - the final synthesis
combining NSL-KDD, UNSW-NB15, and CICIDS2017 results into the comparison
figures/tables for the paper's main results section.

Gracefully skips any panel whose underlying result file is missing.
Run this LAST, after all three datasets' baseline + ablation scripts
have been run.

Expects, per dataset, these output files (already produced):
  NSL-KDD:     baseline_results.json, qpso_results.json, static_trust_baseline_results.json
  UNSW-NB15:   unsw_baseline_results.json, unsw_ablation_results.json
  CICIDS2017:  cicids_baseline_results.json, cicids_ablation_results.json
"""
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
    "font.size": 9, "axes.titlesize": 9, "axes.labelsize": 9,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 7.5,
    "figure.dpi": 300, "savefig.dpi": 300, "axes.grid": True, "grid.alpha": 0.3,
    "axes.axisbelow": True,
})
COLORS = {"NSL-KDD": "#4338ca", "UNSW-NB15": "#dc2626", "CICIDS2017": "#059669"}


def load_json(name):
    path = OUT_DIR / name
    return json.load(open(path)) if path.exists() else None


def savefig(fig, name):
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(FIG_DIR / f"{name}.png", bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {name}.pdf / .png")


def gather_dataset_results():
    datasets = {}

    nsl_baseline = load_json("baseline_results.json")
    nsl_qpso = load_json("qpso_results.json")
    nsl_static = load_json("static_trust_baseline_results.json")
    if nsl_baseline and nsl_qpso and nsl_static:
        datasets["NSL-KDD"] = {
            "accuracy": nsl_baseline["overall_accuracy"],
            "static": {"silent_failure_rate": nsl_static["silent_failure_rate"], "escalation_rate": nsl_static["escalation_rate"]},
            "adaptive": {"silent_failure_rate": nsl_qpso["baseline_metrics"]["silent_failure_rate"], "escalation_rate": nsl_qpso["baseline_metrics"]["escalation_rate"]},
            "qpso": {"silent_failure_rate": nsl_qpso["qpso_metrics"]["silent_failure_rate"], "escalation_rate": nsl_qpso["qpso_metrics"]["escalation_rate"]},
        }

    unsw_baseline = load_json("unsw_baseline_results.json")
    unsw_ablation = load_json("unsw_ablation_results.json")
    if unsw_baseline and unsw_ablation:
        datasets["UNSW-NB15"] = {
            "accuracy": unsw_baseline["overall_accuracy"],
            "static": unsw_ablation["static_trust_metrics"],
            "adaptive": unsw_ablation["adaptive_trust_metrics"],
            "qpso": unsw_ablation["qpso_metrics"],
        }

    cicids_baseline = load_json("cicids_baseline_results.json")
    cicids_ablation = load_json("cicids_ablation_results.json")
    if cicids_baseline and cicids_ablation:
        datasets["CICIDS2017"] = {
            "accuracy": cicids_baseline["overall_accuracy"],
            "static": cicids_ablation["static_trust_metrics"],
            "adaptive": cicids_ablation["adaptive_trust_metrics"],
            "qpso": cicids_ablation["qpso_metrics"],
        }

    return datasets


def fig_cross_dataset_ablation(datasets):
    if len(datasets) < 2:
        print("  [skip] cross-dataset ablation figure - need at least 2 datasets"); return
    names = list(datasets.keys())
    arms = ["static", "adaptive", "qpso"]
    arm_labels = ["Static trust", "Adaptive trust", "Adaptive + QPSO"]

    fig, ax = plt.subplots(figsize=(7.16, 3.4))
    x = np.arange(len(arms))
    width = 0.8 / len(names)
    for i, name in enumerate(names):
        vals = [datasets[name][arm]["escalation_rate"] * 100 for arm in arms]
        ax.bar(x + i * width - 0.4 + width / 2, vals, width, label=name, color=COLORS.get(name, f"C{i}"))
    ax.set_xticks(x)
    ax.set_xticklabels(arm_labels)
    ax.set_ylabel("Escalation rate (%)")
    ax.set_title("Three-way ablation: escalation rate across datasets")
    ax.legend(loc="upper right")
    savefig(fig, "fig_cross_dataset_ablation")


def fig_cross_dataset_accuracy(datasets):
    if len(datasets) < 2:
        print("  [skip] cross-dataset accuracy figure - need at least 2 datasets"); return
    fig, ax = plt.subplots(figsize=(3.45, 2.6))
    names = list(datasets.keys())
    accs = [datasets[n]["accuracy"] * 100 for n in names]
    colors = [COLORS.get(n, "gray") for n in names]
    ax.bar(names, accs, color=colors, alpha=0.85)
    ax.set_ylabel("Baseline accuracy (%)")
    ax.set_title("Baseline classifier accuracy by dataset")
    ax.set_ylim(0, 105)
    for i, v in enumerate(accs):
        ax.text(i, v + 1.5, f"{v:.1f}%", ha="center", fontsize=8)
    savefig(fig, "fig_cross_dataset_accuracy")


def fig_qpso_relative_improvement(datasets):
    if len(datasets) < 2:
        print("  [skip] QPSO relative improvement figure - need at least 2 datasets"); return
    fig, ax = plt.subplots(figsize=(3.45, 2.6))
    names = list(datasets.keys())
    fixed = [datasets[n]["adaptive"]["escalation_rate"] * 100 for n in names]
    tuned = [datasets[n]["qpso"]["escalation_rate"] * 100 for n in names]
    x = np.arange(len(names))
    width = 0.35
    ax.bar(x - width / 2, fixed, width, label="Fixed thresholds", color="#9ca3af")
    ax.bar(x + width / 2, tuned, width, label="QPSO-tuned", color="#7c3aed")
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylabel("Escalation rate (%)")
    ax.set_title("QPSO improvement magnitude by dataset")
    ax.legend()
    savefig(fig, "fig_qpso_relative_improvement")


def generate_cross_dataset_table(datasets):
    if not datasets:
        print("  [skip] cross-dataset table - no datasets found"); return
    lines = [
        "% Table: Cross-dataset three-way ablation summary",
        "\\begin{table*}[t]", "\\centering",
        "\\caption{Three-way ablation results across all validated benchmark datasets.}",
        "\\label{tab:cross_dataset}",
        "\\begin{tabular}{l" + "cc" * len(datasets) + "}",
        "\\hline",
        "System & " + " & ".join(f"\\multicolumn{{2}}{{c}}{{{name}}}" for name in datasets) + " \\\\",
        "& " + " & ".join(["Fail rate & Escalation"] * len(datasets)) + " \\\\",
        "\\hline",
    ]
    arm_labels = {"static": "Static trust", "adaptive": "Adaptive trust", "qpso": "Adaptive + QPSO"}
    for arm, label in arm_labels.items():
        row = [label]
        for name in datasets:
            m = datasets[name][arm]
            row.append(f"{m['silent_failure_rate']:.3f}")
            row.append(f"{m['escalation_rate']:.3f}")
        lines.append(" & ".join(row) + " \\\\")
    lines += ["\\hline", "\\end{tabular}", "\\end{table*}", ""]

    accuracy_lines = [
        "% Table: Baseline accuracy per dataset",
        "\\begin{table}[t]", "\\centering",
        "\\caption{Baseline classifier accuracy per benchmark dataset.}",
        "\\label{tab:baseline_accuracy}",
        "\\begin{tabular}{lc}", "\\hline", "Dataset & Accuracy \\\\", "\\hline",
    ]
    for name in datasets:
        accuracy_lines.append(f"{name} & {datasets[name]['accuracy']:.4f} \\\\")
    accuracy_lines += ["\\hline", "\\end{tabular}", "\\end{table}", ""]

    out_path = OUT_DIR / "cross_dataset_tables.tex"
    with open(out_path, "w") as f:
        f.write("\n".join(lines + accuracy_lines))
    print(f"  saved {out_path.name}")


def main():
    print("Gathering results from all available datasets...")
    datasets = gather_dataset_results()
    print(f"Found complete results for: {list(datasets.keys())}\n")

    if not datasets:
        print("No dataset results found. Run the baseline + ablation scripts for at least one dataset first.")
        return

    print("Generating cross-dataset figures...")
    fig_cross_dataset_ablation(datasets)
    fig_cross_dataset_accuracy(datasets)
    fig_qpso_relative_improvement(datasets)

    print("\nGenerating cross-dataset LaTeX tables...")
    generate_cross_dataset_table(datasets)

    summary_path = OUT_DIR / "cross_dataset_summary.json"
    with open(summary_path, "w") as f:
        json.dump(datasets, f, indent=2)
    print(f"\nSaved machine-readable summary to {summary_path}")
    print(f"All figures in {FIG_DIR}/, tables in {OUT_DIR / 'cross_dataset_tables.tex'}")


if __name__ == "__main__":
    main()
