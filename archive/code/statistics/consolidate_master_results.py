"""
Master results consolidator - reads every result file this project has
produced and compiles them into ONE document (master_results.md) plus a
flat master_results.json, so you can paste a single file back instead of
dozens of separate outputs. Also reports which known result files are
MISSING - useful for the "reproducibility cleanup" audit item.
"""
from pathlib import Path
import json
import csv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = PROJECT_ROOT / "outputs"

EXPECTED_FILES = {
    "Baseline classifiers": [
        "baseline_results.json", "unsw_baseline_results.json",
        "cicids_baseline_results.json", "xgb_baseline_results.json",
    ],
    "Trust engine snapshots": [
        "trust_snapshot.json", "unsw_trust_snapshot.json", "cicids_trust_snapshot.json",
    ],
    "Three-way ablation": [
        "static_trust_baseline_results.json", "qpso_results.json",
        "unsw_ablation_results.json", "cicids_ablation_results.json", "xgb_ablation_results.json",
    ],
    "QPSO Pareto sweep": ["pareto_sweep.json"],
    "Calibration diagnostic": ["calibration_results.json"],
    "10-seed statistical validation": [
        "multi_seed_results.csv", "multi_seed_summary.json",
        "unsw_multi_seed_results.csv", "unsw_multi_seed_summary.json",
        "cicids_multi_seed_results.csv", "cicids_multi_seed_summary.json",
    ],
    "Explanation faithfulness": ["faithfulness_results.json"],
    "Real latency measurement": ["measured_latency.json"],
    "External baseline comparison": ["external_baseline_comparison.csv"],
    "Human pilot study": ["pilot_study_summary.json", "pilot_clustered_reanalysis.json"],
    "Extended arbitration baselines (6-arm)": ["extended_baselines_results.json"],
    "Sensitivity analysis": [
        "sensitivity_trust_noise_v2.json", "sensitivity_recovery.json",
        "sensitivity_decay_stationary.json", "sensitivity_decay_drift.json",
    ],
    "Uncertainty/CI reporting": ["uncertainty_intervals.json"],
    "Classical PSO vs QPSO (10-seed)": ["classical_vs_qpso_10seed_summary.json"],
    "Static trust threshold sweep": ["static_trust_sweep_full_results.json"],
    "Heterogeneity analysis": ["heterogeneity_vs_benefit.csv"],
    "Effect size analysis": ["effect_size_analysis.json"],
    "Relative change analysis": ["relative_change_analysis.json"],
}


def load_any(path):
    if path.suffix == ".json":
        return json.load(open(path))
    elif path.suffix == ".csv":
        with open(path) as f:
            return list(csv.DictReader(f))
    return None


def main():
    master = {}
    missing = []
    present = []
    md_lines = ["# Master Results Consolidation\n", "*Auto-generated - re-run the script rather than editing by hand.*\n"]

    for group_name, filenames in EXPECTED_FILES.items():
        md_lines.append(f"\n## {group_name}\n")
        master[group_name] = {}
        for fname in filenames:
            path = OUT_DIR / fname
            if path.exists():
                try:
                    data = load_any(path)
                    master[group_name][fname] = data
                    present.append(fname)
                    md_lines.append(f"\n### `{fname}`\n```json\n{json.dumps(data, indent=2, default=str)[:3000]}\n```\n")
                except Exception as e:
                    md_lines.append(f"\n### `{fname}` - ERROR READING: {e}\n")
                    missing.append(f"{fname} (present but unreadable: {e})")
            else:
                missing.append(fname)
                md_lines.append(f"\n### `{fname}` - **MISSING**\n")

    with open(OUT_DIR / "master_results.json", "w") as f:
        json.dump(master, f, indent=2, default=str)
    with open(OUT_DIR / "master_results.md", "w") as f:
        f.write("\n".join(md_lines))

    print(f"{'='*70}\nCONSOLIDATION SUMMARY\n{'='*70}")
    print(f"Present: {len(present)} files")
    print(f"Missing: {len(missing)} files")
    if missing:
        print("\nMissing/unreadable files:")
        for m in missing:
            print(f"  - {m}")
    print(f"\nSaved to {OUT_DIR / 'master_results.json'} and {OUT_DIR / 'master_results.md'}")
    print(f"Paste the contents of master_results.md back for the pre-writing audit.")


if __name__ == "__main__":
    main()
