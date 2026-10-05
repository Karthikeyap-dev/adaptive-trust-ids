"""
Analysis E: Relative workload reduction, framed as relative change rather
than raw percentage points. Reads already-saved ablation results and
generates paper-ready sentences automatically, so wording matches the
exact saved numbers.
"""
from pathlib import Path
import json

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = PROJECT_ROOT / "outputs"


def load_json(name):
    path = OUT_DIR / name
    return json.load(open(path)) if path.exists() else None


def relative_change(old, new):
    if old == 0:
        return None
    return (new - old) / old * 100


def report_transition(dataset, from_label, to_label, from_metrics, to_metrics):
    esc_old, esc_new = from_metrics["escalation_rate"], to_metrics["escalation_rate"]
    fail_old, fail_new = from_metrics["silent_failure_rate"], to_metrics["silent_failure_rate"]
    esc_rel = relative_change(esc_old, esc_new)

    print(f"\n  [{dataset}] {from_label} -> {to_label}:")
    if esc_rel is not None:
        direction = "reduces" if esc_rel < 0 else "increases"
        print(f"    Escalation rate: {esc_old:.4f} -> {esc_new:.4f} "
              f"({direction} escalation by {abs(esc_rel):.1f}% relative to {from_label})")
    else:
        print(f"    Escalation rate: {esc_old:.4f} -> {esc_new:.4f} (baseline was zero, relative change undefined)")

    fail_direction = "increasing" if fail_new > fail_old else "decreasing"
    print(f"    Silent failure rate: {fail_old:.4f} -> {fail_new:.4f} "
          f"({fail_direction} from {fail_old:.2%} to {fail_new:.2%})")

    sentence = None
    if esc_rel is not None:
        verb = "reduces" if esc_rel < 0 else "increases"
        sentence = (f'On {dataset}, {to_label} {verb} escalation by approximately {abs(esc_rel):.0f}% '
                    f'relative to {from_label} ({esc_old:.1%} -> {esc_new:.1%}), while silent failures '
                    f'move from {fail_old:.2%} to {fail_new:.2%}.')
        print(f'    Paper-ready sentence: "{sentence}"')

    return {"dataset": dataset, "from": from_label, "to": to_label,
            "escalation_old": esc_old, "escalation_new": esc_new, "escalation_relative_change_pct": esc_rel,
            "silent_failure_old": fail_old, "silent_failure_new": fail_new, "sentence": sentence}


def main():
    results = []

    nsl_static = load_json("static_trust_baseline_results.json")
    nsl_qpso = load_json("qpso_results.json")
    if nsl_static and nsl_qpso:
        print("=" * 78)
        print("NSL-KDD")
        print("=" * 78)
        results.append(report_transition("NSL-KDD", "static trust", "adaptive trust",
                                          nsl_static, nsl_qpso["baseline_metrics"]))
        results.append(report_transition("NSL-KDD", "adaptive (fixed)", "adaptive + QPSO",
                                          nsl_qpso["baseline_metrics"], nsl_qpso["qpso_metrics"]))

    for dataset, fname in [("UNSW-NB15", "unsw_ablation_results.json"), ("CICIDS2017", "cicids_ablation_results.json")]:
        ablation = load_json(fname)
        if ablation:
            print(f"\n{'='*78}")
            print(dataset)
            print("=" * 78)
            results.append(report_transition(dataset, "static trust", "adaptive trust",
                                              ablation["static_trust_metrics"], ablation["adaptive_trust_metrics"]))
            results.append(report_transition(dataset, "adaptive (fixed)", "adaptive + QPSO",
                                              ablation["adaptive_trust_metrics"], ablation["qpso_metrics"]))

    with open(OUT_DIR / "relative_change_analysis.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n\nSaved to {OUT_DIR / 'relative_change_analysis.json'}")


if __name__ == "__main__":
    main()
