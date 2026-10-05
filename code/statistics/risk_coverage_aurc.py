"""
Risk-coverage curve and Area Under the Risk-Coverage Curve (AURC),
addressing the review point that the paper's arbitration comparison
relies on a single delta=5% operating point, when the selective-
prediction literature it positions against (Chow 1970; El-Yaniv &
Wiener 2010) standardly reports a full risk-coverage curve.

This does NOT require new experiments - it reuses the same real
predictions (pred_category, raw_confidence, correct) already used
throughout this project, computing risk and coverage at every possible
threshold rather than the single delta=5% ceiling used elsewhere in
the paper.

For a ranking score (confidence, or trust, or 1-P(wrong) for the
rejector), sorting alerts by score and sweeping the threshold gives:
    coverage(t)  = fraction of alerts with score >= t (auto-executed)
    risk(t)      = fraction of THOSE alerts that are incorrect
AURC is the area under risk(coverage) - lower is better (less risk at
a given coverage level, integrated across all coverage levels, not
just one operating point).

Compares three rankings on the same real data: confidence-only,
adaptive trust (per-category posterior mean at final state), and the
learned rejector's P(correct) if that model/predictions are available.

Usage: python3 risk_coverage_aurc.py <predictions_csv> [output_prefix]
Optionally: python3 risk_coverage_aurc.py <predictions_csv> <output_prefix> <trust_predictions_csv>
  where trust_predictions_csv has an additional 'trust_score' column
  (final per-category trust value repeated per row) if you want the
  adaptive-trust ranking included; otherwise only confidence-only is
  computed (still useful, and the AURC/curve computation is identical).
"""
from pathlib import Path
import json
import sys
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = PROJECT_ROOT / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

N_THRESHOLDS = 200


def risk_coverage_curve(scores, correct, n_thresholds=N_THRESHOLDS):
    """scores: higher = more confident/trusted (kept). correct: 1/0 array.
    Returns (coverage_array, risk_array), both length n_thresholds,
    sorted by increasing coverage."""
    scores = np.asarray(scores)
    correct = np.asarray(correct)
    n = len(scores)
    thresholds = np.quantile(scores, np.linspace(0, 1, n_thresholds))
    coverages, risks = [], []
    for t in thresholds:
        kept = scores >= t
        cov = kept.mean()
        if kept.sum() == 0:
            risk = 0.0
        else:
            risk = 1.0 - correct[kept].mean()  # risk = error rate among kept (auto-executed) alerts
        coverages.append(cov)
        risks.append(risk)
    # Sort by coverage ascending for a clean curve and AURC integration
    order = np.argsort(coverages)
    coverages = np.array(coverages)[order]
    risks = np.array(risks)[order]
    return coverages, risks


def compute_aurc(coverages, risks):
    """Trapezoidal integration of risk over coverage in [0,1]."""
    # Ensure endpoints exist for a well-defined integral
    if coverages[0] > 0:
        coverages = np.concatenate([[0.0], coverages])
        risks = np.concatenate([[risks[0]], risks])
    if coverages[-1] < 1:
        coverages = np.concatenate([coverages, [1.0]])
        risks = np.concatenate([risks, [risks[-1]]])
    trapz_fn = getattr(np, "trapezoid", None) or np.trapz
    return float(trapz_fn(risks, coverages))


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 risk_coverage_aurc.py <predictions_csv> [output_prefix] [trust_predictions_csv]")
        sys.exit(1)
    preds_path = Path(sys.argv[1])
    prefix = sys.argv[2] if len(sys.argv) > 2 else preds_path.stem
    trust_path = Path(sys.argv[3]) if len(sys.argv) > 3 else None

    if not preds_path.exists():
        raise FileNotFoundError(f"Missing {preds_path}")
    preds = pd.read_csv(preds_path)
    print(f"Loaded {len(preds)} predictions from {preds_path}\n")

    correct = preds["correct"].values
    confidence = preds["raw_confidence"].values

    print("=" * 70)
    print("Confidence-only risk-coverage curve")
    print("=" * 70)
    cov_conf, risk_conf = risk_coverage_curve(confidence, correct)
    aurc_conf = compute_aurc(cov_conf, risk_conf)
    print(f"AURC (confidence-only) = {aurc_conf:.6f}")
    # Print a compact table of ~10 representative points
    idxs = np.linspace(0, len(cov_conf) - 1, 11).astype(int)
    print(f"{'Coverage':>10}{'Risk':>10}")
    for i in idxs:
        print(f"{cov_conf[i]:>10.3f}{risk_conf[i]:>10.4f}")

    results = {
        "confidence_only": {
            "aurc": aurc_conf,
            "coverage": cov_conf.tolist(),
            "risk": risk_conf.tolist(),
        }
    }

    if trust_path is not None and trust_path.exists():
        trust_preds = pd.read_csv(trust_path)
        if "trust_score" in trust_preds.columns:
            print(f"\n{'=' * 70}")
            print("Adaptive-trust risk-coverage curve")
            print("=" * 70)
            cov_trust, risk_trust = risk_coverage_curve(
                trust_preds["trust_score"].values, trust_preds["correct"].values)
            aurc_trust = compute_aurc(cov_trust, risk_trust)
            print(f"AURC (adaptive trust) = {aurc_trust:.6f}")
            print(f"AURC comparison: confidence-only={aurc_conf:.6f}  adaptive-trust={aurc_trust:.6f}  "
                  f"({'trust lower risk overall' if aurc_trust < aurc_conf else 'confidence lower risk overall'})")
            results["adaptive_trust"] = {
                "aurc": aurc_trust, "coverage": cov_trust.tolist(), "risk": risk_trust.tolist(),
            }
        else:
            print(f"\n[trust_predictions_csv provided but missing 'trust_score' column - skipping trust curve]")
    else:
        print(f"\n[No trust_predictions_csv provided - only confidence-only AURC computed. "
              f"To add the adaptive-trust curve, pass a CSV with a 'trust_score' column "
              f"(per-row trust value for that alert's predicted category) as a third argument.]")

    with open(OUT_DIR / f"{prefix}_risk_coverage_aurc.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {prefix}_risk_coverage_aurc.json in {OUT_DIR}")


if __name__ == "__main__":
    main()
