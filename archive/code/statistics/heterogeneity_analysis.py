"""
Analysis C: Reliability heterogeneity vs. adaptive/QPSO benefit.

Quantifies per-category reliability heterogeneity (std, range of trust
scores) for each dataset/classifier configuration and relates it to the
measured benefit of adaptive trust and QPSO over static trust. Reads
already-saved results - no retraining. With only ~4 configurations, this
is a descriptive relationship, not a formal statistical correlation test.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = PROJECT_ROOT / "outputs"


def load_json(name):
    path = OUT_DIR / name
    return json.load(open(path)) if path.exists() else None


def compute_heterogeneity(trust_snapshot):
    trusts = [row["trust"] for row in trust_snapshot]
    return {"std": float(np.std(trusts)), "range": float(max(trusts) - min(trusts)),
            "min": float(min(trusts)), "max": float(max(trusts)), "n_categories": len(trusts)}


def main():
    rows = []

    nsl_trust = load_json("trust_snapshot.json")
    nsl_static = load_json("static_trust_baseline_results.json")
    nsl_qpso = load_json("qpso_results.json")
    if nsl_trust and nsl_static and nsl_qpso:
        het = compute_heterogeneity(nsl_trust)
        static_esc = nsl_static["escalation_rate"]
        adaptive_esc = nsl_qpso["baseline_metrics"]["escalation_rate"]
        qpso_esc = nsl_qpso["qpso_metrics"]["escalation_rate"]
        rows.append({"config": "NSL-KDD (RandomForest)", **het,
                      "static_escalation": static_esc, "adaptive_escalation": adaptive_esc, "qpso_escalation": qpso_esc,
                      "adaptive_benefit_pp": (static_esc - adaptive_esc) * 100,
                      "qpso_additional_benefit_pp": (adaptive_esc - qpso_esc) * 100})

    for name, trust_file, ablation_file in [
        ("UNSW-NB15 (RandomForest)", "unsw_trust_snapshot.json", "unsw_ablation_results.json"),
        ("CICIDS2017 (RandomForest)", "cicids_trust_snapshot.json", "cicids_ablation_results.json"),
    ]:
        trust_snapshot = load_json(trust_file)
        ablation = load_json(ablation_file)
        if trust_snapshot and ablation:
            het = compute_heterogeneity(trust_snapshot)
            static_esc = ablation["static_trust_metrics"]["escalation_rate"]
            adaptive_esc = ablation["adaptive_trust_metrics"]["escalation_rate"]
            qpso_esc = ablation["qpso_metrics"]["escalation_rate"]
            rows.append({"config": name, **het,
                          "static_escalation": static_esc, "adaptive_escalation": adaptive_esc, "qpso_escalation": qpso_esc,
                          "adaptive_benefit_pp": (static_esc - adaptive_esc) * 100,
                          "qpso_additional_benefit_pp": (adaptive_esc - qpso_esc) * 100})

    if not rows:
        print("No complete dataset configurations found - run the baseline/ablation scripts first.")
        return

    df = pd.DataFrame(rows).sort_values("std")
    print("=" * 100)
    print("RELIABILITY HETEROGENEITY vs. ADAPTIVE/QPSO BENEFIT")
    print("=" * 100)
    print(df[["config", "std", "range", "adaptive_benefit_pp", "qpso_additional_benefit_pp"]].to_string(index=False))

    print(f"\n{'='*100}")
    if len(df) >= 3:
        corr_std_adaptive = df["std"].corr(df["adaptive_benefit_pp"])
        print(f"Descriptive correlation (heterogeneity std vs. adaptive-trust benefit): r = {corr_std_adaptive:.3f}")
        print(f"NOTE: with n={len(df)} configurations, this is a DESCRIPTIVE relationship only - "
              f"far too few points for a meaningful formal significance test.")
    else:
        print("Fewer than 3 configurations available - correlation not computed.")

    df.to_csv(OUT_DIR / "heterogeneity_vs_benefit.csv", index=False)
    print(f"\nSaved to {OUT_DIR / 'heterogeneity_vs_benefit.csv'}")


if __name__ == "__main__":
    main()
