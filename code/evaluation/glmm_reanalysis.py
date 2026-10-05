"""
Mixed-effects (GLMM) reanalysis of the human pilot data (Issue #28).

Models per-judgment correctness with random intercepts for BOTH
participant and alert (crossed random effects), rather than aggregating
to participant level only.

Requires: pip install statsmodels

Usage: python3 glmm_reanalysis.py <combined_pilot_csv>
Expects columns: participant_id, alert_index_in_study (or row_index),
phase1_correct, phase2_correct.
"""
from pathlib import Path
import sys
import json
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = PROJECT_ROOT / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def reshape_to_long(df):
    alert_col = "alert_index_in_study" if "alert_index_in_study" in df.columns else "row_index"
    long_rows = []
    for _, row in df.iterrows():
        long_rows.append({"participant_id": row["participant_id"], "alert_id": row[alert_col],
                           "phase": 0, "correct": int(row["phase1_correct"])})
        long_rows.append({"participant_id": row["participant_id"], "alert_id": row[alert_col],
                           "phase": 1, "correct": int(row["phase2_correct"])})
    return pd.DataFrame(long_rows)


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 glmm_reanalysis.py <combined_pilot_csv>")
        sys.exit(1)
    data_path = Path(sys.argv[1])
    if not data_path.exists():
        raise FileNotFoundError(f"Missing {data_path}")

    df = pd.read_csv(data_path)
    for col in ["phase1_correct", "phase2_correct"]:
        if df[col].dtype == object:
            df[col] = df[col].map({"True": 1, "False": 0, True: 1, False: 0}).astype(int)
        else:
            df[col] = df[col].astype(int)

    long_df = reshape_to_long(df)
    print(f"Reshaped {len(df)} wide rows into {len(long_df)} long-format observations "
          f"({long_df['participant_id'].nunique()} participants, {long_df['alert_id'].nunique()} alerts)\n")

    result1 = None
    print("=" * 70)
    print("MODEL 1: Random intercept for participant only (binomial GLMM)")
    print("=" * 70)
    try:
        import statsmodels.genmod.bayes_mixed_glm as bmg
        model1_glmm = bmg.BinomialBayesMixedGLM.from_formula(
            "correct ~ phase", {"participant": "0 + C(participant_id)"}, long_df)
        result1 = model1_glmm.fit_vb()
        print(result1.summary())
    except Exception as e:
        print(f"[BinomialBayesMixedGLM failed: {e} - falling back to linear MixedLM approximation]")
        model1 = smf.mixedlm("correct ~ phase", long_df, groups=long_df["participant_id"])
        result1 = model1.fit()
        print(result1.summary())

    result2 = None
    print(f"\n{'=' * 70}")
    print("MODEL 2: Crossed random effects - participant AND alert")
    print("=" * 70)
    try:
        import statsmodels.genmod.bayes_mixed_glm as bmg
        model2_glmm = bmg.BinomialBayesMixedGLM.from_formula(
            "correct ~ phase",
            {"participant": "0 + C(participant_id)", "alert": "0 + C(alert_id)"},
            long_df)
        result2 = model2_glmm.fit_vb()
        print(result2.summary())
        phase_coef = result2.fe_mean[1]
        phase_sd = result2.fe_sd[1]
        print(f"\nPhase effect (log-odds, assisted vs. blind): {phase_coef:.4f} +/- {phase_sd:.4f}")
        print(f"Odds ratio: {np.exp(phase_coef):.4f}")
    except Exception as e:
        print(f"[Crossed-random-effects model failed: {e}]")
        print("Model 1 above is still a valid improvement over pure participant-level aggregation.")

    summary = {
        "n_observations": len(long_df),
        "n_participants": int(long_df["participant_id"].nunique()),
        "n_alerts": int(long_df["alert_id"].nunique()),
        "model1_summary": str(result1.summary()) if result1 is not None else None,
        "model2_summary": str(result2.summary()) if result2 is not None else None,
    }
    with open(OUT_DIR / "glmm_reanalysis_results.json", "w") as f:
        json.dump(summary, f, indent=2)
    long_df.to_csv(OUT_DIR / "pilot_data_long_format.csv", index=False)
    print(f"\nSaved long-format data and model summaries to {OUT_DIR}")


if __name__ == "__main__":
    main()
