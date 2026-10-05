"""
Cluster-aware reanalysis of the human pilot study.

Addresses a real statistical issue: 125 responses are NOT 125 independent
observations - they are 25 alerts x 5 participants, so responses within
the same participant are correlated. Pooling them as if independent
overstates the effective sample size.

The fix at n=5 participants: aggregate to the PARTICIPANT level first
(5 means), then run paired tests on those 5 independent units - standard
"by-participants" analysis in HCI/psychology research with small samples.

Run alongside analyze_pilot_results.py (does not replace it).
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = PROJECT_ROOT / "outputs"
PILOT_RESULTS_PATH = OUT_DIR / "pilot_study_results.csv"


def cohens_d_paired(x, y):
    diff = np.array(x) - np.array(y)
    return float(diff.mean() / diff.std(ddof=1)) if diff.std(ddof=1) > 0 else float("nan")


def main():
    if not PILOT_RESULTS_PATH.exists():
        print(f"No pilot results found at {PILOT_RESULTS_PATH}.")
        return

    df = pd.read_csv(PILOT_RESULTS_PATH)
    n_participants = df["participant_id"].nunique()
    print(f"Loaded {len(df)} responses from {n_participants} participants: {df['participant_id'].unique().tolist()}\n")

    print("=" * 78)
    print("PARTICIPANT-LEVEL SUMMARY (the correct unit of analysis)")
    print("=" * 78)
    participant_stats = df.groupby("participant_id").agg(
        n_responses=("phase1_correct", "count"),
        phase1_accuracy=("phase1_correct", "mean"),
        phase2_accuracy=("phase2_correct", "mean"),
        n_changed_mind=("changed_mind_after_explanation", "sum"),
    ).reset_index()
    participant_stats["improvement"] = participant_stats["phase2_accuracy"] - participant_stats["phase1_accuracy"]

    changed_correct_list, changed_incorrect_list = [], []
    for pid in participant_stats["participant_id"]:
        sub = df[(df["participant_id"] == pid) & (df["changed_mind_after_explanation"])]
        changed_correct_list.append(int((sub["phase2_correct"] & ~sub["phase1_correct"]).sum()))
        changed_incorrect_list.append(int((~sub["phase2_correct"] & sub["phase1_correct"]).sum()))
    participant_stats["mind_changes_became_correct"] = changed_correct_list
    participant_stats["mind_changes_became_incorrect"] = changed_incorrect_list

    print(participant_stats.to_string(index=False))

    print(f"\n{'='*78}")
    print("PAIRED TEST AT PARTICIPANT LEVEL (n=5 independent units - the correct n)")
    print("=" * 78)
    p1 = participant_stats["phase1_accuracy"].values
    p2 = participant_stats["phase2_accuracy"].values

    t_stat, t_p = stats.ttest_rel(p2, p1)
    w_stat, w_p = None, None
    try:
        w_stat, w_p = stats.wilcoxon(p2, p1)
    except ValueError as e:
        print(f"  (Wilcoxon not computable: {e} - expected with very small/tied samples)")

    d = cohens_d_paired(p2, p1)

    print(f"  Mean phase1 (blind) accuracy across participants: {p1.mean():.3f} (range {p1.min():.3f}-{p1.max():.3f})")
    print(f"  Mean phase2 (assisted) accuracy across participants: {p2.mean():.3f} (range {p2.min():.3f}-{p2.max():.3f})")
    print(f"  Mean improvement: {(p2-p1).mean():.3f}")
    print(f"  Paired t-test: t={t_stat:.3f}, p={t_p:.4f}")
    if w_p is not None:
        print(f"  Wilcoxon signed-rank: W={w_stat:.3f}, p={w_p:.4f}")
    print(f"  Cohen's d (paired): {d:.3f}")
    print(f"\n  NOTE: with n=5 participants, this test has very limited statistical power.")
    print(f"  A non-significant p-value here does NOT mean there is no effect - report the")
    print(f"  effect size and direction as PRELIMINARY evidence, not a confirmed effect.")

    print(f"\n{'='*78}")
    print("ROBUSTNESS CHECK: is the mind-change pattern broadly shared, or one-participant-driven?")
    print("=" * 78)
    print(participant_stats[["participant_id", "n_changed_mind", "mind_changes_became_correct",
                              "mind_changes_became_incorrect"]].to_string(index=False))
    n_positive = (participant_stats["mind_changes_became_correct"] > 0).sum()
    print(f"\n  {n_positive} of {n_participants} participants had at least one mind-change that became correct.")

    result = {
        "n_participants": int(n_participants),
        "participant_level_stats": participant_stats.to_dict(orient="records"),
        "paired_test_participant_level": {
            "phase1_mean": float(p1.mean()), "phase2_mean": float(p2.mean()),
            "mean_improvement": float((p2 - p1).mean()),
            "paired_t_statistic": float(t_stat), "paired_t_p_value": float(t_p),
            "wilcoxon_statistic": float(w_stat) if w_stat is not None else None,
            "wilcoxon_p_value": float(w_p) if w_p is not None else None,
            "cohens_d": d,
        },
        "n_participants_with_positive_mind_change": int(n_positive),
    }
    with open(OUT_DIR / "pilot_clustered_reanalysis.json", "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved to {OUT_DIR / 'pilot_clustered_reanalysis.json'}")

    print(f"\n{'='*78}\nSUGGESTED CLAIM LANGUAGE\n{'='*78}")
    print(f"""  Instead of: "Our system statistically improves human accuracy."
  Use: "This pilot provides preliminary evidence (n=5 participants) that AI-generated
  explanations can improve human decision accuracy (mean improvement = {(p2-p1).mean():.1%} across
  participants, Cohen's d = {d:.2f}); the small sample size limits statistical power, and this
  should be treated as a supporting, exploratory result rather than the paper's central claim.\"""")


if __name__ == "__main__":
    main()
