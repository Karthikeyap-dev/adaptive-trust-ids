"""
Human decision-time analysis (addresses review point #26 - human
workload/decision time was previously unmeasured).

IMPORTANT INTERPRETATION NOTE: the pilot study tool records ONE
timestamp per completed row - the moment the participant submitted
their FINAL (phase 2 / assisted) judgment for that alert. There is no
separate phase1-submission timestamp. This means the time delta
between consecutive alerts measures the TOTAL time spent on an alert
(both the blind phase 1 judgment AND the assisted phase 2 judgment
combined), not phase-specific time. This script computes and reports
that quantity honestly under this interpretation, and does NOT claim
to separate blind-phase time from assisted-phase time, since the data
does not support that finer-grained claim.

Computes, per participant and pooled:
  - mean/median/SD time per alert (inter-alert timestamp deltas)
  - whether time-per-alert differs for judgments where the participant
    changed their mind after the explanation (changed_mind_after_explanation)
    vs. those where they did not
  - outlier handling: the first alert of the session and any gap
    exceeding a configurable threshold (default 10 minutes) are
    excluded from the "time per alert" statistic, since these likely
    reflect a break rather than genuine alert-processing time, and
    reported separately so nothing is silently discarded

Usage: python3 decision_time_analysis.py <pilot_csv_1> [<pilot_csv_2> ...]
Example: python3 decision_time_analysis.py pilot_study_results_CSEbatch3.csv pilot_study_results_INFOSEC.csv
"""
from pathlib import Path
import json
import sys
import numpy as np
import pandas as pd

OUT_DIR = Path(".")
BREAK_THRESHOLD_SECONDS = 600  # gaps longer than this are treated as a break, not decision time


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 decision_time_analysis.py <pilot_csv_1> [<pilot_csv_2> ...]")
        sys.exit(1)

    dfs = []
    for path_str in sys.argv[1:]:
        path = Path(path_str)
        if not path.exists():
            raise FileNotFoundError(f"Missing {path}")
        dfs.append(pd.read_csv(path))
    df = pd.concat(dfs, ignore_index=True)
    print(f"Loaded {len(df)} rows across {len(sys.argv) - 1} file(s), "
          f"{df['participant_id'].nunique()} participants\n")

    df["timestamp"] = pd.to_datetime(df["timestamp"])
    # Order by participant and alert_index_in_study, which reflects the
    # actual order alerts were shown in (NOT necessarily timestamp order,
    # though they should coincide for a well-behaved session)
    df = df.sort_values(["participant_id", "alert_index_in_study"]).reset_index(drop=True)

    all_deltas = []  # (participant_id, alert_index, delta_seconds, changed_mind, is_first, is_break)
    for pid, group in df.groupby("participant_id"):
        group = group.sort_values("alert_index_in_study").reset_index(drop=True)
        timestamps = group["timestamp"].values
        for i in range(len(group)):
            if i == 0:
                all_deltas.append((pid, group.loc[i, "alert_index_in_study"], None,
                                    group.loc[i, "changed_mind_after_explanation"], True, False))
                continue
            delta = (timestamps[i] - timestamps[i - 1]) / np.timedelta64(1, "s")
            is_break = delta > BREAK_THRESHOLD_SECONDS
            all_deltas.append((pid, group.loc[i, "alert_index_in_study"], float(delta),
                                group.loc[i, "changed_mind_after_explanation"], False, is_break))

    delta_df = pd.DataFrame(all_deltas, columns=["participant_id", "alert_index", "delta_seconds",
                                                   "changed_mind", "is_first", "is_break"])

    valid = delta_df[~delta_df["is_first"] & ~delta_df["is_break"]]
    excluded_first = delta_df["is_first"].sum()
    excluded_breaks = delta_df["is_break"].sum()

    print("=" * 70)
    print("OVERALL TIME PER ALERT (both phases combined, per the note above)")
    print("=" * 70)
    print(f"Valid observations: {len(valid)} (excluded {excluded_first} first-alerts-of-session, "
          f"{excluded_breaks} gaps > {BREAK_THRESHOLD_SECONDS}s treated as breaks)")
    print(f"Mean:   {valid['delta_seconds'].mean():.1f} s")
    print(f"Median: {valid['delta_seconds'].median():.1f} s")
    print(f"SD:     {valid['delta_seconds'].std():.1f} s")
    print(f"IQR:    [{valid['delta_seconds'].quantile(0.25):.1f}, {valid['delta_seconds'].quantile(0.75):.1f}] s")

    print(f"\n{'=' * 70}")
    print("TIME PER ALERT: changed mind vs. did not change mind")
    print("=" * 70)
    changed = valid[valid["changed_mind"] == True]  # noqa: E712
    not_changed = valid[valid["changed_mind"] == False]  # noqa: E712
    print(f"Changed mind (n={len(changed)}):     mean={changed['delta_seconds'].mean():.1f}s  "
          f"median={changed['delta_seconds'].median():.1f}s")
    print(f"Did not change (n={len(not_changed)}): mean={not_changed['delta_seconds'].mean():.1f}s  "
          f"median={not_changed['delta_seconds'].median():.1f}s")
    if len(changed) > 1 and len(not_changed) > 1:
        from scipy import stats
        u_stat, p_val = stats.mannwhitneyu(changed["delta_seconds"].dropna(),
                                             not_changed["delta_seconds"].dropna(), alternative="two-sided")
        print(f"Mann-Whitney U test: p={p_val:.4f}")

    print(f"\n{'=' * 70}")
    print("PER-PARTICIPANT SUMMARY")
    print("=" * 70)
    per_participant = valid.groupby("participant_id")["delta_seconds"].agg(["mean", "median", "count"])
    print(per_participant.to_string())

    results = {
        "n_participants": int(df["participant_id"].nunique()),
        "n_valid_observations": int(len(valid)),
        "n_excluded_first": int(excluded_first),
        "n_excluded_breaks": int(excluded_breaks),
        "break_threshold_seconds": BREAK_THRESHOLD_SECONDS,
        "overall_mean_seconds": float(valid["delta_seconds"].mean()),
        "overall_median_seconds": float(valid["delta_seconds"].median()),
        "overall_sd_seconds": float(valid["delta_seconds"].std()),
        "iqr_25": float(valid["delta_seconds"].quantile(0.25)),
        "iqr_75": float(valid["delta_seconds"].quantile(0.75)),
        "changed_mind_mean_seconds": float(changed["delta_seconds"].mean()) if len(changed) else None,
        "not_changed_mean_seconds": float(not_changed["delta_seconds"].mean()) if len(not_changed) else None,
        "per_participant": per_participant.reset_index().to_dict("records"),
    }
    with open(OUT_DIR / "decision_time_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved to decision_time_results.json")


if __name__ == "__main__":
    main()
