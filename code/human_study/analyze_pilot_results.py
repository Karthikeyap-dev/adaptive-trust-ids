"""
Analyzes results from the v2 (two-phase, anchoring-bias-fixed) pilot study.

Key sanity check built in: flags if phase1_agrees_with_system is
suspiciously close to 1.0 again (would indicate the bias wasn't actually
fixed, or something else is contaminating the data) - a real independent
pilot should show SOME disagreement between blind human judgment and the
system, especially on the categories the system struggles with (r2l/u2r).
"""
from pathlib import Path
import json
import pandas as pd
from scipy.stats import chi2

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = PROJECT_ROOT / "outputs"
PILOT_RESULTS_PATH = OUT_DIR / "pilot_study_results.csv"


def fleiss_kappa(rating_matrix):
    n_items, n_categories = rating_matrix.shape
    n_raters = rating_matrix.sum(axis=1)[0]
    p_j = rating_matrix.sum(axis=0) / (n_items * n_raters)
    P_i = ((rating_matrix ** 2).sum(axis=1) - n_raters) / (n_raters * (n_raters - 1))
    P_bar = P_i.mean()
    P_e_bar = (p_j ** 2).sum()
    if P_e_bar == 1:
        return 1.0
    return (P_bar - P_e_bar) / (1 - P_e_bar)


def main():
    if not PILOT_RESULTS_PATH.exists():
        print(f"No pilot results found. Run human_pilot_study.py with at least 3 participants first.")
        return

    df = pd.read_csv(PILOT_RESULTS_PATH)
    n_participants = df["participant_id"].nunique()
    print(f"Loaded {len(df)} responses from {n_participants} participant(s): {df['participant_id'].unique().tolist()}\n")

    phase1_agree = df["phase1_agrees_with_system"].mean()
    print(f"--- Sanity check ---")
    print(f"Phase 1 (BLIND) agreement with system: {phase1_agree:.3f}")
    if phase1_agree > 0.95:
        print("  WARNING: still suspiciously close to 1.0. Double-check the tool wasn't accidentally "
              "showing the system's prediction before Phase 1, and that participants understood the "
              "instructions.")
    else:
        print("  Looks like genuine independent variation - good.")

    print(f"\n--- Explanation influence (report this as the primary pilot finding) ---")
    changed = df["changed_mind_after_explanation"].mean()
    print(f"Participants changed their answer after seeing the explanation: {changed:.3f} of responses")
    changed_df = df[df["changed_mind_after_explanation"]]
    became_correct, became_incorrect = 0, 0
    mcnemar_stat, mcnemar_p = None, None
    if len(changed_df) > 0:
        became_correct = int((changed_df["phase2_correct"] & ~changed_df["phase1_correct"]).sum())
        became_incorrect = int((~changed_df["phase2_correct"] & changed_df["phase1_correct"]).sum())
        print(f"  Of those mind-changes: {became_correct} became correct, {became_incorrect} became incorrect "
              f"(net effect: {'positive' if became_correct > became_incorrect else 'negative or neutral'})")

        # McNemar's test (with continuity correction) on the discordant pairs only -
        # tests whether the direction of mind-changes (helped vs hurt) is
        # significantly lopsided, or could plausibly be due to chance alone.
        b, c = became_correct, became_incorrect
        if b + c > 0:
            mcnemar_stat = (abs(b - c) - 1) ** 2 / (b + c)
            mcnemar_p = 1 - chi2.cdf(mcnemar_stat, df=1)
            print(f"  McNemar's test (continuity-corrected): chi2={mcnemar_stat:.3f}, p={mcnemar_p:.6f}  "
                  f"{'*** significant - explanation influence is directionally real, not chance' if mcnemar_p < 0.05 else 'not significant'}")
        else:
            print("  No discordant pairs - McNemar's test not applicable.")

    print(f"\n--- Explanation quality ratings (report this as a primary pilot finding) ---")
    print(f"Usefulness: mean={df['usefulness_rating'].mean():.2f}, std={df['usefulness_rating'].std():.2f} (1-5)")
    print(f"Clarity:    mean={df['clarity_rating'].mean():.2f}, std={df['clarity_rating'].std():.2f} (1-5)")

    print(f"\n--- Blind human accuracy (report as SECONDARY/exploratory only - see caveat) ---")
    phase1_accuracy = df["phase1_correct"].mean()
    phase2_accuracy = df["phase2_correct"].mean()
    system_accuracy = df["system_correct"].mean()
    print(f"Phase 1 human accuracy (blind, vs ground truth): {phase1_accuracy:.3f}")
    print(f"Phase 2 human accuracy (after seeing system + explanation): {phase2_accuracy:.3f}")
    print(f"System accuracy on the same alerts:                        {system_accuracy:.3f}")
    print("CAVEAT: participants were non-domain-experts given only a field glossary and loose orientation "
          "hints (not the system's own decision rules), but this comparison should still be reported as "
          "exploratory/secondary evidence, not a rigorous human-baseline claim, given the small sample "
          "and non-expert raters.")

    summary = {
        "n_participants": int(n_participants), "n_responses": int(len(df)),
        "phase1_blind_agreement_with_system": float(phase1_agree),
        "phase1_human_accuracy": float(phase1_accuracy),
        "phase1_human_error_rate": float(1 - phase1_accuracy),
        "phase2_assisted_accuracy": float(phase2_accuracy),
        "system_accuracy_same_alerts": float(system_accuracy),
        "phase2_agreement_with_system": float(df["phase2_agrees_with_system"].mean()),
        "fraction_changed_mind": float(changed),
        "mind_changes_became_correct": became_correct,
        "mind_changes_became_incorrect": became_incorrect,
        "mcnemar_chi2": float(mcnemar_stat) if mcnemar_stat is not None else None,
        "mcnemar_p_value": float(mcnemar_p) if mcnemar_p is not None else None,
        "usefulness_mean": float(df["usefulness_rating"].mean()), "usefulness_std": float(df["usefulness_rating"].std()),
        "clarity_mean": float(df["clarity_rating"].mean()), "clarity_std": float(df["clarity_rating"].std()),
    }

    if n_participants >= 3:
        pivot = df.pivot_table(index="alert_index_in_study", columns="phase1_blind_label", aggfunc="size", fill_value=0)
        for cat in ["normal", "dos", "probe", "r2l", "u2r"]:
            if cat not in pivot.columns:
                pivot[cat] = 0
        pivot = pivot[["normal", "dos", "probe", "r2l", "u2r"]]
        kappa = fleiss_kappa(pivot.values)
        print(f"\nFleiss' kappa (blind-phase inter-rater agreement, {n_participants} participants): {kappa:.3f}")
        summary["fleiss_kappa_blind_phase"] = float(kappa)

    with open(OUT_DIR / "pilot_study_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved summary to {OUT_DIR / 'pilot_study_summary.json'}")


if __name__ == "__main__":
    main()
