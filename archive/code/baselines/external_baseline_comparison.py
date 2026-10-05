"""
External baseline comparison table.

Addresses: "no comparison against published external baselines."

IMPORTANT: the accuracy figures below are transcribed from published papers
found via literature search - verify each citation yourself before
submission (check the actual paper, not just this transcription) and
update the BibTeX keys to match your reference manager. Do not treat this
as a substitute for reading the cited papers.

Methodological note for the paper: many published NSL-KDD papers reporting
~99% accuracy use an EASIER evaluation protocol (k-fold cross-validation or
a random split within KDDTrain+ only), NOT the official KDDTrain+/KDDTest+
split, which deliberately includes attack subtypes absent from training.
The comparison below flags each entry's protocol where known - this
distinction is important to state explicitly in your paper, since it
explains why your (correctly, harder-protocol) accuracy is lower than some
other papers' numbers, without that being a weakness of your method.
"""
from pathlib import Path
import json
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = PROJECT_ROOT / "outputs"

# Verify each of these against the original paper before submission.
EXTERNAL_BASELINES = [
    {
        "citation": "Wu et al. (2024), CMC",
        "method": "Enhanced Random Forest (SMOTE + K-means hybrid sampling)",
        "protocol": "Official KDDTrain+/KDDTest+ split",
        "test_accuracy": 0.7847,
        "notes": "Directly comparable protocol to this work; closest published reference point.",
    },
    {
        "citation": "Multiple studies (various, see literature review)",
        "method": "RandomForest / SVM / DT, various preprocessing",
        "protocol": "K-fold CV or random split WITHIN KDDTrain+ only",
        "test_accuracy": 0.98,  # representative of the ~98-99.8% cluster commonly reported this way
        "notes": "NOT directly comparable - easier protocol, no held-out novel-attack test set. "
                 "Cite as context, not as a same-protocol comparison.",
    },
]


def main():
    baseline = json.load(open(OUT_DIR / "baseline_results.json")) if (OUT_DIR / "baseline_results.json").exists() else None
    rows = list(EXTERNAL_BASELINES)
    if baseline:
        rows.append({
            "citation": "This work",
            "method": "RandomForest (class-balanced), Adaptive Trust Engine + QPSO arbitration",
            "protocol": "Official KDDTrain+/KDDTest+ split",
            "test_accuracy": baseline["overall_accuracy"],
            "notes": "Same protocol as Wu et al. (2024) - directly comparable.",
        })

    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "external_baseline_comparison.csv", index=False)
    print(df.to_string(index=False))
    print(f"\nSaved to {OUT_DIR / 'external_baseline_comparison.csv'}")
    print("\nREMINDER: verify each citation against the original paper before using in your submission.")


if __name__ == "__main__":
    main()
