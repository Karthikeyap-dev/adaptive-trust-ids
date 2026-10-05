"""
Cheap multi-seed stability check for CICIDS2017.

NOT the expensive session-disjoint redesign (deliberately skipped - see
discussion: most attack categories exist in only one capture file, making
a clean session-disjoint split structurally awkward for this dataset).

This instead re-runs the stratified split + training with different
random seeds, reusing the expensive one-time cleaning (deduplication,
NaN/Infinity handling) which doesn't change across seeds - only the
train/test partition and the RandomForest's internal randomness vary.

Answers a narrower, cheaper question than session-disjoint validation:
is the ~99.9% accuracy a stable property of this cleaned dataset under
different random splits, or does it fluctuate?
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report

from cicids_data_loader import load_and_combine_raw, clean_features

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = PROJECT_ROOT / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

N_SEEDS = 10
NON_FEATURE_COLS = ["Label", "category", "Destination Port"]


def run_one_seed(combined, feature_cols, seed):
    train_df, test_df = train_test_split(
        combined, test_size=0.2, random_state=seed, stratify=combined["category"]
    )

    train_X, test_X = train_df[feature_cols], test_df[feature_cols]
    train_y, test_y = train_df["category"], test_df["category"]

    clf = RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=seed, class_weight="balanced_subsample")
    clf.fit(train_X, train_y)
    pred_y = clf.predict(test_X)

    acc = accuracy_score(test_y, pred_y)
    f1_macro = f1_score(test_y, pred_y, average="macro")
    f1_weighted = f1_score(test_y, pred_y, average="weighted")

    report = classification_report(test_y, pred_y, output_dict=True, zero_division=0)
    botnet_f1 = report.get("botnet", {}).get("f1-score", None)

    return {
        "seed": seed, "accuracy": acc, "f1_macro": f1_macro,
        "f1_weighted": f1_weighted, "botnet_f1": botnet_f1,
    }


def main():
    print(f"Loading and cleaning CICIDS2017 once (dedup + NaN/Infinity handling - the expensive part)...")
    combined = load_and_combine_raw()
    feature_cols = [c for c in combined.columns if c not in NON_FEATURE_COLS]
    combined = clean_features(combined, feature_cols)
    print(f"Cleaned dataset: {len(combined)} rows, {len(feature_cols)} features\n")

    results = []
    for seed in range(N_SEEDS):
        r = run_one_seed(combined, feature_cols, seed)
        results.append(r)
        print(f"  seed {seed}: accuracy={r['accuracy']:.4f}  macro_F1={r['f1_macro']:.4f}  "
              f"botnet_F1={r['botnet_f1']:.4f}" if r['botnet_f1'] is not None else
              f"  seed {seed}: accuracy={r['accuracy']:.4f}  macro_F1={r['f1_macro']:.4f}  botnet_F1=N/A")

    df = pd.DataFrame(results)
    df.to_csv(OUT_DIR / "cicids_multi_seed_results.csv", index=False)

    print(f"\n{'='*60}\nSUMMARY (mean +/- std across {N_SEEDS} seeds)\n{'='*60}")
    summary = {}
    for col in ["accuracy", "f1_macro", "f1_weighted", "botnet_f1"]:
        vals = df[col].dropna()
        summary[col] = {"mean": float(vals.mean()), "std": float(vals.std()), "min": float(vals.min()), "max": float(vals.max())}
        print(f"  {col:15s}  {vals.mean():.4f} +/- {vals.std():.4f}  (range: {vals.min():.4f} - {vals.max():.4f})")

    with open(OUT_DIR / "cicids_multi_seed_summary.json", "w") as f:
        json.dump({"n_seeds": N_SEEDS, "summary": summary}, f, indent=2)

    acc_std = summary["accuracy"]["std"]
    print(f"\n{'Stable - accuracy std < 0.001, worth citing as evidence against split-specific luck.' if acc_std < 0.001 else 'Some variability observed - report the range, not just a single number.'}")
    print(f"Saved to {OUT_DIR / 'cicids_multi_seed_summary.json'}")


if __name__ == "__main__":
    main()
