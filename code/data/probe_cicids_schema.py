"""
Run this FIRST, before anything else, once you've downloaded the 8
CICIDS2017 CSV files into data/CICIDS2017/. It checks the real column
names, label values, and data-quality issues (NaN/Infinity) so the loader
can be built correctly on the first try instead of guessing.

Send me the full printed output.
"""
from pathlib import Path
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "CICIDS2017"


def main():
    if not DATA_DIR.exists():
        print(f"Directory not found: {DATA_DIR}\nCreate it and put the 8 CICIDS2017 CSVs there first.")
        return

    csv_files = sorted(DATA_DIR.glob("*.csv"))
    print(f"Found {len(csv_files)} CSV file(s) in {DATA_DIR}:")
    for f in csv_files:
        print(f"  {f.name}")

    if not csv_files:
        print("No CSV files found - nothing to probe.")
        return

    # Inspect just the first file in detail
    first = csv_files[0]
    print(f"\n{'='*70}\nDetailed inspection of: {first.name}\n{'='*70}")
    df = pd.read_csv(first, nrows=5000, low_memory=False)
    print(f"Shape (first 5000 rows sampled): {df.shape}")
    print(f"\nColumn names (exact, including any leading/trailing spaces):")
    for c in df.columns:
        print(f"  {repr(c)}")

    label_candidates = [c for c in df.columns if "label" in c.lower()]
    print(f"\nLikely label column(s): {label_candidates}")
    for lc in label_candidates:
        print(f"\nValue counts for {repr(lc)} (first 5000 rows):")
        print(df[lc].value_counts())

    print(f"\nAny NaN values in this sample? {df.isna().any().any()}")
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    inf_check = df[numeric_cols].apply(lambda s: np.isinf(s).sum()) if len(numeric_cols) else pd.Series(dtype=int)
    print(f"Columns with Infinity values (first 5000 rows): {inf_check[inf_check > 0].to_dict()}")

    print(f"\n{'='*70}\nQuick label check across ALL {len(csv_files)} files (just the label column, fast)\n{'='*70}")
    all_labels = []
    for f in csv_files:
        try:
            d = pd.read_csv(f, usecols=lambda c: "label" in c.lower(), low_memory=False)
            lc = d.columns[0]
            print(f"  {f.name}: {d[lc].value_counts().to_dict()}")
            all_labels.append(d[lc])
        except Exception as e:
            print(f"  {f.name}: ERROR reading - {e}")

    if all_labels:
        combined = pd.concat(all_labels)
        print(f"\nCombined label distribution across all files:")
        print(combined.value_counts())


if __name__ == "__main__":
    main()
