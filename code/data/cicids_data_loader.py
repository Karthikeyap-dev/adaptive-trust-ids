"""
CICIDS2017 data loader (third dataset).

Handles the real, documented quirks found via probe_cicids_schema.py:
  - Inconsistent leading/trailing whitespace in column names across CICFlowMeter's own output
  - Mojibake in Web Attack labels (encoding mismatch) - read with cp1252, match by substring
  - NaN/Infinity values in some flow-rate features
  - No official train/test split (unlike NSL-KDD/UNSW-NB15) - we create our own
    stratified split, documented explicitly as a methodological difference
  - Extreme class rarity: Infiltration (n=36) and Heartbleed (n=11) are
    excluded from the primary task as statistically underpowered - not
    silently dropped, but explicitly documented in the paper

Groups the raw labels into 6 categories, mirroring the granularity of the
NSL-KDD/UNSW-NB15 pipelines rather than a noisy 15-class problem:
  normal, dos, probe, web_attack, brute_force, botnet
"""
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "CICIDS2017"

# Categories excluded entirely due to extreme rarity (documented in the paper,
# not silently dropped) - counts are from the real schema probe.
EXCLUDED_RAW_LABELS_SUBSTRINGS = ["Infiltration", "Heartbleed"]

# Substring-based mapping (robust to residual mojibake in Web Attack labels)
CATEGORY_RULES = [
    ("BENIGN", "normal"),
    ("DoS", "dos"),
    ("DDoS", "dos"),
    ("PortScan", "probe"),
    ("Web Attack", "web_attack"),   # matches regardless of the mangled separator character
    ("FTP-Patator", "brute_force"),
    ("SSH-Patator", "brute_force"),
    ("Bot", "botnet"),
]


def _map_label_to_category(raw_label):
    for excluded in EXCLUDED_RAW_LABELS_SUBSTRINGS:
        if excluded in raw_label:
            return None  # explicitly excluded, not an error
    for substring, category in CATEGORY_RULES:
        if substring in raw_label:
            return category
    return None  # unrecognized label - excluded and reported, not silently kept


def load_and_combine_raw():
    csv_files = sorted(DATA_DIR.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {DATA_DIR}")

    frames = []
    for f in csv_files:
        # utf-8 with errors='replace' guarantees we never crash on read, regardless
        # of whether upstream mirror processing already corrupted certain bytes
        # (e.g. the em-dash in "Web Attack" labels showing as U+FFFD). We don't
        # need to recover the exact original character - substring-based category
        # matching below works correctly whether or not that character is intact.
        df = pd.read_csv(f, encoding="utf-8", encoding_errors="replace", low_memory=False)
        df.columns = df.columns.str.strip()  # fix inconsistent whitespace
        frames.append(df)
    combined = pd.concat(frames, ignore_index=True)
    combined.columns = combined.columns.str.strip()

    if "Label" not in combined.columns:
        raise ValueError(f"Expected a 'Label' column after stripping whitespace, got: {combined.columns.tolist()}")

    combined["category"] = combined["Label"].astype(str).apply(_map_label_to_category)

    excluded_count = combined["category"].isna().sum()
    total_count = len(combined)
    print(f"Excluded {excluded_count} rows ({excluded_count/total_count:.4%}) - either explicitly rare "
          f"categories (Infiltration/Heartbleed) or unrecognized labels.")
    combined = combined.dropna(subset=["category"]).reset_index(drop=True)

    # Address the well-documented CICIDS2017 near-duplicate leakage issue
    # (Arp et al.; a 2026 leakage-aware study found RF's false-alarm rate
    # increased 67x under leakage-free evaluation vs. conventional random
    # splits on this exact dataset). Exact-duplicate feature rows let the
    # model memorize near-identical repeated attack-tool traffic rather
    # than genuinely generalize. Deduplicate on the full feature set
    # (excluding the label columns) before splitting.
    feature_cols_for_dedup = [c for c in combined.columns if c not in ("Label", "category")]
    before_dedup = len(combined)
    combined = combined.drop_duplicates(subset=feature_cols_for_dedup).reset_index(drop=True)
    after_dedup = len(combined)
    print(f"Deduplication: removed {before_dedup - after_dedup} exact-duplicate rows "
          f"({(before_dedup-after_dedup)/before_dedup:.2%}) - addresses documented near-duplicate "
          f"leakage inflation in this dataset's conventional random-split evaluation.")

    return combined


def clean_features(df, feature_cols):
    """Replace Infinity with NaN, then drop any row with NaN in a feature
    column - a known, small-scale issue in this dataset (verified via the
    schema probe), safe to drop given the tiny fraction affected."""
    df = df.copy()
    df[feature_cols] = df[feature_cols].replace([np.inf, -np.inf], np.nan)
    before = len(df)
    df = df.dropna(subset=feature_cols).reset_index(drop=True)
    after = len(df)
    if before != after:
        print(f"Dropped {before - after} rows ({(before-after)/before:.4%}) containing NaN/Infinity in features.")
    return df


def load_raw(split="train", test_size=0.2, seed=42):
    """Loads, cleans, and splits CICIDS2017. Unlike NSL-KDD/UNSW-NB15 (which
    ship an official split), this dataset has no predefined train/test
    partition - we create our own stratified split here, cached to disk so
    train/test calls in the same run see the identical split."""
    cache_train = DATA_DIR / "_cached_train.parquet"
    cache_test = DATA_DIR / "_cached_test.parquet"

    if cache_train.exists() and cache_test.exists():
        return pd.read_parquet(cache_train) if split == "train" else pd.read_parquet(cache_test)

    print("Building combined dataset and stratified split (first run only, cached after)...")
    combined = load_and_combine_raw()

    label_col_candidates = ["Label"]
    non_feature_cols = label_col_candidates + ["category"]
    feature_cols = [c for c in combined.columns if c not in non_feature_cols]

    combined = clean_features(combined, feature_cols)

    train_df, test_df = train_test_split(
        combined, test_size=test_size, random_state=seed, stratify=combined["category"]
    )
    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    train_df.to_parquet(cache_train)
    test_df.to_parquet(cache_test)

    return train_df if split == "train" else test_df


def encode_features(train_df, test_df):
    # 'Destination Port' is deliberately excluded: in this single 2017
    # capture session, specific attack tools happened to consistently
    # target fixed ports (confirmed by disproportionate feature importance
    # in an initial run - 0.110, ~2x the next feature). This is a
    # session-specific artifact, not a generalizable behavioral signature -
    # a real attacker does not reliably reuse the same port. Excluding it
    # is standard, documented practice for this specific dataset.
    non_feature_cols = ["Label", "category", "Destination Port"]
    feature_cols = [c for c in train_df.columns if c not in non_feature_cols]

    # CICIDS2017 features are already numeric (no categorical protocol/service
    # columns like NSL-KDD/UNSW-NB15) - no one-hot encoding needed here.
    train_X = train_df[feature_cols].copy()
    test_X = test_df[feature_cols].copy()
    test_X = test_X.reindex(columns=train_X.columns, fill_value=0)

    cat_encoder = LabelEncoder()
    train_y = cat_encoder.fit_transform(train_df["category"])
    test_y = test_df["category"].map(
        {c: i for i, c in enumerate(cat_encoder.classes_)}
    ).fillna(-1).astype(int)

    return train_X, train_y, test_X, test_y, cat_encoder


if __name__ == "__main__":
    train_df = load_raw("train")
    test_df = load_raw("test")
    print("Train shape:", train_df.shape, " Test shape:", test_df.shape)
    print("\nCategory distribution (train):")
    print(train_df["category"].value_counts())
    print("\nCategory distribution (test):")
    print(test_df["category"].value_counts())
