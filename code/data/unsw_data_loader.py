"""
UNSW-NB15 data loading and preprocessing (second benchmark dataset).

Mirrors data_loader.py's structure for NSL-KDD, so the rest of the
pipeline (trust engine, arbitration, QPSO, XAI) works against this
dataset with no changes - only the loading/encoding layer is
dataset-specific.

Expects UNSW_NB15_training-set.csv and UNSW_NB15_testing-set.csv in
data/UNSW_NB15/ (the standard, pre-split ML-ready release - 175,341
train / 82,332 test rows, 45 columns, 10-class attack_cat).
"""
from pathlib import Path
import pandas as pd
from sklearn.preprocessing import LabelEncoder

PROJECT_ROOT = Path(__file__).resolve().parent.parent
REPO = PROJECT_ROOT / "data" / "UNSW_NB15"

CAT_COLS = ["proto", "service", "state"]

# Some mirrors of this dataset have minor spelling/casing inconsistencies
# in attack_cat (e.g. "Backdoors" vs "Backdoor") - normalize them here.
CATEGORY_NORMALIZATION = {
    "Backdoors": "Backdoor",
}


def load_raw(split="train"):
    fname = "UNSW_NB15_training-set.csv" if split == "train" else "UNSW_NB15_testing-set.csv"
    path = REPO / fname
    if not path.exists():
        raise FileNotFoundError(
            f"Could not find {path}\n"
            f"Put the UNSW-NB15 training/testing CSVs in: {REPO}"
        )
    df = pd.read_csv(path)
    df["category"] = df["attack_cat"].fillna("Normal").astype(str).str.strip()
    df["category"] = df["category"].replace(CATEGORY_NORMALIZATION)
    return df


def encode_features(train_df, test_df):
    """One-hot encode categorical features (proto/service/state), align
    train/test columns, and label-encode the category target - same
    pattern as the NSL-KDD loader's encode_features."""
    feature_cols = [c for c in train_df.columns if c not in ("id", "attack_cat", "label", "category")]
    cat_cols_present = [c for c in CAT_COLS if c in feature_cols]

    train_X = pd.get_dummies(train_df[feature_cols], columns=cat_cols_present)
    test_X = pd.get_dummies(test_df[feature_cols], columns=cat_cols_present)
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
