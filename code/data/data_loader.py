"""
NSL-KDD data loading and preprocessing.

Loads KDDTrain+ / KDDTest+, assigns column names, maps the 22+ attack
labels down to 5 categories (normal, dos, probe, r2l, u2r) - this category
column is what the Adaptive Trust Engine later keys its per-category
trust scores on - and encodes categorical features for a classifier.

Paths are computed relative to this file's location, so this works no
matter what the project folder is named or where it's placed on disk.
"""
from pathlib import Path
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# This file lives in <project_root>/src/data_loader.py
PROJECT_ROOT = Path(__file__).resolve().parent.parent
REPO = PROJECT_ROOT / "data" / "NSL_KDD_repo"

COLUMNS = [
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes",
    "land", "wrong_fragment", "urgent", "hot", "num_failed_logins", "logged_in",
    "num_compromised", "root_shell", "su_attempted", "num_root",
    "num_file_creations", "num_shells", "num_access_files", "num_outbound_cmds",
    "is_host_login", "is_guest_login", "count", "srv_count", "serror_rate",
    "srv_serror_rate", "rerror_rate", "srv_rerror_rate", "same_srv_rate",
    "diff_srv_rate", "srv_diff_host_rate", "dst_host_count",
    "dst_host_srv_count", "dst_host_same_srv_rate", "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate",
    "dst_host_serror_rate", "dst_host_srv_serror_rate", "dst_host_rerror_rate",
    "dst_host_srv_rerror_rate", "label", "difficulty",
]

ATTACK_TO_CATEGORY = {
    "back": "dos", "land": "dos", "neptune": "dos", "pod": "dos",
    "smurf": "dos", "teardrop": "dos", "apache2": "dos", "udpstorm": "dos",
    "processtable": "dos", "worm": "dos", "mailbomb": "dos",
    "ipsweep": "probe", "nmap": "probe", "portsweep": "probe", "satan": "probe",
    "mscan": "probe", "saint": "probe",
    "buffer_overflow": "u2r", "loadmodule": "u2r", "perl": "u2r",
    "rootkit": "u2r", "httptunnel": "u2r", "ps": "u2r", "sqlattack": "u2r",
    "xterm": "u2r",
    "ftp_write": "r2l", "guess_passwd": "r2l", "imap": "r2l", "multihop": "r2l",
    "phf": "r2l", "spy": "r2l", "warezclient": "r2l", "warezmaster": "r2l",
    "snmpgetattack": "r2l", "named": "r2l", "xlock": "r2l", "xsnoop": "r2l",
    "sendmail": "r2l", "snmpguess": "r2l",
    "normal": "normal",
}


def load_raw(split="train"):
    path = REPO / ("KDDTrain+.txt" if split == "train" else "KDDTest+.txt")
    if not path.exists():
        raise FileNotFoundError(
            f"Could not find {path}\n"
            f"Expected the NSL-KDD dataset at: {REPO}\n"
            f"Fix: from the project root, run:\n"
            f"  git clone https://github.com/defcom17/NSL_KDD.git data/NSL_KDD_repo"
        )
    df = pd.read_csv(path, names=COLUMNS)
    df = df.drop(columns=["difficulty"])
    df["category"] = df["label"].map(ATTACK_TO_CATEGORY)
    unmapped = df["category"].isna().sum()
    if unmapped:
        print(f"[warn] {unmapped} rows had unmapped attack labels, dropping them")
        df = df.dropna(subset=["category"])
    df["is_attack"] = (df["category"] != "normal").astype(int)
    return df


def encode_features(train_df, test_df):
    """One-hot encode categorical features, label-encode the target category.
    Fit encoders on train, apply to both, and align columns so test has
    the same feature set as train (missing dummy columns filled with 0)."""
    cat_cols = ["protocol_type", "service", "flag"]
    feature_cols = [c for c in train_df.columns if c not in ("label", "category", "is_attack")]

    train_X = pd.get_dummies(train_df[feature_cols], columns=cat_cols)
    test_X = pd.get_dummies(test_df[feature_cols], columns=cat_cols)
    test_X = test_X.reindex(columns=train_X.columns, fill_value=0)

    cat_encoder = LabelEncoder()
    train_y_cat = cat_encoder.fit_transform(train_df["category"])
    test_y_cat = test_df["category"].map(
        {c: i for i, c in enumerate(cat_encoder.classes_)}
    ).fillna(-1).astype(int)

    return train_X, train_y_cat, test_X, test_y_cat, cat_encoder


if __name__ == "__main__":
    train_df = load_raw("train")
    test_df = load_raw("test")
    print("Train shape:", train_df.shape, " Test shape:", test_df.shape)
    print("\nCategory distribution (train):")
    print(train_df["category"].value_counts())
    print("\nCategory distribution (test):")
    print(test_df["category"].value_counts())
