"""
CICIDS2017 temporal-split robustness check (Section 4.1).

Same preprocessing as data/cicids_data_loader.py (label mapping, Infiltration/Heartbleed
exclusion, exact-duplicate removal, NaN/Infinity removal, 'Destination Port' excluded) and the
same RandomForest settings as baselines/cicids_train_baseline.py. Instead of the stratified
random 80/20 split, each category is split in TIME order: its earliest 80% of records (by capture
day, Monday -> Friday, and row order within each file) train the model, its latest 20% test it.
This checks that the near-perfect random-split accuracy is not an artifact of neighbouring
records leaking between train and test.

Run from code/:   python3 statistics/cicids_temporal_split.py [optional/path/to/CICIDS2017]   (~5-15 min)
Writes outputs/cicids_temporal_split_rerun.json
"""
from pathlib import Path
import json, re, sys
import numpy as np, pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

CODE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(CODE / "data"))
import cicids_data_loader as L

DAY_ORDER = {"monday": 0, "tuesday": 1, "wednesday": 2, "thursday": 3, "friday": 4}
TRAIN_FRAC = 0.8


def chronological_key(path):
    name = path.name.lower()
    day = next((v for k, v in DAY_ORDER.items() if name.startswith(k)), 9)
    half = 0 if "morning" in name else 1          # Thursday/Friday have morning and afternoon files
    sub = 0 if "portscan" in name else 1          # Friday afternoon: PortScan precedes DDoS
    return (day, half, sub, name)


def find_csvs():
    """Look in the loader's DATA_DIR and in <project>/datasets/CICIDS2017 (including subfolders)."""
    candidates = [L.DATA_DIR, CODE.parent / "datasets" / "CICIDS2017", CODE / "datasets" / "CICIDS2017"]
    if len(sys.argv) > 1:
        candidates.insert(0, Path(sys.argv[1]).expanduser())
    for c in candidates:
        if c.is_dir():
            files = [f for f in c.rglob("*.csv") if not f.name.startswith(("_", "."))]
            if files:
                print(f"Using CICIDS2017 CSVs from: {c}")
                return files
    raise FileNotFoundError("No CICIDS2017 CSV files found in: " + ", ".join(str(c) for c in candidates)
                            + "\nPass the folder explicitly: python3 statistics/cicids_temporal_split.py /path/to/CICIDS2017")


def load_chronological():
    files = sorted(find_csvs(), key=chronological_key)
    print("File order used:", [f.name for f in files])
    frames = []
    for i, f in enumerate(files):
        df = pd.read_csv(f, encoding="utf-8", encoding_errors="replace", low_memory=False)
        df.columns = df.columns.str.strip()
        df["_order"] = np.arange(len(df)) + i * 10**7      # global time order: file, then row
        frames.append(df)
    d = pd.concat(frames, ignore_index=True)
    d["category"] = d["Label"].astype(str).apply(L._map_label_to_category)
    d = d.dropna(subset=["category"]).reset_index(drop=True)
    feats = [c for c in d.columns if c not in ("Label", "category", "_order")]
    d = d.drop_duplicates(subset=feats).reset_index(drop=True)      # keeps the earliest occurrence
    d = L.clean_features(d, feats)
    return d.sort_values("_order").reset_index(drop=True)


def main():
    d = load_chronological()
    train_parts, test_parts = [], []
    for cat, g in d.groupby("category", sort=True):
        k = int(round(len(g) * TRAIN_FRAC))
        train_parts.append(g.iloc[:k]); test_parts.append(g.iloc[k:])
    train = pd.concat(train_parts).drop(columns="_order").reset_index(drop=True)
    test = pd.concat(test_parts).drop(columns="_order").reset_index(drop=True)
    Xtr, ytr, Xte, yte, enc = L.encode_features(train, test)
    clf = RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=42, class_weight="balanced_subsample")
    print(f"Training RandomForest on {len(train)} records, testing on {len(test)} ...")
    clf.fit(Xtr, ytr)
    pred = clf.predict(Xte)
    res = {"split_type": "per_category_temporal", "train_frac_per_category": TRAIN_FRAC,
           "file_order": "Monday to Friday (chronological), row order within each file",
           "train_n": int(len(train)), "test_n": int(len(test)),
           "accuracy": float(accuracy_score(yte, pred)), "macro_f1": float(f1_score(yte, pred, average="macro")),
           "category_order": list(enc.classes_),
           "confusion_matrix": confusion_matrix(yte, pred, labels=range(len(enc.classes_))).tolist()}
    out = CODE / "outputs" / "cicids_temporal_split_rerun.json"
    json.dump(res, open(out, "w"), indent=1)
    print(f"\nTemporal split: accuracy {res['accuracy']:.4f}, macro F1 {res['macro_f1']:.4f} "
          f"(train {res['train_n']}, test {res['test_n']})")
    old = CODE / "outputs" / "cicids_baseline_results.json"
    if old.exists():
        o = json.load(open(old))
        acc = o.get("overall_accuracy", o.get("accuracy"))
        if acc is not None:
            print(f"Random stratified split (paper): accuracy {acc:.4f}  ->  change {(acc - res['accuracy']) * 100:.2f} points")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
