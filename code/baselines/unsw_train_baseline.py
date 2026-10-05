"""
Milestone 1 (UNSW-NB15 second dataset): Baseline intrusion-detection
classifier, mirroring train_baseline.py's structure for NSL-KDD.

Saves to SEPARATE filenames (unsw_ prefix) so nothing overwrites your
existing NSL-KDD results - both datasets' results coexist in models/ and
outputs/.
"""
from pathlib import Path
import json
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score

from unsw_data_loader import load_raw, encode_features

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = PROJECT_ROOT / "models"
OUT_DIR = PROJECT_ROOT / "outputs"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    print("Loading UNSW-NB15 data...")
    train_df = load_raw("train")
    test_df = load_raw("test")

    train_X, train_y, test_X, test_y, cat_encoder = encode_features(train_df, test_df)
    print(f"Feature count after encoding: {train_X.shape[1]}")

    print("Training RandomForest classifier...")
    clf = RandomForestClassifier(
        n_estimators=200, n_jobs=-1, random_state=42, class_weight="balanced_subsample",
    )
    clf.fit(train_X, train_y)

    print("Evaluating on test set...")
    pred_y = clf.predict(test_X)
    pred_proba = clf.predict_proba(test_X)

    acc = accuracy_score(test_y, pred_y)
    f1_macro = f1_score(test_y, pred_y, average="macro")
    f1_weighted = f1_score(test_y, pred_y, average="weighted")

    labels_present = sorted(set(test_y) | set(pred_y))
    target_names = [cat_encoder.classes_[i] for i in labels_present]
    report = classification_report(
        test_y, pred_y, labels=labels_present, target_names=target_names, output_dict=True
    )
    cm = confusion_matrix(test_y, pred_y, labels=labels_present)

    print(f"\nOverall accuracy: {acc:.4f}")
    print(f"Macro F1: {f1_macro:.4f}   Weighted F1: {f1_weighted:.4f}\n")
    print(classification_report(test_y, pred_y, labels=labels_present, target_names=target_names))
    print("Confusion matrix (rows=true, cols=pred), order:", target_names)
    print(cm)

    joblib.dump(clf, MODEL_DIR / "unsw_baseline_rf.joblib")
    joblib.dump(cat_encoder, MODEL_DIR / "unsw_category_encoder.joblib")
    joblib.dump(list(train_X.columns), MODEL_DIR / "unsw_feature_columns.joblib")

    results = {
        "dataset": "UNSW-NB15",
        "overall_accuracy": acc, "f1_macro": f1_macro, "f1_weighted": f1_weighted,
        "per_category_report": report, "confusion_matrix": cm.tolist(), "category_order": target_names,
    }
    with open(OUT_DIR / "unsw_baseline_results.json", "w") as f:
        json.dump(results, f, indent=2)

    pred_records = pd.DataFrame({
        "true_category": [cat_encoder.classes_[i] for i in test_y],
        "pred_category": [cat_encoder.classes_[i] for i in pred_y],
        "raw_confidence": pred_proba.max(axis=1),
        "correct": (test_y == pred_y).astype(int),
    })
    pred_records.to_csv(OUT_DIR / "unsw_baseline_predictions.csv", index=False)

    print(f"\nSaved model/encoder/features to {MODEL_DIR}/ (unsw_ prefix)")
    print(f"Saved results to {OUT_DIR / 'unsw_baseline_results.json'}")
    print(f"Saved predictions to {OUT_DIR / 'unsw_baseline_predictions.csv'}")


if __name__ == "__main__":
    main()
