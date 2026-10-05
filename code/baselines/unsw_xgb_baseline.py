"""
XGBoost baseline on UNSW-NB15 - completes the cross-classifier validation
matrix. Mirrors xgb_baseline.py's structure for NSL-KDD.
"""
from pathlib import Path
import json
import joblib
import pandas as pd
from xgboost import XGBClassifier
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
    print(f"Feature count: {train_X.shape[1]}")

    print("Training XGBoost classifier...")
    class_counts = pd.Series(train_y).value_counts()
    weight_map = {cls: len(train_y) / (len(class_counts) * count) for cls, count in class_counts.items()}
    sample_weight = pd.Series(train_y).map(weight_map).values

    clf = XGBClassifier(
        n_estimators=200, max_depth=6, learning_rate=0.1,
        objective="multi:softprob", num_class=len(cat_encoder.classes_),
        n_jobs=-1, random_state=42, eval_metric="mlogloss",
    )
    clf.fit(train_X, train_y, sample_weight=sample_weight)

    print("Evaluating on test set...")
    pred_y = clf.predict(test_X)
    pred_proba = clf.predict_proba(test_X)

    acc = accuracy_score(test_y, pred_y)
    f1_macro = f1_score(test_y, pred_y, average="macro")
    f1_weighted = f1_score(test_y, pred_y, average="weighted")

    labels_present = sorted(set(test_y) | set(pred_y))
    target_names = [cat_encoder.classes_[i] for i in labels_present]
    report = classification_report(test_y, pred_y, labels=labels_present, target_names=target_names, output_dict=True)
    cm = confusion_matrix(test_y, pred_y, labels=labels_present)

    print(f"\nOverall accuracy: {acc:.4f}")
    print(f"Macro F1: {f1_macro:.4f}   Weighted F1: {f1_weighted:.4f}\n")
    print(classification_report(test_y, pred_y, labels=labels_present, target_names=target_names))

    joblib.dump(clf, MODEL_DIR / "unsw_xgb_baseline.joblib")

    results = {
        "classifier": "XGBoost", "dataset": "UNSW-NB15",
        "overall_accuracy": acc, "f1_macro": f1_macro, "f1_weighted": f1_weighted,
        "per_category_report": report, "confusion_matrix": cm.tolist(), "category_order": target_names,
    }
    with open(OUT_DIR / "unsw_xgb_baseline_results.json", "w") as f:
        json.dump(results, f, indent=2)

    pred_records = pd.DataFrame({
        "true_category": [cat_encoder.classes_[i] for i in test_y],
        "pred_category": [cat_encoder.classes_[i] for i in pred_y],
        "raw_confidence": pred_proba.max(axis=1),
        "correct": (test_y == pred_y).astype(int),
    })
    pred_records.to_csv(OUT_DIR / "unsw_xgb_baseline_predictions.csv", index=False)

    print(f"\nSaved to {MODEL_DIR}/ and {OUT_DIR}/ (unsw_xgb_ prefix)")

    rf_results_path = OUT_DIR / "unsw_baseline_results.json"
    if rf_results_path.exists():
        rf = json.load(open(rf_results_path))
        print(f"\n--- Classifier comparison (UNSW-NB15) ---")
        print(f"{'Metric':<15}{'RandomForest':>15}{'XGBoost':>15}")
        print(f"{'Accuracy':<15}{rf['overall_accuracy']:>15.4f}{acc:>15.4f}")
        print(f"{'Macro F1':<15}{rf['f1_macro']:>15.4f}{f1_macro:>15.4f}")


if __name__ == "__main__":
    main()
