"""
Milestone 4: Confidence calibration diagnostic.

Raw model confidence (max class probability) is usually overconfident.
This measures Expected Calibration Error (ECE) before and after applying
isotonic regression calibration, and saves the calibrator plus per-row
calibrated confidence.

Known finding (documented in README): calibration fit on in-distribution
data tends to get WORSE on the test set here, because the test set
contains genuinely out-of-distribution novel attacks the calibrator
never saw evidence for. This is expected, not a bug - it motivates the
need for adaptive/online calibration rather than a static one-time fit.
"""
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import train_test_split

from data_loader import load_raw, encode_features

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = PROJECT_ROOT / "models"
OUT_DIR = PROJECT_ROOT / "outputs"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR.mkdir(parents=True, exist_ok=True)


def expected_calibration_error(confidences, correct, n_bins=10):
    """Standard ECE: bin predictions by confidence, compare average confidence
    to actual accuracy within each bin, weight by bin size."""
    confidences = np.asarray(confidences)
    correct = np.asarray(correct)
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n = len(confidences)
    bin_details = []
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (confidences > lo) & (confidences <= hi) if i > 0 else (confidences >= lo) & (confidences <= hi)
        if mask.sum() == 0:
            continue
        bin_acc = correct[mask].mean()
        bin_conf = confidences[mask].mean()
        weight = mask.sum() / n
        ece += weight * abs(bin_acc - bin_conf)
        bin_details.append({
            "bin_range": f"{lo:.1f}-{hi:.1f}", "count": int(mask.sum()),
            "avg_confidence": float(bin_conf), "avg_accuracy": float(bin_acc),
        })
    return ece, bin_details


def main():
    print("Loading data and re-splitting train into train_sub / calibration set...")
    train_df = load_raw("train")
    test_df = load_raw("test")
    train_X, train_y, test_X, test_y, cat_encoder = encode_features(train_df, test_df)

    train_sub_X, calib_X, train_sub_y, calib_y = train_test_split(
        train_X, train_y, test_size=0.2, random_state=42, stratify=train_y
    )

    print("Training base classifier on train_sub...")
    base_clf = RandomForestClassifier(
        n_estimators=200, n_jobs=-1, random_state=42, class_weight="balanced_subsample"
    )
    base_clf.fit(train_sub_X, train_sub_y)

    raw_proba = base_clf.predict_proba(test_X)
    raw_pred = raw_proba.argmax(axis=1)
    raw_conf = raw_proba.max(axis=1)
    raw_correct = (raw_pred == test_y).astype(int)
    ece_before, bins_before = expected_calibration_error(raw_conf, raw_correct)

    print("Fitting isotonic calibration on held-out calibration set...")
    calib_proba = base_clf.predict_proba(calib_X)
    calib_conf = calib_proba.max(axis=1)
    calib_pred = calib_proba.argmax(axis=1)
    calib_correct_binary = (calib_pred == calib_y).astype(int)

    calibrator = IsotonicRegression(out_of_bounds="clip")
    calibrator.fit(calib_conf, calib_correct_binary)

    cal_pred = raw_pred
    cal_conf = calibrator.predict(raw_conf)
    cal_correct = raw_correct
    ece_after, bins_after = expected_calibration_error(cal_conf, cal_correct)

    print(f"\nECE before calibration: {ece_before:.4f}")
    print(f"ECE after calibration:  {ece_after:.4f}")
    improvement = (ece_before - ece_after) / ece_before * 100 if ece_before > 0 else 0
    print(f"Change: {improvement:.1f}%  "
          f"({'improved' if improvement > 0 else 'got worse - see README, this is an expected finding'})\n")

    print("Reliability diagram data (before -> after), by confidence bin:")
    for b_before, b_after in zip(bins_before, bins_after):
        print(f"  {b_before['bin_range']}: "
              f"conf={b_before['avg_confidence']:.2f}->{b_after['avg_confidence']:.2f}  "
              f"acc={b_before['avg_accuracy']:.2f}->{b_after['avg_accuracy']:.2f}  "
              f"n={b_before['count']}")

    joblib.dump(base_clf, MODEL_DIR / "baseline_rf_for_calibration.joblib")
    joblib.dump(calibrator, MODEL_DIR / "confidence_calibrator.joblib")
    with open(OUT_DIR / "calibration_results.json", "w") as f:
        json.dump({
            "ece_before": ece_before, "ece_after": ece_after,
            "change_pct": improvement,
            "bins_before": bins_before, "bins_after": bins_after,
        }, f, indent=2)

    records = pd.DataFrame({
        "true_category": [cat_encoder.classes_[i] for i in test_y],
        "pred_category": [cat_encoder.classes_[i] for i in cal_pred],
        "raw_confidence": raw_conf,
        "calibrated_confidence": cal_conf,
        "correct": cal_correct,
    })
    records.to_csv(OUT_DIR / "calibrated_predictions.csv", index=False)

    print(f"\nSaved base model to {MODEL_DIR / 'baseline_rf_for_calibration.joblib'}")
    print(f"Saved calibrator to {MODEL_DIR / 'confidence_calibrator.joblib'}")
    print(f"Saved calibration metrics to {OUT_DIR / 'calibration_results.json'}")
    print(f"Saved calibrated predictions to {OUT_DIR / 'calibrated_predictions.csv'}")


if __name__ == "__main__":
    main()
