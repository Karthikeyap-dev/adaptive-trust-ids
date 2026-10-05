"""
Cross-dataset calibration analysis - extends the NSL-KDD-only calibration
diagnostic to all three RandomForest configurations, computing Expected
Calibration Error (ECE), Brier score, and Negative Log-Likelihood (NLL).
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = PROJECT_ROOT / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

N_BINS = 10


def expected_calibration_error(confidences, correct, n_bins=N_BINS):
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n = len(confidences)
    bins_detail = []
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        in_bin = (confidences > lo) & (confidences <= hi) if i > 0 else (confidences >= lo) & (confidences <= hi)
        count = in_bin.sum()
        if count == 0:
            continue
        avg_conf = confidences[in_bin].mean()
        avg_acc = correct[in_bin].mean()
        ece += (count / n) * abs(avg_conf - avg_acc)
        bins_detail.append({"bin_range": f"{lo:.1f}-{hi:.1f}", "count": int(count),
                             "avg_confidence": float(avg_conf), "avg_accuracy": float(avg_acc)})
    return float(ece), bins_detail


def brier_score(confidences, correct):
    return float(np.mean((confidences - correct) ** 2))


def negative_log_likelihood(confidences, correct, eps=1e-12):
    p = np.clip(confidences, eps, 1 - eps)
    ll = correct * np.log(p) + (1 - correct) * np.log(1 - p)
    return float(-np.mean(ll))


def analyze_dataset(name, preds_path):
    if not preds_path.exists():
        print(f"  [skip] {name}: {preds_path} not found")
        return None
    df = pd.read_csv(preds_path)
    conf = df["raw_confidence"].values.astype(float)
    correct = df["correct"].values.astype(float)

    ece, bins_detail = expected_calibration_error(conf, correct)
    brier = brier_score(conf, correct)
    nll = negative_log_likelihood(conf, correct)

    print(f"  {name}: ECE={ece:.4f}  Brier={brier:.4f}  NLL={nll:.4f}  (n={len(df)})")
    return {"dataset": name, "n": len(df), "ece": ece, "brier_score": brier, "nll": nll, "bins": bins_detail}


def main():
    print("Cross-dataset calibration analysis (RandomForest, all three datasets)\n")
    print("NOTE: uses top-1 predicted-class confidence and binary correctness,")
    print("consistent with how confidence is used throughout the arbitration")
    print("mechanism (a scalar trust/confidence signal, not a full class-")
    print("probability vector) - NOT a full multiclass Brier/NLL over all classes.\n")

    configs = [
        ("NSL-KDD", OUT_DIR / "baseline_predictions.csv"),
        ("UNSW-NB15", OUT_DIR / "unsw_baseline_predictions.csv"),
        ("CICIDS2017", OUT_DIR / "cicids_baseline_predictions.csv"),
    ]

    results = []
    for name, path in configs:
        r = analyze_dataset(name, path)
        if r is not None:
            results.append(r)

    print(f"\n{'='*60}\nSUMMARY TABLE\n{'='*60}")
    print(f"{'Dataset':<14}{'ECE':>10}{'Brier':>10}{'NLL':>10}")
    for r in results:
        print(f"{r['dataset']:<14}{r['ece']:>10.4f}{r['brier_score']:>10.4f}{r['nll']:>10.4f}")

    with open(OUT_DIR / "cross_dataset_calibration.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {OUT_DIR / 'cross_dataset_calibration.json'}")


if __name__ == "__main__":
    main()
