"""
Real measured latency, replacing the hardcoded LATENCY_AUTO_MS / LATENCY_ESCALATE_MS
placeholder constants used earlier for illustrative workload modeling.

What CAN be honestly measured computationally:
  - Classifier (ML agent) inference latency - real wall-clock time
  - QPSO optimization convergence time - real wall-clock time

What CANNOT be measured computationally, and should NOT be fabricated:
  - Human analyst review time. This requires an actual human pilot (see
    human_pilot_study.py). Report it as "not measured in this study" or
    from the pilot data if you run one - never as an invented constant.

Recommendation for the paper: report the two real measurements below as
"system-side computational latency," and either drop escalation-side
latency from the results entirely, or explicitly label any workload
trade-off figure's time axis as "relative cost units, not measured wall-
clock time" if you keep the illustrative model for the Pareto discussion.
"""
from pathlib import Path
import json
import time
import numpy as np
import joblib

from data_loader import load_raw, encode_features
import quantum_optimizer as qo

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = PROJECT_ROOT / "models"
OUT_DIR = PROJECT_ROOT / "outputs"

N_INFERENCE_TRIALS = 1000


def measure_inference_latency():
    clf = joblib.load(MODEL_DIR / "baseline_rf.joblib")
    feature_columns = joblib.load(MODEL_DIR / "feature_columns.joblib")
    train_df = load_raw("train")
    test_df = load_raw("test")
    _, _, test_X, _, _ = encode_features(train_df, test_df)
    test_X = test_X[feature_columns]

    rng = np.random.default_rng(42)
    sample_idx = rng.choice(len(test_X), size=N_INFERENCE_TRIALS, replace=True)

    # Warm up (first call has JIT/cache overhead, exclude from timing)
    _ = clf.predict_proba(test_X.iloc[[0]])

    single_row_times = []
    for idx in sample_idx:
        row = test_X.iloc[[idx]]
        t0 = time.perf_counter()
        clf.predict_proba(row)
        single_row_times.append(time.perf_counter() - t0)
    single_row_times = np.array(single_row_times) * 1000  # to ms

    t0 = time.perf_counter()
    clf.predict_proba(test_X)
    batch_time_ms = (time.perf_counter() - t0) * 1000
    per_row_batch_ms = batch_time_ms / len(test_X)

    return {
        "single_row_inference_ms_mean": float(single_row_times.mean()),
        "single_row_inference_ms_std": float(single_row_times.std()),
        "single_row_inference_ms_p95": float(np.percentile(single_row_times, 95)),
        "full_batch_inference_ms_total": float(batch_time_ms),
        "full_batch_n_rows": int(len(test_X)),
        "per_row_amortized_batch_ms": float(per_row_batch_ms),
        "n_trials": N_INFERENCE_TRIALS,
    }


def measure_qpso_convergence_time():
    preds_path = OUT_DIR / "baseline_predictions.csv"
    if not preds_path.exists():
        return None
    df = qo.load_data()
    times = []
    for seed in range(5):
        t0 = time.perf_counter()
        qo.qpso_optimize(df, n_particles=25, n_iterations=40, seed=seed)
        times.append(time.perf_counter() - t0)
    times = np.array(times)
    return {
        "qpso_convergence_seconds_mean": float(times.mean()),
        "qpso_convergence_seconds_std": float(times.std()),
        "n_particles": 25, "n_iterations": 40, "n_repeats": 5,
        "note": "QPSO runs periodically (policy re-tuning), not per-alert - this is NOT added to per-decision latency.",
    }


def main():
    print("Measuring REAL classifier inference latency...")
    inference_results = measure_inference_latency()
    for k, v in inference_results.items():
        print(f"  {k}: {v}")

    print("\nMeasuring REAL QPSO optimization convergence time (5 repeats)...")
    qpso_results = measure_qpso_convergence_time()
    if qpso_results:
        for k, v in qpso_results.items():
            print(f"  {k}: {v}")

    output = {
        "classifier_inference": inference_results,
        "qpso_convergence": qpso_results,
        "human_review_latency": "NOT MEASURED - requires a real human pilot study (see human_pilot_study.py). "
                                  "Do not report an invented value for this.",
    }
    with open(OUT_DIR / "measured_latency.json", "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {OUT_DIR / 'measured_latency.json'}")


if __name__ == "__main__":
    main()
