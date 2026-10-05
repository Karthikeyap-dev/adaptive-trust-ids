"""
Real wall-clock latency measurement - NSL-KDD, RandomForest.

Fast script: loads the already-trained baseline model if saved, otherwise
retrains once (untimed), then times real inference + policy-search
convergence directly on your machine. No retraining loop, no multi-seed -
just a clean, single, real measurement to replace the removed placeholder
in Section 5.8.

Run this once, then send me the printed output and I'll put the real,
confirmed numbers back into the manuscript.
"""
from pathlib import Path
import json
import time
import numpy as np
import pandas as pd
import joblib

from data_loader import load_raw, encode_features

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = PROJECT_ROOT / "models"
OUT_DIR = PROJECT_ROOT / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

N_INFERENCE_TRIALS = 1000
N_QPSO_REPEATS = 5

LOWER = np.array([0.55, 0.55, 0.05, 0.05])
UPPER = np.array([0.95, 0.95, 0.65, 0.65])


def evaluate_policy(params, trust_arr, conf_arr, correct_arr):
    trust_high, conf_high, trust_low, conf_low = params
    n = len(trust_arr)
    if trust_low >= trust_high or conf_low >= conf_high:
        return -10.0
    auto_mask = (trust_arr >= trust_high) & (conf_arr >= conf_high)
    reject_mask = (~auto_mask) & ((trust_arr <= trust_low) | (conf_arr <= conf_low))
    escalate_mask = ~auto_mask & ~reject_mask
    auto_count = auto_mask.sum()
    silent_fail = (auto_mask & (correct_arr == 0)).sum()
    silent_rate = silent_fail / max(auto_count, 1)
    escalation_rate = (reject_mask.sum() + escalate_mask.sum()) / n
    violation = max(0.0, silent_rate - 0.05)
    return (-5.0 - violation) if violation > 0 else (0.5 * (1 - silent_rate) + 0.5 * (1 - escalation_rate))


def qpso_step(trust_arr, conf_arr, correct_arr, n_particles=25, n_iterations=40, seed=42):
    rng = np.random.default_rng(seed)
    dim = 4
    swarm = rng.uniform(LOWER, UPPER, size=(n_particles, dim))
    pbest = swarm.copy()
    pbest_fitness = np.array([evaluate_policy(p, trust_arr, conf_arr, correct_arr) for p in swarm])
    gbest = pbest[np.argmax(pbest_fitness)].copy()
    gbest_fitness = pbest_fitness.max()
    for it in range(n_iterations):
        beta = 1.0 - 0.6 * (it / n_iterations)
        mbest = pbest.mean(axis=0)
        for i in range(n_particles):
            phi = rng.uniform(0, 1, size=dim)
            attractor = phi * pbest[i] + (1 - phi) * gbest
            u = rng.uniform(1e-6, 1.0, size=dim)
            sign = rng.choice([-1, 1], size=dim)
            swarm[i] = np.clip(attractor + sign * beta * np.abs(mbest - swarm[i]) * np.log(1.0 / u), LOWER, UPPER)
            fit = evaluate_policy(swarm[i], trust_arr, conf_arr, correct_arr)
            if fit > pbest_fitness[i]:
                pbest[i], pbest_fitness[i] = swarm[i].copy(), fit
        if pbest_fitness.max() > gbest_fitness:
            gbest, gbest_fitness = pbest[np.argmax(pbest_fitness)].copy(), pbest_fitness.max()
    return gbest


def main():
    print("Loading NSL-KDD and the RandomForest baseline model...")
    train_df = load_raw("train")
    test_df = load_raw("test")
    train_X, train_y, test_X, test_y, cat_encoder = encode_features(train_df, test_df)

    model_path = MODEL_DIR / "baseline_rf.joblib"
    if model_path.exists():
        clf = joblib.load(model_path)
        print(f"Loaded existing model from {model_path}")
    else:
        from sklearn.ensemble import RandomForestClassifier
        print("No saved model found - training once (not timed)...")
        clf = RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=42, class_weight="balanced_subsample")
        clf.fit(train_X, train_y)

    print(f"\nMeasuring single-row inference latency ({N_INFERENCE_TRIALS} trials)...")
    single_row = test_X.iloc[[0]]
    for _ in range(10):
        clf.predict_proba(single_row)
    times = []
    for _ in range(N_INFERENCE_TRIALS):
        t0 = time.perf_counter()
        clf.predict_proba(single_row)
        times.append((time.perf_counter() - t0) * 1000)
    times = np.array(times)

    print(f"Measuring batch-amortized inference latency (full test set, {len(test_X)} rows)...")
    t0 = time.perf_counter()
    clf.predict_proba(test_X)
    batch_total_ms = (time.perf_counter() - t0) * 1000

    print(f"\nMeasuring policy-search convergence time ({N_QPSO_REPEATS} repeats)...")
    pred_proba = clf.predict_proba(test_X)
    pred_y = pred_proba.argmax(axis=1)
    conf = pred_proba.max(axis=1)
    correct = (pred_y == test_y).astype(int)
    trust_arr = np.full(len(test_X), 0.7)

    qpso_times = []
    for r in range(N_QPSO_REPEATS):
        t0 = time.perf_counter()
        qpso_step(trust_arr, conf, correct, seed=r)
        qpso_times.append(time.perf_counter() - t0)
    qpso_times = np.array(qpso_times)

    results = {
        "classifier_inference": {
            "single_row_inference_ms_mean": float(times.mean()),
            "single_row_inference_ms_std": float(times.std()),
            "single_row_inference_ms_p95": float(np.percentile(times, 95)),
            "full_batch_inference_ms_total": float(batch_total_ms),
            "full_batch_n_rows": int(len(test_X)),
            "per_row_amortized_batch_ms": float(batch_total_ms / len(test_X)),
            "n_trials": N_INFERENCE_TRIALS,
        },
        "policy_search_convergence": {
            "convergence_seconds_mean": float(qpso_times.mean()),
            "convergence_seconds_std": float(qpso_times.std()),
            "n_particles": 25,
            "n_iterations": 40,
            "n_repeats": N_QPSO_REPEATS,
            "note": "Policy search runs periodically (re-tuning), not per-alert - not added to per-decision latency.",
        },
        "measured_on": "REPLACE WITH YOUR MACHINE SPEC",
    }

    print(f"\n{'='*60}\nRESULTS\n{'='*60}")
    print(f"Single-row inference: {times.mean():.2f}ms +/- {times.std():.2f}ms (p95={np.percentile(times,95):.2f}ms)")
    print(f"Batch-amortized: {batch_total_ms/len(test_X):.5f}ms/row")
    print(f"Policy-search convergence: {qpso_times.mean():.3f}s +/- {qpso_times.std():.3f}s")

    with open(OUT_DIR / "measured_latency_confirmed.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {OUT_DIR / 'measured_latency_confirmed.json'}")
    print("Please fill in 'measured_on' with your actual machine spec before using this in the paper.")


if __name__ == "__main__":
    main()
