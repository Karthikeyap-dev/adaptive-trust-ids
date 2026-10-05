"""
Quantitative explanation faithfulness evaluation (deletion test).

Standard XAI faithfulness protocol: if SHAP correctly identifies the
features actually driving a prediction, then REMOVING those top-k
features (replacing with the training-set mean) should cause a LARGE drop
in the model's predicted probability for that class. Removing k RANDOM
features should cause a much smaller drop on average, since most features
aren't influential.

If (SHAP-guided drop) is significantly larger than (random drop), that is
quantitative evidence the explanations are faithful to the model's actual
reasoning - not just plausible-sounding text. This is the standard
deletion-test protocol used in the XAI literature.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import joblib
import shap
from scipy import stats

from data_loader import load_raw, encode_features

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = PROJECT_ROOT / "models"
OUT_DIR = PROJECT_ROOT / "outputs"

N_SAMPLES = 200
TOP_K = 5
SEED = 42


def main():
    print("Loading model and data...")
    clf = joblib.load(MODEL_DIR / "baseline_rf.joblib")
    feature_columns = joblib.load(MODEL_DIR / "feature_columns.joblib")

    train_df = load_raw("train")
    test_df = load_raw("test")
    train_X, train_y, test_X, test_y, cat_encoder = encode_features(train_df, test_df)
    test_X = test_X[feature_columns]
    train_X = train_X[feature_columns]

    feature_means = train_X.mean()

    rng = np.random.default_rng(SEED)
    sample_idx = rng.choice(len(test_X), size=N_SAMPLES, replace=False)

    print(f"Computing SHAP values for {N_SAMPLES} sampled predictions...")
    explainer = shap.TreeExplainer(clf)
    sample_X = test_X.iloc[sample_idx]
    shap_values = explainer.shap_values(sample_X)

    shap_drops, random_drops = [], []
    records = []

    for i, idx in enumerate(sample_idx):
        row = test_X.iloc[idx]
        proba = clf.predict_proba(row.to_frame().T)[0]
        pred_class_idx = proba.argmax()
        baseline_prob = proba[pred_class_idx]

        sv_row = shap_values[i, :, pred_class_idx]
        top_k_idx = np.argsort(np.abs(sv_row))[::-1][:TOP_K]
        random_k_idx = rng.choice(len(feature_columns), size=TOP_K, replace=False)

        perturbed_shap = row.copy()
        for j in top_k_idx:
            perturbed_shap.iloc[j] = feature_means.iloc[j]
        prob_after_shap = clf.predict_proba(perturbed_shap.to_frame().T)[0][pred_class_idx]
        drop_shap = baseline_prob - prob_after_shap

        perturbed_random = row.copy()
        for j in random_k_idx:
            perturbed_random.iloc[j] = feature_means.iloc[j]
        prob_after_random = clf.predict_proba(perturbed_random.to_frame().T)[0][pred_class_idx]
        drop_random = baseline_prob - prob_after_random

        shap_drops.append(drop_shap)
        random_drops.append(drop_random)
        records.append({
            "row_index": int(idx), "baseline_prob": float(baseline_prob),
            "prob_after_shap_removal": float(prob_after_shap), "drop_shap": float(drop_shap),
            "prob_after_random_removal": float(prob_after_random), "drop_random": float(drop_random),
        })

        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{N_SAMPLES} processed...")

    shap_drops = np.array(shap_drops)
    random_drops = np.array(random_drops)

    print(f"\nMean confidence drop from removing top-{TOP_K} SHAP features: {shap_drops.mean():.4f} (+/- {shap_drops.std():.4f})")
    print(f"Mean confidence drop from removing {TOP_K} RANDOM features:    {random_drops.mean():.4f} (+/- {random_drops.std():.4f})")

    stat, p = stats.wilcoxon(shap_drops, random_drops)
    print(f"\nWilcoxon signed-rank test (SHAP drop vs random drop): W={stat:.2f}, p={p:.6f}")
    faithful = p < 0.05 and shap_drops.mean() > random_drops.mean()
    print("SHAP-guided removal causes significantly larger drops - explanations are faithful."
          if faithful else "No significant faithfulness signal detected - investigate further.")

    results = {
        "n_samples": N_SAMPLES, "top_k": TOP_K,
        "mean_drop_shap": float(shap_drops.mean()), "std_drop_shap": float(shap_drops.std()),
        "mean_drop_random": float(random_drops.mean()), "std_drop_random": float(random_drops.std()),
        "wilcoxon_statistic": float(stat), "wilcoxon_p_value": float(p),
        "faithful": bool(faithful),
    }
    with open(OUT_DIR / "faithfulness_results.json", "w") as f:
        json.dump(results, f, indent=2)
    pd.DataFrame(records).to_csv(OUT_DIR / "faithfulness_records.csv", index=False)
    print(f"\nSaved results to {OUT_DIR / 'faithfulness_results.json'}")


if __name__ == "__main__":
    main()
