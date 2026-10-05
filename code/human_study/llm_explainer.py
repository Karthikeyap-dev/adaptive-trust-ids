"""
Milestone 2: LLM/SLM explanation layer.

Two-stage design:
  1. SHAP (TreeExplainer) finds WHICH features actually drove the
     prediction - deterministic, auditable, classical XAI.
  2. A local LLM (via Ollama) turns that feature attribution into a
     short, human-readable rationale for a SOC analyst.

Only escalated/rejected decisions get an explanation generated - auto-
executed decisions aren't shown to a human, so there's no one to explain
them to, and this keeps LLM calls cheap (tens, not tens of thousands).

Requires Ollama running locally with a model pulled, e.g.:
    ollama pull llama3.2
    ollama serve   (usually starts automatically after install)

To use the Anthropic API instead, set ANTHROPIC_API_KEY and change
USE_ANTHROPIC_API to True below.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import joblib
import shap
import requests

from data_loader import load_raw, encode_features

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = PROJECT_ROOT / "models"
OUT_DIR = PROJECT_ROOT / "outputs"

N_SAMPLES = 15          # how many escalated cases to explain (demo-scale, not the full test set)
TOP_K_FEATURES = 5      # how many top contributing features to surface per explanation
SEED = 42

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3.2"  # change to whatever you've pulled, e.g. "qwen2.5:3b", "phi3"

USE_ANTHROPIC_API = False  # set True + set ANTHROPIC_API_KEY env var to use the API instead of Ollama
ANTHROPIC_MODEL = "claude-sonnet-4-6"


def call_ollama(prompt):
    try:
        resp = requests.post(
            OLLAMA_URL,
            json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False},
            timeout=60,
        )
        resp.raise_for_status()
        return resp.json()["response"].strip()
    except requests.exceptions.ConnectionError:
        return (
            "[ERROR: could not reach Ollama at localhost:11434. "
            "Make sure Ollama is installed and running, and you've pulled a model:\n"
            f"  ollama pull {OLLAMA_MODEL}\n"
            "  ollama serve\n"
            "then re-run this script.]"
        )
    except Exception as e:
        return f"[ERROR calling Ollama: {e}]"


def call_anthropic(prompt):
    import os
    try:
        import anthropic
    except ImportError:
        return "[ERROR: run 'pip install anthropic' to use the Anthropic API path.]"
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return "[ERROR: set the ANTHROPIC_API_KEY environment variable to use the API path.]"
    client = anthropic.Anthropic(api_key=api_key)
    msg = client.messages.create(
        model=ANTHROPIC_MODEL,
        max_tokens=300,
        messages=[{"role": "user", "content": prompt}],
    )
    return msg.content[0].text.strip()


def call_llm(prompt):
    return call_anthropic(prompt) if USE_ANTHROPIC_API else call_ollama(prompt)


def build_prompt(pred_category, confidence, trust, top_features):
    feature_lines = "\n".join(
        f"  - {name} = {value:.4g}  (SHAP contribution: {'+' if contrib > 0 else ''}{contrib:.4f})"
        for name, value, contrib in top_features
    )
    return f"""You are assisting a SOC (Security Operations Center) analyst who is reviewing a network traffic alert flagged by an automated intrusion detection system.

The system's prediction: "{pred_category}"

There are TWO SEPARATE numbers below. Do not combine them, average them, or restate one as if it were the other:
  - MODEL CONFIDENCE for this specific alert: {confidence:.2f} (how sure the model is about THIS prediction)
  - ADAPTIVE TRUST SCORE for the "{pred_category}" category overall: {trust:.2f} (how reliable this category's predictions have HISTORICALLY been, based on past human feedback - a property of the category, not of this one alert)

The top features that most influenced this prediction (from SHAP feature attribution):
{feature_lines}

Write a short, plain-language explanation (3-4 sentences) for the analyst covering:
1. Why the system flagged this traffic as "{pred_category}", referencing the feature evidence above. Describe each feature's contribution independently - do NOT invent a ratio, fraction, or relationship between two different features unless that relationship is explicitly given to you above.
2. State the model confidence and the trust score as two distinct numbers, and explain what each means separately. Then say whether the analyst should lean on this prediction or scrutinize it carefully.
Keep it concise and operational - this analyst is busy and needs to make a fast, informed decision, not read a report."""


def main():
    print("Loading model, data, and arbitration decisions...")
    clf = joblib.load(MODEL_DIR / "baseline_rf.joblib")
    cat_encoder = joblib.load(MODEL_DIR / "category_encoder.joblib")
    feature_columns = joblib.load(MODEL_DIR / "feature_columns.joblib")

    train_df = load_raw("train")
    test_df = load_raw("test")
    _, _, test_X, _, _ = encode_features(train_df, test_df)
    test_X = test_X[feature_columns]  # ensure exact column order matches the trained model

    decisions_path = OUT_DIR / "arbitration_decisions.csv"
    if not decisions_path.exists():
        raise FileNotFoundError(f"Run decision_arbitration.py first - missing {decisions_path}")
    decisions = pd.read_csv(decisions_path)

    escalated_mask = decisions["decision"].isin(["escalate_review", "reject_or_escalate"])
    escalated_indices = decisions[escalated_mask].index.tolist()

    rng = np.random.default_rng(SEED)
    sample_indices = rng.choice(escalated_indices, size=min(N_SAMPLES, len(escalated_indices)), replace=False)

    print(f"Computing SHAP values for {len(sample_indices)} sampled escalated cases...")
    explainer = shap.TreeExplainer(clf)
    sample_X = test_X.iloc[sample_indices]
    shap_values = explainer.shap_values(sample_X)  # shape: (n_samples, n_features, n_classes)

    results = []
    for i, idx in enumerate(sample_indices):
        row = decisions.iloc[idx]
        pred_category = row["pred_category"]
        pred_class_idx = list(cat_encoder.classes_).index(pred_category)

        sv_for_row = shap_values[i, :, pred_class_idx]
        top_feature_idx = np.argsort(np.abs(sv_for_row))[::-1][:TOP_K_FEATURES]
        top_features = [
            (feature_columns[j], test_X.iloc[idx][feature_columns[j]], sv_for_row[j])
            for j in top_feature_idx
        ]

        prompt = build_prompt(pred_category, row["confidence"], row["trust"], top_features)
        print(f"\n[{i+1}/{len(sample_indices)}] Generating explanation for row {idx} "
              f"(predicted={pred_category}, true={row['true_category']})...")
        explanation = call_llm(prompt)

        results.append({
            "row_index": int(idx),
            "true_category": row["true_category"],
            "pred_category": pred_category,
            "correct": bool(row["correct"]),
            "confidence": float(row["confidence"]),
            "trust": float(row["trust"]),
            "decision": row["decision"],
            "top_features": [{"feature": n, "value": float(v), "shap_contribution": float(c)} for n, v, c in top_features],
            "llm_explanation": explanation,
        })
        print(explanation)

    with open(OUT_DIR / "llm_explanations.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved {len(results)} explanations to {OUT_DIR / 'llm_explanations.json'}")


if __name__ == "__main__":
    main()
