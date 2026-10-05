"""
Milestone 8: Full end-to-end integrated pipeline.

Ties together every component built so far into ONE script:
  raw traffic features -> baseline classifier (prediction + confidence)
    -> Adaptive Trust Engine lookup (per-category trust)
    -> Decision arbitration (using QPSO-tuned thresholds, chosen by risk ceiling)
    -> [if escalated] SHAP + LLM explanation for the human analyst

Two modes:
  --mode demo   : streams N randomly sampled alerts one at a time with a
                  readable, presentation-friendly log (use this for showing
                  your PI how the whole framework actually behaves).
  --mode full   : runs the entire NSL-KDD test set through the pipeline and
                  reports aggregate metrics, to confirm this integrated
                  version reproduces the same numbers as the individual
                  milestone scripts (sanity-check that integration didn't
                  silently change any behavior).

Usage:
  python3 run_pipeline.py --mode demo --n 10 --risk_ceiling 0.05
  python3 run_pipeline.py --mode demo --n 5  --risk_ceiling 0.10 --explain
  python3 run_pipeline.py --mode full
"""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from data_loader import load_raw, encode_features

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = PROJECT_ROOT / "models"
OUT_DIR = PROJECT_ROOT / "outputs"

LATENCY_AUTO_MS = 5
LATENCY_ESCALATE_MS = 4000

# Fallback thresholds if no pareto_sweep.json / qpso_results.json exist yet
DEFAULT_PARAMS = {"trust_high": 0.75, "conf_high": 0.80, "trust_low": 0.60, "conf_low": 0.50, "w_trust": 0.6}


def load_artifacts():
    """Load every saved artifact the pipeline depends on. Fails loudly with a
    clear message pointing to the milestone script that produces it, if missing."""
    required = {
        MODEL_DIR / "baseline_rf.joblib": "train_baseline.py",
        MODEL_DIR / "category_encoder.joblib": "train_baseline.py",
        MODEL_DIR / "feature_columns.joblib": "train_baseline.py",
        OUT_DIR / "trust_snapshot.json": "simulate_feedback.py",
    }
    for path, script in required.items():
        if not path.exists():
            raise FileNotFoundError(f"Missing {path} - run {script} first.")

    clf = joblib.load(MODEL_DIR / "baseline_rf.joblib")
    cat_encoder = joblib.load(MODEL_DIR / "category_encoder.joblib")
    feature_columns = joblib.load(MODEL_DIR / "feature_columns.joblib")
    with open(OUT_DIR / "trust_snapshot.json") as f:
        snapshot = json.load(f)
    trust_by_category = {row["category"]: row["trust"] for row in snapshot}
    return clf, cat_encoder, feature_columns, trust_by_category


def load_arbitration_params(risk_ceiling):
    """Pick arbitration parameters closest to the requested risk ceiling from
    the Pareto sweep, if available; otherwise fall back to the single QPSO
    result, then to hardcoded defaults. This is the 'risk tolerance knob' a
    real SOC operator would use."""
    sweep_path = OUT_DIR / "pareto_sweep.json"
    if sweep_path.exists():
        with open(sweep_path) as f:
            sweep = json.load(f)
        closest = min(sweep, key=lambda r: abs(r["ceiling"] - risk_ceiling))
        print(f"[params] Using Pareto sweep entry for ceiling={closest['ceiling']:.0%} "
              f"(closest match to requested {risk_ceiling:.0%})")
        return closest["params"]

    qpso_path = OUT_DIR / "qpso_results.json"
    if qpso_path.exists():
        with open(qpso_path) as f:
            qpso = json.load(f)
        print("[params] pareto_sweep.json not found, using single qpso_results.json instead")
        return qpso["qpso_params"]

    print("[params] No optimizer results found, using hardcoded default thresholds")
    return DEFAULT_PARAMS


def arbitrate(trust, confidence, params):
    trust_high, conf_high = params["trust_high"], params["conf_high"]
    trust_low, conf_low = params["trust_low"], params["conf_low"]
    if trust >= trust_high and confidence >= conf_high:
        return "auto_execute"
    if trust <= trust_low or confidence <= conf_low:
        return "reject_or_escalate"
    return "escalate_review"


def explain_decision(row_features, pred_category, confidence, trust, clf, feature_columns):
    """Generate a SHAP + LLM explanation - only called for escalated demo rows."""
    import shap
    from llm_explainer import build_prompt, call_llm

    explainer = shap.TreeExplainer(clf)
    sv = explainer.shap_values(row_features.to_frame().T)  # shape (1, n_features, n_classes)
    cat_list = list(joblib.load(MODEL_DIR / "category_encoder.joblib").classes_)
    pred_class_idx = cat_list.index(pred_category)
    sv_for_row = sv[0, :, pred_class_idx]
    top_idx = np.argsort(np.abs(sv_for_row))[::-1][:5]
    top_features = [(feature_columns[j], row_features[feature_columns[j]], sv_for_row[j]) for j in top_idx]

    prompt = build_prompt(pred_category, confidence, trust, top_features)
    return call_llm(prompt)


def run_demo(n, risk_ceiling, do_explain, seed=7):
    clf, cat_encoder, feature_columns, trust_by_category = load_artifacts()
    params = load_arbitration_params(risk_ceiling)

    train_df = load_raw("train")
    test_df = load_raw("test")
    _, _, test_X, _, _ = encode_features(train_df, test_df)
    test_X = test_X[feature_columns]

    rng = np.random.default_rng(seed)
    sample_idx = rng.choice(len(test_X), size=n, replace=False)

    print(f"\n{'='*70}\nADAPTIVE TRUST-AWARE DECISION PIPELINE - LIVE DEMO ({n} alerts)\n"
          f"Risk ceiling: {risk_ceiling:.0%}   Arbitration params: {params}\n{'='*70}\n")

    log_rows = []
    for i, idx in enumerate(sample_idx):
        row_features = test_X.iloc[idx]
        true_category = test_df.iloc[idx]["category"]

        proba = clf.predict_proba(row_features.to_frame().T)[0]
        pred_class_idx = proba.argmax()
        pred_category = cat_encoder.classes_[pred_class_idx]
        confidence = proba.max()
        trust = trust_by_category.get(pred_category, 0.5)

        decision = arbitrate(trust, confidence, params)
        correct = (pred_category == true_category)

        print(f"--- Alert {i+1}/{n} ---")
        print(f"  Prediction: {pred_category}   Confidence: {confidence:.3f}   "
              f"Trust ({pred_category}): {trust:.3f}")
        print(f"  Decision: {decision.upper()}")
        if not correct:
            print(f"  [note: true category was actually '{true_category}' - shown for demo verification only, "
                  f"the pipeline does not see this]")

        explanation = None
        if do_explain and decision in ("escalate_review", "reject_or_escalate"):
            print("  Generating explanation for human analyst...")
            explanation = explain_decision(row_features, pred_category, confidence, trust, clf, feature_columns)
            print(f"  Explanation: {explanation}")
        print()

        log_rows.append({
            "true_category": true_category, "pred_category": pred_category,
            "confidence": float(confidence), "trust": float(trust),
            "decision": decision, "correct": bool(correct), "explanation": explanation,
        })

    pd.DataFrame(log_rows).to_csv(OUT_DIR / "pipeline_demo_log.csv", index=False)
    print(f"Saved demo log to {OUT_DIR / 'pipeline_demo_log.csv'}")


def run_full(risk_ceiling):
    clf, cat_encoder, feature_columns, trust_by_category = load_artifacts()
    params = load_arbitration_params(risk_ceiling)

    preds_path = OUT_DIR / "baseline_predictions.csv"
    if not preds_path.exists():
        raise FileNotFoundError("Run train_baseline.py first.")
    preds = pd.read_csv(preds_path)

    trust = preds["pred_category"].map(trust_by_category).fillna(0.5).values
    confidence = preds["raw_confidence"].values
    correct = preds["correct"].values
    n = len(preds)

    decisions = np.array([arbitrate(t, c, params) for t, c in zip(trust, confidence)])
    auto_mask = decisions == "auto_execute"
    non_auto_mask = ~auto_mask

    auto_count = auto_mask.sum()
    silent_failures = (auto_mask & (correct == 0)).sum()
    silent_rate = silent_failures / max(auto_count, 1)
    escalation_rate = non_auto_mask.sum() / n
    avg_latency = (auto_count * LATENCY_AUTO_MS + non_auto_mask.sum() * LATENCY_ESCALATE_MS) / n

    print(f"\n{'='*70}\nFULL PIPELINE VALIDATION RUN (entire NSL-KDD test set, n={n})\n{'='*70}")
    print(f"Arbitration params used: {params}\n")
    print(f"Auto-executed:        {auto_count} ({auto_count/n:.1%})")
    print(f"Escalated/rejected:    {non_auto_mask.sum()} ({non_auto_mask.sum()/n:.1%})")
    print(f"Silent failure rate:   {silent_rate:.2%}")
    print(f"Escalation rate:       {escalation_rate:.2%}")
    print(f"Avg latency:           {avg_latency:.1f} ms")
    print(f"\n(Compare these to decision_arbitration.py / quantum_optimizer.py output - "
          f"they should match, confirming the integrated pipeline behaves identically "
          f"to the individually-tested components.)")


def main():
    parser = argparse.ArgumentParser(description="Adaptive Trust-Aware Decision Pipeline")
    parser.add_argument("--mode", choices=["demo", "full"], default="demo")
    parser.add_argument("--n", type=int, default=10, help="Number of alerts to sample in demo mode")
    parser.add_argument("--risk_ceiling", type=float, default=0.05,
                         help="Target silent-failure ceiling, e.g. 0.05 = 5%% - picks matching arbitration params")
    parser.add_argument("--explain", action="store_true", help="Generate LLM explanations for escalated demo alerts")
    args = parser.parse_args()

    if args.mode == "demo":
        run_demo(args.n, args.risk_ceiling, args.explain)
    else:
        run_full(args.risk_ceiling)


if __name__ == "__main__":
    main()
