"""
Human Pilot Study Tool (v2 - two-phase blind design).

FIX: the original version showed the system's prediction alongside a
category dropdown DEFAULTED to that same prediction. Result: 100% human-
system agreement across every participant - a classic anchoring artifact,
not a real measurement. This version fixes it with a standard two-phase
protocol:

  Phase 1 (BLIND): participant sees only the raw alert features - no
      system prediction, no explanation. They give their own independent
      judgment. This is the real, uncontaminated human-accuracy measurement.
  Phase 2 (REVEALED): the system's prediction and AI Reasoning Summary are
      now shown. The participant gives a final answer (defaulted to THEIR
      OWN Phase 1 answer, not the system's, to avoid the same bug in
      reverse) and rates the explanation's usefulness/clarity.

This also yields a genuinely useful extra metric: how often the
explanation changed the participant's mind (Phase 1 answer != Phase 2
answer) - a concrete measure of the explanation layer's practical value.

Usage:
    streamlit run human_pilot_study.py

Run with 3-5 different people, each using their own Participant ID, on the
SAME fixed alert set (required for inter-rater agreement analysis).
"""
from pathlib import Path
from datetime import datetime
import json

import numpy as np
import pandas as pd
import streamlit as st
import joblib
import shap

from data_loader import load_raw, encode_features
from llm_explainer import build_prompt, call_llm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = PROJECT_ROOT / "models"
OUT_DIR = PROJECT_ROOT / "outputs"
PILOT_ALERTS_PATH = OUT_DIR / "pilot_study_alerts.json"
PILOT_RESULTS_PATH = OUT_DIR / "pilot_study_results.csv"

N_ALERTS = 25
CATEGORIES = ["normal", "dos", "probe", "r2l", "u2r"]
# A small, human-readable subset of raw fields shown in the BLIND phase -
# not the full 41-field vector, but enough to make an informed guess,
# and nothing from the model.
DISPLAY_FIELDS = [
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes",
    "logged_in", "num_failed_logins", "root_shell", "count", "srv_count",
    "serror_rate", "rerror_rate", "same_srv_rate", "diff_srv_rate",
]

st.set_page_config(page_title="Human Pilot Study", layout="centered")
st.title("🧑‍🔬 Human Pilot Study")
st.caption("Two-phase design: your independent judgment first, then the system's assistance.")


@st.cache_resource
def get_model_artifacts():
    clf = joblib.load(MODEL_DIR / "baseline_rf.joblib")
    cat_encoder = joblib.load(MODEL_DIR / "category_encoder.joblib")
    feature_columns = joblib.load(MODEL_DIR / "feature_columns.joblib")
    return clf, cat_encoder, feature_columns


@st.cache_resource
def build_fixed_alert_set():
    if PILOT_ALERTS_PATH.exists():
        try:
            return json.load(open(PILOT_ALERTS_PATH))
        except json.JSONDecodeError:
            st.warning("Found a corrupted alert-set cache file (likely from an interrupted previous run) - "
                       "regenerating it now.")
            PILOT_ALERTS_PATH.unlink()

    clf, cat_encoder, feature_columns = get_model_artifacts()
    train_df = load_raw("train")
    test_df = load_raw("test")
    _, _, test_X, _, _ = encode_features(train_df, test_df)
    test_X = test_X[feature_columns]

    rng = np.random.default_rng(123)
    sample_idx = rng.choice(len(test_X), size=N_ALERTS, replace=False)

    explainer = shap.TreeExplainer(clf)
    alerts = []
    progress_bar = st.progress(0, text="Generating alert set (one-time setup, ~10-15 min - do not close this tab)...")
    for i, idx in enumerate(sample_idx):
        row_encoded = test_X.iloc[idx]
        row_raw = test_df.iloc[idx]
        true_category = row_raw["category"]
        proba = clf.predict_proba(row_encoded.to_frame().T)[0]
        pred_class_idx = proba.argmax()
        pred_category = cat_encoder.classes_[pred_class_idx]
        confidence = float(proba.max())

        sv = explainer.shap_values(row_encoded.to_frame().T)
        sv_row = sv[0, :, pred_class_idx]
        top_idx = np.argsort(np.abs(sv_row))[::-1][:5]
        top_features = [(feature_columns[j], float(row_encoded[feature_columns[j]]), float(sv_row[j])) for j in top_idx]

        prompt = build_prompt(pred_category, confidence, 0.7, top_features)
        explanation = call_llm(prompt)

        display_values = {
            f: (row_raw[f].item() if hasattr(row_raw[f], "item") else row_raw[f])
            for f in DISPLAY_FIELDS if f in row_raw
        }

        alerts.append({
            "row_index": int(idx), "true_category": true_category,
            "pred_category": pred_category, "confidence": confidence,
            "top_features": top_features, "explanation": explanation,
            "display_values": display_values,
        })
        progress_bar.progress((i + 1) / N_ALERTS, text=f"Generating alert set: {i+1}/{N_ALERTS}...")

    progress_bar.empty()

    # Atomic write: build the full file content in a temp file first, then
    # rename it into place in one step. This means the real file can NEVER
    # end up half-written, even if the process is killed mid-write.
    tmp_path = PILOT_ALERTS_PATH.with_suffix(".tmp")
    with open(tmp_path, "w") as f:
        json.dump(alerts, f, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    tmp_path.replace(PILOT_ALERTS_PATH)

    return alerts


def save_response(record):
    df_row = pd.DataFrame([record])
    df_row.to_csv(PILOT_RESULTS_PATH, mode="a", header=not PILOT_RESULTS_PATH.exists(), index=False)


# ---------------------------------------------------------------------------
participant_id = st.text_input("Participant ID (e.g. your initials)")
if not participant_id:
    st.info("Enter a Participant ID to begin.")
    st.stop()

with st.spinner("Preparing alert set (first run only)..."):
    alerts = build_fixed_alert_set()

if "pilot_idx" not in st.session_state:
    st.session_state.pilot_idx = 0
if "pilot_phase" not in st.session_state:
    st.session_state.pilot_phase = 1
if "phase1_answer" not in st.session_state:
    st.session_state.phase1_answer = None

idx = st.session_state.pilot_idx
if idx >= len(alerts):
    st.success(f"🎉 Done! You reviewed all {len(alerts)} alerts. Thank you.")
    st.stop()

alert = alerts[idx]
st.progress(idx / len(alerts), text=f"Alert {idx+1} of {len(alerts)}  —  Phase {st.session_state.pilot_phase} of 2")

# ===========================================================================
# PHASE 1: BLIND - raw features only, no system output shown
# ===========================================================================
if st.session_state.pilot_phase == 1:
    with st.expander("📖 Field glossary — read this first (especially on your first alert)", expanded=(idx == 0)):
        st.markdown("""
**What the fields mean:**

| Field | Meaning |
|---|---|
| `duration` | How long the connection lasted (seconds) |
| `protocol_type` | tcp / udp / icmp |
| `service` | What was accessed (http, ftp, smtp, private, domain_u, etc.) |
| `flag` | How it ended — `SF`=completed normally, `S0`=never got a reply, `REJ`=rejected |
| `src_bytes` / `dst_bytes` | Data sent out / data sent back |
| `logged_in` | 1 = successfully logged in |
| `num_failed_logins` | Number of failed login attempts |
| `root_shell` | 1 = attacker got admin-level access (rare, serious) |
| `count` / `srv_count` | Number of recent connections to this host / this service |
| `serror_rate` / `rerror_rate` | Fraction of recent connections that failed / were rejected (0-1) |
| `same_srv_rate` / `diff_srv_rate` | Fraction of recent connections to the same / to different services |

**Loosely, in plain terms (use your own judgment - these are just orientation, not strict rules):**
- Lots of failed/incomplete connections in a short time can look like an automated flood.
- Touching many different services quickly, without much data actually moving, can look like someone scanning around.
- A failed login, or unusual activity without ever successfully logging in, can look like a break-in attempt.
- Getting admin-level access (`root_shell`=1) is unusual and worth noting.
- If nothing stands out, it's probably ordinary traffic.

There's no single correct way to read these - use your own judgment based on what seems unusual to you. Getting it exactly right isn't the point; your honest, independent read is what we're measuring.
""")

    st.markdown("#### Raw connection record")
    st.caption("No system prediction shown yet — this is your independent judgment.")
    display_df = pd.DataFrame([alert["display_values"]]).T.reset_index()
    display_df.columns = ["field", "value"]
    st.dataframe(display_df, use_container_width=True, hide_index=True)

    st.markdown("#### Your independent assessment")
    phase1_label = st.selectbox(
        "Based on these raw features alone, what category do YOU believe this is?",
        CATEGORIES, index=None, placeholder="Select your answer...", key=f"phase1_{idx}",
    )
    if st.button("Submit Phase 1 →", type="primary", disabled=phase1_label is None):
        st.session_state.phase1_answer = phase1_label
        st.session_state.pilot_phase = 2
        st.rerun()

# ===========================================================================
# PHASE 2: REVEALED - system prediction + explanation now shown
# ===========================================================================
else:
    st.markdown(f"**Your Phase 1 answer:** {st.session_state.phase1_answer}")
    st.markdown("#### System's prediction")
    st.write(f"**Predicted category:** {alert['pred_category']}  |  **Confidence:** {alert['confidence']:.2f}")

    st.markdown("#### AI Reasoning Summary")
    feat_df = pd.DataFrame(alert["top_features"], columns=["feature", "value", "shap_contribution"])
    st.dataframe(feat_df, use_container_width=True, hide_index=True)
    st.info(alert["explanation"])

    st.markdown("---")
    st.markdown("#### Your final assessment")
    default_idx = CATEGORIES.index(st.session_state.phase1_answer)  # defaults to THEIR OWN phase-1 answer, not the system's
    final_label = st.selectbox("Now that you've seen the system's reasoning, what is your FINAL answer?",
                                CATEGORIES, index=default_idx, key=f"phase2_label_{idx}")
    usefulness = st.slider("How USEFUL was the explanation for your final decision? (1=not at all, 5=extremely)",
                            1, 5, 3, key=f"useful_{idx}")
    clarity = st.slider("How CLEAR/understandable was the explanation? (1=confusing, 5=very clear)",
                         1, 5, 3, key=f"clarity_{idx}")

    if st.button("Submit and continue →", type="primary"):
        save_response({
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "participant_id": participant_id,
            "alert_index_in_study": idx,
            "row_index": alert["row_index"],
            "true_category": alert["true_category"],
            "system_prediction": alert["pred_category"],
            "phase1_blind_label": st.session_state.phase1_answer,
            "phase2_final_label": final_label,
            "changed_mind_after_explanation": final_label != st.session_state.phase1_answer,
            "phase1_correct": st.session_state.phase1_answer == alert["true_category"],
            "phase2_correct": final_label == alert["true_category"],
            "system_correct": alert["pred_category"] == alert["true_category"],
            "phase1_agrees_with_system": st.session_state.phase1_answer == alert["pred_category"],
            "phase2_agrees_with_system": final_label == alert["pred_category"],
            "usefulness_rating": usefulness,
            "clarity_rating": clarity,
        })
        st.session_state.pilot_idx += 1
        st.session_state.pilot_phase = 1
        st.session_state.phase1_answer = None
        st.rerun()
