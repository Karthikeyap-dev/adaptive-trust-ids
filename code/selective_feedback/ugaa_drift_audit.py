"""
Instrumented re-run of the UGAA drift test (Table 14).

Uses the SAME protocol, seeds (10-19), target selection, thresholds, k=4 and
RNG call order as ugaa_experiment.drift_test, so coverage reproduces the
saved results exactly. Adds what reviewers will ask for:

  1. Realized audit spend DURING the drift window (audited auto-executed
     alerts / auto-executed alerts, all categories) - the expenditure that
     actually produced the coverage, not the steady-state average.
  2. Harm metric: drift events auto-executed without review (forced-incorrect
     target alerts that became silent failures) and events until the target
     category is first escalated (containment delay).
  3. Equal-spend control: fixed-rate audit re-run per seed at UGAA's realized
     drift-window rate, so UGAA vs fixed is compared at identical expenditure.
  4. Paired Wilcoxon tests across seeds (same shuffle and window per seed).
  5. Detection delay with never-detected seeds counted as 150 (censored),
     instead of averaging over detected seeds only.

Usage: python3 ugaa_drift_audit.py <predictions_csv> [output_prefix]
"""
from pathlib import Path
import json, sys
import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ugaa_experiment import (TrustEngine, UncertaintyGuidedAuditor, arbitration_decision,
                             DECAY, DRIFT_BURST_LEN, TARGET_AUDIT_RATE, SEED_OFFSET, N_SEEDS)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = PROJECT_ROOT / "outputs"


def drift_test_instrumented(preds, seed, method, audit_rate=TARGET_AUDIT_RATE,
                            feedback_error_rate=0.08, pre_drift_n=1500):
    rng = np.random.default_rng(seed)
    shuffled = preds.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    # --- target selection: identical to ugaa_experiment.drift_test ---
    cat_counts = shuffled["pred_category"].value_counts()
    eligible = cat_counts[cat_counts >= 20].index.tolist() or cat_counts.index.tolist()
    cat_acc = shuffled[shuffled["pred_category"].isin(eligible)].groupby("pred_category")["correct"].mean()
    target = next((c for c in cat_acc.sort_values(ascending=False).index
                   if cat_counts[c] >= int(DRIFT_BURST_LEN * 1.5)), None)
    if target is None:
        return None

    engine = TrustEngine(decay=DECAY)
    auditor = UncertaintyGuidedAuditor(target_rate=audit_rate) if method == "ugaa" else None
    n = len(shuffled)
    pre_drift_n = min(pre_drift_n, max(n - DRIFT_BURST_LEN, 100))
    pre, post_pool = shuffled.iloc[:pre_drift_n], shuffled.iloc[pre_drift_n:]
    prev = (post_pool["pred_category"] == target).mean()
    if prev <= 0:
        return None
    window = min(int(np.ceil(DRIFT_BURST_LEN / prev * 1.8)), len(post_pool))
    drift_window = post_pool.sample(n=window, replace=(window > len(post_pool)),
                                    random_state=seed + 1).reset_index(drop=True)

    c = dict(auto=0, audited=0, tgt_auto=0, tgt_audited=0, tgt_esc_fb=0)

    def step(row, force_incorrect, count):
        cat, conf = row["pred_category"], row["raw_confidence"]
        is_correct = False if force_incorrect else bool(row["correct"])
        decision = arbitration_decision(engine.trust_for(cat), conf)
        if method == "escalation_only":
            fb = decision == "escalate"
        elif method == "fixed_audit":
            fb = (decision == "escalate") or (rng.random() < audit_rate)
        else:
            p = auditor.audit_probability(engine, cat)
            fb = (decision == "escalate") or (rng.random() < p)
        if count:
            if decision == "auto":
                c["auto"] += 1
                c["audited"] += int(fb)
            if force_incorrect:
                c["tgt_auto"] += int(decision == "auto")
                c["tgt_audited"] += int(decision == "auto" and fb)
                c["tgt_esc_fb"] += int(decision == "escalate")
        if fb:
            observed = is_correct if rng.random() > feedback_error_rate else (not is_correct)
            engine.update(cat, observed)
        return fb, decision

    for _, row in pre.iterrows():
        step(row, False, False)
    pre_trust = engine.trust_for(target)

    seen, detected, first_det, first_esc = 0, 0, None, None
    for _, row in drift_window.iterrows():
        if row["pred_category"] == target:
            if seen >= DRIFT_BURST_LEN:
                continue
            fb, decision = step(row, True, True)
            seen += 1
            if fb:
                detected += 1
                first_det = seen - 1 if first_det is None else first_det
            if decision == "escalate" and first_esc is None:
                first_esc = seen - 1
        else:
            step(row, False, True)
        if seen >= DRIFT_BURST_LEN:
            break
    if seen < DRIFT_BURST_LEN:
        return None

    return {
        "target_category": target, "pre_drift_trust": float(pre_trust),
        "post_drift_trust": float(engine.trust_for(target)),
        "feedback_coverage": detected / DRIFT_BURST_LEN,
        "first_detection_event": first_det,
        "detection_delay_censored": first_det if first_det is not None else DRIFT_BURST_LEN,
        "containment_delay_censored": first_esc if first_esc is not None else DRIFT_BURST_LEN,
        "drift_silent_failures": c["tgt_auto"],
        "target_audits": c["tgt_audited"], "target_escalation_feedback": c["tgt_esc_fb"],
        "window_auto_executed": c["auto"], "window_audited": c["audited"],
        "window_realized_audit_rate": c["audited"] / max(c["auto"], 1),
    }


METRICS = [("feedback_coverage", "higher"), ("detection_delay_censored", "lower"),
           ("containment_delay_censored", "lower"), ("drift_silent_failures", "lower"),
           ("window_realized_audit_rate", "-")]


def paired(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if np.allclose(a, b):
        return None
    return float(stats.wilcoxon(a, b).pvalue)


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    path = Path(sys.argv[1])
    prefix = sys.argv[2] if len(sys.argv) > 2 else path.stem
    preds = pd.read_csv(path)
    seeds = range(SEED_OFFSET, SEED_OFFSET + N_SEEDS)
    print(f"Loaded {len(preds)} predictions; locked seeds {seeds.start}-{seeds.stop - 1}")

    res = {}
    for m in ["escalation_only", "fixed_audit", "ugaa"]:
        res[m] = {s: drift_test_instrumented(preds, s, m) for s in seeds}
    # Equal-spend control: fixed audit at UGAA's realized drift-window rate, per seed
    res["fixed_matched"] = {s: (drift_test_instrumented(preds, s, "fixed_audit",
                                audit_rate=res["ugaa"][s]["window_realized_audit_rate"])
                                if res["ugaa"][s] else None) for s in seeds}
    ok = [s for s in seeds if all(res[m][s] is not None for m in res)]
    target = res["ugaa"][ok[0]]["target_category"]

    table = {}
    print(f"\nTarget category: {target}   seeds used: {len(ok)}/{N_SEEDS}")
    print(f"{'metric':<30}{'esc-only':>10}{'fixed 5%':>10}{'UGAA':>10}{'fixed@UGAA':>12}"
          f"{'p(U vs 5%)':>12}{'p(U vs @U)':>12}")
    for key, _ in METRICS:
        vals = {m: [res[m][s][key] for s in ok] for m in res}
        table[key] = {m: {"mean": float(np.mean(v)), "sd": float(np.std(v, ddof=1))} for m, v in vals.items()}
        p1, p2 = paired(vals["ugaa"], vals["fixed_audit"]), paired(vals["ugaa"], vals["fixed_matched"])
        table[key]["p_ugaa_vs_fixed5"], table[key]["p_ugaa_vs_fixed_matched"] = p1, p2
        scale = 100 if key in ("feedback_coverage", "window_realized_audit_rate") else 1
        fmt = lambda m: f"{np.mean(vals[m]) * scale:.1f}"
        pf = lambda p: "—" if p is None else f"{p:.4f}"
        print(f"{key:<30}{fmt('escalation_only'):>10}{fmt('fixed_audit'):>10}{fmt('ugaa'):>10}"
              f"{fmt('fixed_matched'):>12}{pf(p1):>12}{pf(p2):>12}")

    with open(OUT_DIR / f"{prefix}_ugaa_drift_audit.json", "w") as f:
        json.dump({"prefix": prefix, "target_category": target, "seeds": list(ok), "summary": table,
                   "per_seed": {m: {str(s): res[m][s] for s in ok} for m in res}}, f, indent=2, default=float)
    print(f"\nSaved {prefix}_ugaa_drift_audit.json")


if __name__ == "__main__":
    main()
