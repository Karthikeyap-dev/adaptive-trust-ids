"""
Instrumented re-run of additional_drift_scenarios.py (the 19/6/5 record).

Same scenarios, target selection, thresholds, k=4, stream construction and
RNG call order as the original, so coverage on seeds 0-9 reproduces the
saved results exactly. Adds:

  1. Realized audit spend during the drift phase (audited auto-executed /
     auto-executed, all categories).
  2. Harm metric: target-category drift events that were auto-executed AND
     wrong (silent failures caused by the drift).
  3. Equal-spend control: fixed audit re-run per seed at UGAA's realized
     drift-phase spend.
  4. Locked seeds 10-19 (the original used 0-9, the same seeds that selected
     k=4). Seeds 0-9 are also run, only to verify reproduction.
  5. Paired Wilcoxon per comparison; win/tie/loss counted only where the
     difference is significant (p < 0.05), ties otherwise.

Usage: python3 drift_scenarios_audit.py <predictions_csv> [output_prefix]
"""
from pathlib import Path
import json, sys
import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from additional_drift_scenarios import (TrustEngine, UncertaintyGuidedAuditor, arbitration_decision,
                                        build_pre_drift_state, DECAY, DRIFT_BURST_LEN, AUDIT_RATE)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = PROJECT_ROOT / "outputs"
VERIFY_SEEDS = range(0, 10)
LOCKED_SEEDS = range(10, 20)


class Counter:
    def __init__(self):
        self.auto = self.audited = 0
        self.tgt = {}        # category -> dict(detected, silent)

    def rate(self):
        return self.audited / max(self.auto, 1)


def process(engine, auditor, method, stream, rng, audit_rate, force=None, targets=(), ctr=None,
            noisy=True, forced_idx=None):
    """Mirror of additional_drift_scenarios.process_stream with counters.
    force: reliability in [0,1] for target categories (scenarios A/B).
    forced_idx: set of row ids forced incorrect (scenario C)."""
    first = None
    for i, item in enumerate(stream):
        idx, row = item if forced_idx is not None else (None, item)
        cat, conf = row["pred_category"], row["raw_confidence"]
        if forced_idx is not None:
            is_forced = idx in forced_idx
            is_correct = False if is_forced else bool(row["correct"])
        elif force is not None and cat in targets:
            is_forced = True
            is_correct = rng.random() < force
        else:
            is_forced = False
            is_correct = bool(row["correct"])
        decision = arbitration_decision(engine.trust_for(cat), conf)
        if method == "fixed_audit":
            fb = (decision == "escalate") or (rng.random() < audit_rate)
        else:
            p = auditor.audit_probability(engine, cat)
            fb = (decision == "escalate") or (rng.random() < p)
        if ctr is not None:
            if decision == "auto":
                ctr.auto += 1
                ctr.audited += int(fb)
            if is_forced:
                t = ctr.tgt.setdefault(cat, {"detected": 0, "silent": 0})
                t["silent"] += int(decision == "auto" and not is_correct)
                t["detected"] += int(fb)
        if fb:
            if noisy:
                observed = is_correct if rng.random() > 0.08 else (not is_correct)
            else:
                observed = is_correct
            engine.update(cat, observed)
            if is_forced and first is None:
                first = i
    return first


def _new(method, audit_rate):
    return TrustEngine(decay=DECAY), (UncertaintyGuidedAuditor() if method == "ugaa" else None)


def scenario_partial(preds, seed, method, audit_rate, decline_to):
    setup = build_pre_drift_state(preds, seed)
    if setup is None:
        return None
    target, pre, post_pool = setup
    rng = np.random.default_rng(seed)
    engine, auditor = _new(method, audit_rate)
    process(engine, auditor, method, pre, rng, audit_rate)
    prev = (post_pool["pred_category"] == target).mean()
    if prev <= 0:
        return None
    w = min(int(np.ceil(DRIFT_BURST_LEN / prev * 1.8)), len(post_pool))
    win = post_pool.sample(n=w, replace=(w > len(post_pool)), random_state=seed + 1).to_dict("records")
    drift = [r for r in win if r["pred_category"] == target][:DRIFT_BURST_LEN]
    other = [r for r in win if r["pred_category"] != target]
    if len(drift) < DRIFT_BURST_LEN:
        return None
    inter = drift + other
    rng.shuffle(inter)
    ctr = Counter()
    process(engine, auditor, method, inter, rng, audit_rate, force=decline_to, targets=(target,), ctr=ctr)
    t = ctr.tgt.get(target, {"detected": 0, "silent": 0})
    return {"coverage": t["detected"] / DRIFT_BURST_LEN, "silent": t["silent"], "spend": ctr.rate()}


def scenario_gradual(preds, seed, method, audit_rate, levels=(0.95, 0.90, 0.80, 0.70, 0.60)):
    setup = build_pre_drift_state(preds, seed)
    if setup is None:
        return None
    target, pre, post_pool = setup
    rng = np.random.default_rng(seed)
    engine, auditor = _new(method, audit_rate)
    process(engine, auditor, method, pre, rng, audit_rate)
    prev = (post_pool["pred_category"] == target).mean()
    if prev <= 0:
        return None
    per = DRIFT_BURST_LEN // len(levels)
    need = per * len(levels)
    w = min(int(np.ceil(need / prev * 2.0)), len(post_pool))
    win = post_pool.sample(n=w, replace=(w > len(post_pool)), random_state=seed + 1)
    trows = win[win["pred_category"] == target].to_dict("records")
    orows = win[win["pred_category"] != target].to_dict("records")
    if len(trows) < need:
        return None
    ctr = Counter()
    for lvl in levels:
        ev, trows = trows[:per], trows[per:]
        mix = ev + orows[:len(ev)]
        orows = orows[len(ev):]
        rng.shuffle(mix)
        process(engine, auditor, method, mix, rng, audit_rate, force=lvl, targets=(target,), ctr=ctr)
    t = ctr.tgt.get(target, {"detected": 0, "silent": 0})
    return {"coverage": t["detected"] / need, "silent": t["silent"], "spend": ctr.rate()}


def scenario_two(preds, seed, method, audit_rate):
    sh = preds.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    cc = sh["pred_category"].value_counts()
    if len(cc) < 2:
        return None
    common = cc.idxmax()
    rare_el = cc[(cc >= DRIFT_BURST_LEN) & (cc.index != common)]
    if len(rare_el) == 0:
        return None
    rare = rare_el.idxmin()
    rp, cp = sh[sh["pred_category"] == rare], sh[sh["pred_category"] == common]
    rr = np.random.default_rng(seed + 100)
    r_idx = rr.choice(rp.index, size=DRIFT_BURST_LEN, replace=False)
    c_idx = rr.choice(cp.index, size=DRIFT_BURST_LEN, replace=False)
    reserved = set(r_idx) | set(c_idx)
    rem = sh[~sh.index.isin(reserved)].reset_index(drop=True)
    pre_n = min(1500, max(len(rem) - 200, 100))
    pre = rem.iloc[:pre_n].to_dict("records")
    opool = rem.iloc[pre_n:]
    rng = np.random.default_rng(seed)
    engine, auditor = _new(method, audit_rate)
    process(engine, auditor, method, pre, rng, audit_rate)
    ce, re_ = sh.loc[list(c_idx)].to_dict("records"), sh.loc[list(r_idx)].to_dict("records")
    opool = opool[~opool["pred_category"].isin([common, rare])]
    on = min(len(opool), DRIFT_BURST_LEN * 2)
    oe = opool.sample(n=on, random_state=seed + 1).to_dict("records") if on > 0 else []
    inter = list(zip(c_idx, ce)) + list(zip(r_idx, re_)) + [(None, r) for r in oe]
    rng.shuffle(inter)
    ctr = Counter()
    # original scenario C applies NO feedback noise and draws no noise RNG: mirrored with noisy=False
    process(engine, auditor, method, inter, rng, audit_rate, ctr=ctr, noisy=False, forced_idx=reserved)
    tc = ctr.tgt.get(common, {"detected": 0, "silent": 0})
    tr = ctr.tgt.get(rare, {"detected": 0, "silent": 0})
    return {"coverage_common": tc["detected"] / DRIFT_BURST_LEN, "coverage_rare": tr["detected"] / DRIFT_BURST_LEN,
            "silent_common": tc["silent"], "silent_rare": tr["silent"], "spend": ctr.rate()}


SCENARIOS = {
    "partial_70": (lambda p, s, m, a: scenario_partial(p, s, m, a, 0.70), [("coverage", "silent")]),
    "partial_60": (lambda p, s, m, a: scenario_partial(p, s, m, a, 0.60), [("coverage", "silent")]),
    "gradual": (scenario_gradual, [("coverage", "silent")]),
    "two_common": (scenario_two, [("coverage_common", "silent_common")]),
    "two_rare": (scenario_two, [("coverage_rare", "silent_rare")]),
}


def verdict(u, f, higher_better=True):
    u, f = np.asarray(u, float), np.asarray(f, float)
    if np.allclose(u, f):
        return "tie", None
    p = float(stats.wilcoxon(u, f).pvalue)
    if p >= 0.05:
        return "tie", p
    better = (u.mean() > f.mean()) if higher_better else (u.mean() < f.mean())
    return ("win" if better else "loss"), p


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    path = Path(sys.argv[1])
    prefix = sys.argv[2] if len(sys.argv) > 2 else path.stem
    preds = pd.read_csv(path)
    print(f"Loaded {len(preds)} predictions")
    out = {"verify_seeds_0_9": {}, "locked_seeds_10_19": {}}

    # Reproduction check on seeds 0-9 (original protocol, coverage only)
    cache = {}
    for sc, (fn, _) in SCENARIOS.items():
        key = "two" if sc.startswith("two") else sc
        for m in ["fixed_audit", "ugaa"]:
            if (key, m) not in cache:
                cache[(key, m)] = [fn(preds, s, m, AUDIT_RATE) for s in VERIFY_SEEDS]
    for sc, (_, metrics) in SCENARIOS.items():
        key = "two" if sc.startswith("two") else sc
        cov = metrics[0][0]
        out["verify_seeds_0_9"][sc] = {m: float(np.mean([r[cov] for r in cache[(key, m)] if r]))
                                       for m in ["fixed_audit", "ugaa"]}
    print("\nReproduction (seeds 0-9) coverage fixed / UGAA:")
    for sc, v in out["verify_seeds_0_9"].items():
        print(f"  {sc:<12} {v['fixed_audit']:.4f} / {v['ugaa']:.4f}")

    # Locked seeds 10-19 with spend logging and equal-spend control
    print(f"\nLocked seeds 10-19:")
    print(f"{'scenario':<12}{'spend f/U':>12}{'cov f5':>8}{'cov f@U':>8}{'cov U':>8}{'verdict(U vs f@U)':>20}"
          f"{'silent f@U':>11}{'silent U':>9}{'verdict':>12}")
    tally = {"coverage": {"win": 0, "tie": 0, "loss": 0}, "silent": {"win": 0, "tie": 0, "loss": 0}}
    runs = {}
    for sc, (fn, metrics) in SCENARIOS.items():
        key = "two" if sc.startswith("two") else sc
        if key not in runs:
            f5 = {s: fn(preds, s, "fixed_audit", AUDIT_RATE) for s in LOCKED_SEEDS}
            ug = {s: fn(preds, s, "ugaa", AUDIT_RATE) for s in LOCKED_SEEDS}
            fm = {s: (fn(preds, s, "fixed_audit", ug[s]["spend"]) if ug[s] else None) for s in LOCKED_SEEDS}
            runs[key] = (f5, ug, fm)
        f5, ug, fm = runs[key]
        ok = [s for s in LOCKED_SEEDS if f5[s] and ug[s] and fm[s]]
        cov, sil = metrics[0]
        g = lambda d, k: [d[s][k] for s in ok]
        vc, pc = verdict(g(ug, cov), g(fm, cov), True)
        vs, ps = verdict(g(ug, sil), g(fm, sil), False)
        tally["coverage"][vc] += 1
        tally["silent"][vs] += 1
        row = {"n_seeds": len(ok),
               "spend_fixed5": float(np.mean(g(f5, "spend"))), "spend_ugaa": float(np.mean(g(ug, "spend"))),
               "spend_fixed_matched": float(np.mean(g(fm, "spend"))),
               "coverage_fixed5": float(np.mean(g(f5, cov))), "coverage_fixed_matched": float(np.mean(g(fm, cov))),
               "coverage_ugaa": float(np.mean(g(ug, cov))), "coverage_verdict": vc, "coverage_p": pc,
               "silent_fixed5": float(np.mean(g(f5, sil))), "silent_fixed_matched": float(np.mean(g(fm, sil))),
               "silent_ugaa": float(np.mean(g(ug, sil))), "silent_verdict": vs, "silent_p": ps}
        out["locked_seeds_10_19"][sc] = row
        pf = lambda p: "" if p is None else f" p={p:.3f}"
        print(f"{sc:<12}{row['spend_fixed5']*100:>5.1f}/{row['spend_ugaa']*100:<5.1f}%"
              f"{row['coverage_fixed5']*100:>8.1f}{row['coverage_fixed_matched']*100:>8.1f}{row['coverage_ugaa']*100:>8.1f}"
              f"{vc + pf(pc):>20}{row['silent_fixed_matched']:>11.1f}{row['silent_ugaa']:>9.1f}{vs + pf(ps):>14}")
    out["tally_vs_equal_spend"] = tally
    print(f"\nTally vs equal-spend fixed audit (significant only): coverage {tally['coverage']}  "
          f"silent failures {tally['silent']}")
    with open(OUT_DIR / f"{prefix}_drift_scenarios_audit.json", "w") as f:
        json.dump(out, f, indent=2, default=float)
    print(f"Saved {prefix}_drift_scenarios_audit.json")


if __name__ == "__main__":
    main()
