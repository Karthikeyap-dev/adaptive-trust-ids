"""
Section 5.7: human-study accuracy split by whether the classifier was correct on each alert
(the over-reliance check), plus the overall blind vs. assisted comparison.
Reads outputs/pilot_study_results_INFOSEC.csv and outputs/pilot_study_results_CSEbatch3.csv.

Run from code/:   python3 human_study/ai_correctness_breakdown.py
"""
from pathlib import Path
import pandas as pd
from scipy.stats import wilcoxon, ttest_rel

OUT = Path(__file__).resolve().parent.parent / "outputs"
d = pd.concat([pd.read_csv(OUT / f) for f in ("pilot_study_results_INFOSEC.csv", "pilot_study_results_CSEbatch3.csv")])
d["blind"] = (d.phase1_blind_label == d.true_category).astype(int)
d["assisted"] = (d.phase2_final_label == d.true_category).astype(int)
d["ai_correct"] = (d.system_prediction == d.true_category).astype(int)

p = d.groupby("participant_id")[["blind", "assisted"]].mean()
diff = p.assisted - p.blind
print(f"Participants: {p.shape[0]}, responses: {len(d)}")
print(f"Blind {p.blind.mean()*100:.1f}%  ->  assisted {p.assisted.mean()*100:.1f}%  (+{diff.mean()*100:.1f} points, "
      f"Cohen's d={diff.mean()/diff.std(ddof=1):.3f}, t-test p={ttest_rel(p.assisted, p.blind).pvalue:.2g}, "
      f"Wilcoxon p={wilcoxon(p.assisted, p.blind).pvalue:.2g})")
alerts = d.drop_duplicates("alert_index_in_study")
print(f"Classifier correct on {alerts.ai_correct.sum()} of {len(alerts)} alerts")
for ok, name in ((1, "classifier correct"), (0, "classifier wrong")):
    s = d[d.ai_correct == ok].groupby("participant_id")[["blind", "assisted"]].mean()
    print(f"  {name:<18}: blind {s.blind.mean()*100:.1f}%  ->  assisted {s.assisted.mean()*100:.1f}%")
