# ugaa-adaptive-trust

Code for **"Reliability-Aware Human–AI Arbitration via Category-Specific Adaptive Trust for Intrusion Detection"** (submitted to *Cybersecurity*, Springer).

- Code archive (this repository): DOI `10.5281/zenodo.XXXXXXX`
- Results, seed logs and prediction files: DOI `10.5281/zenodo.YYYYYYY`
- Licence: MIT

## Setup
```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```
Python 3.13. The three benchmark datasets are **not** redistributed; download them from the original sources and place them as described in `data_README.md`:
NSL-KDD (https://www.unb.ca/cic/datasets/nsl.html), UNSW-NB15 (https://research.unsw.edu.au/projects/unsw-nb15-dataset), CICIDS2017 (https://www.unb.ca/cic/datasets/ids-2017.html).

To reproduce results **without retraining**, download the results archive (second DOI) and copy its `outputs/` folder into `code/outputs/`.

All commands run from the `code/` folder unless stated otherwise.

## Paper results → scripts

| Paper item | Script | Output (code/outputs/) |
|---|---|---|
| Classifier training, Table 4 | `baselines/train_baseline.py`, `xgb_baseline.py`, `unsw_train_baseline.py`, `unsw_xgb_baseline.py`, `cicids_train_baseline.py`, `cicids_xgb_baseline.py` | `*_baseline_predictions.csv`, `*_baseline_results.json` |
| Wilson CIs (Table 4) | `statistics/compute_uncertainty_intervals.py` | `uncertainty_intervals.json` |
| Temporal-split check (§4.1) | `statistics/cicids_temporal_split.py` (needs raw CICIDS2017) | `cicids_temporal_split_rerun.json` |
| Table 5, calibration paragraph | `statistics/cross_dataset_calibration.py`, `baselines/calibrate_confidence.py` | `cross_dataset_calibration.json`, `calibration_results.json` |
| Fig 4, Fig 5 | `figures/fig4_fig5_reds.py` | `figures/fig4_*`, `figures/fig5_*` |
| Fig 6 | `figures/fig6_trust_credible.py` | `figures/fig6_*` |
| Tables 6, 8, 10 | `cd optimizer && python3 locked_eval_robust.py ../outputs/<predictions>.csv <config> 20` | `<config>_locked_eval_robust.csv/.json` |
| Table 7 | `statistics/effect_sizes_robust.py` | `effect_sizes_robust.csv` |
| Table 9 | `cd optimizer && python3 point_estimate_online.py` | `point_estimate_online.json` |
| Fig 7, Table 12 | `cd optimizer && python3 delta_sweep_robust.py ...` then `--plot` | `*_delta_sweep_robust.csv`, `delta_sweep_robust.png` |
| Prior sensitivity (§3.3) | `cd optimizer && python3 prior_sensitivity_online.py` | `unsw_rf_prior_sensitivity_online.json` |
| Table 11 (AURC) | `statistics/risk_coverage_aurc.py` | `*_risk_coverage_aurc.json` |
| Table 13 | `selective_feedback/selective_feedback.py` | `*_selective_feedback*.json` |
| UGAA k selection (seeds 0–9), 24.1-point gap | `selective_feedback/ugaa_extended_experiments.py` | `*_ugaa_extended_results.json` |
| UGAA steady-state spend | `selective_feedback/ugaa_experiment.py` | `*_ugaa_results.json` |
| Table 14 | `cd selective_feedback && python3 ugaa_drift_audit.py ../outputs/<predictions>.csv <config>` | `*_ugaa_drift_audit.json` |
| Broader drift scenarios (11/19/0) | `cd selective_feedback && python3 drift_scenarios_audit.py ../outputs/<predictions>.csv <config>` | `*_drift_scenarios_audit.json` |
| §5.6 noise, burst recovery, λ sweep | `statistics/noise_recovery_online.py` | `noise_recovery_online.json` |
| Table 15, Table 5 (last column), bias sweep | `cd archetypes && python3 archetype_online.py` | `archetype_online_results.json` |
| §5.7 human study | `human_study/analyze_pilot_results.py`, `human_study/ai_correctness_breakdown.py`, `evaluation/glmm_reanalysis.py`, `human_study/decision_time_analysis.py` | `pilot_study_summary.json`, `glmm_reanalysis_results.json`, `decision_time_results.json` |
| SHAP deletion test | `human_study/explanation_faithfulness.py` | `faithfulness_results.json` |
| Inference latency | `cost/measure_real_latency_v2.py` | `measured_latency_confirmed.json` |
| Environment | `statistics/gather_reproducibility_info.py` | `reproducibility_info_*.json` |
| Figs 1–3 (diagrams) | SVG sources in `figures/diagrams/` | — |

## Archive
`archive/` holds earlier, **superseded** experiments kept for transparency. They are **not used for any result in the paper** and some produce different numbers (see `archive/README.md`).
