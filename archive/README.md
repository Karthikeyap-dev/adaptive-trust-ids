# Archive (superseded experiments)

Nothing in this folder is used for any result in the paper. Kept for transparency only.
Some scripts contain known issues that were corrected in the published pipeline:

| Script(s) | Superseded by | Known issue |
|---|---|---|
| `locked_eval_ablation.py`, `threshold_grid_search.py`, `run_all_six_multiseed.py`, `pareto_sweep.py`, QPSO optimizer | `optimizer/locked_eval_robust.py`, `delta_sweep_robust.py`, `point_estimate_online.py` | Decisions used the end-of-stream trust snapshot instead of online trust; point-estimate threshold selection; QPSO replaced by exhaustive grid search |
| `multi_seed_and_ablation/*`, `confidence_only_six_configs.py`, `learning_to_defer_baseline.py`, static-trust baselines | `optimizer/locked_eval_robust.py` | Same snapshot issue; not on the locked split |
| `archetype_stress_test.py` | `archetypes/archetype_online.py` | Snapshot trust; single seed |
| `sensitivity_analysis_v2.py` | `statistics/noise_recovery_online.py` | "Stationary" decay sweep injected drift; drift sweep used the wrong pre-drift baseline; single seed |
| `prior_sensitivity.py` | `optimizer/prior_sensitivity_online.py` | Run on a configuration whose tuned policy ignores trust |
| `heterogeneity_*.py`, `effect_size_analysis.py`, `relative_change_analysis.py`, `consolidate_master_results.py` | removed from the paper / `statistics/effect_sizes_robust.py` | Based on superseded results |
| `figures/exploratory/*`, old figure scripts | `figures/fig4_fig5_reds.py`, `fig6_trust_credible.py` | Earlier figure versions |
