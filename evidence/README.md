# Evidence Bundle

This folder keeps a compact set of CSV tables for the main manuscript and essential supplementary checks. The release is intentionally lean; exploratory and intermediate tables are omitted.

## Main and Essential Tables

- `table_public_checkpoint_sources.csv`: target-matched public-checkpoint baseline summary.
- `table_no_vina_risk_detection.csv`: no-Vina physical-reliability critic detection metrics.
- `table_no_vina_wilson_selection.csv`: Wilson selective-risk rows under the no-Vina setting.
- `table_same_pool_repair.csv`: same-candidate-pool selection and repair comparison.
- `table_pareto_public_sources.csv`: paired public-source Pareto comparison.
- `table_portfolio_summary.csv`: portfolio policy confirmation summary.
- `table_composite_bootstrap.csv`: bootstrap confidence intervals for composite improvement.
- `table_multiseed_stability.csv`: seed-stability summary.
- `table_orthogonal_validation.csv`: orthogonal validation checks.

All numeric tables are plain CSV files so that reviewers can inspect them directly or load them with pandas.
