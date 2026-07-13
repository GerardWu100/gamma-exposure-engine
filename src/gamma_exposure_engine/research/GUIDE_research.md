# GUIDE_research

`src/gamma_exposure_engine/research/` contains modular analysis blocks that operate on aligned daily
gamma-mass/response data. The exposure-side columns describe unsigned option
structure; they do not identify dealer inventory.

## Modules

- `dataset.py`: aligns exposure day `t` with response day `t+1`.
- `descriptive.py`: quantile summaries and robustness splits.
- `bootstrap.py`: confidence intervals for quantile means.
- `statistical_tests.py`: Spearman and Kruskal-Wallis significance tests.
- `regime.py`: volatility-regime conditioned summaries.
- `multi_factor.py`: factor-target and factor-factor rank correlations.
- `predictive.py`: walk-forward linear and Ridge baselines plus diagnostics.

## Blog-Ready Backbone

The following outputs are natural table sources for a future blog post:

- `quantile_summary.csv`
- `statistical_tests.csv`
- `regime_summary.csv`
- `predictive_comparison.csv`
- `threshold_robustness.csv`
- `subperiod_stability.csv`
- `band_sensitivity.csv`
- `loo_month_sensitivity.csv`
