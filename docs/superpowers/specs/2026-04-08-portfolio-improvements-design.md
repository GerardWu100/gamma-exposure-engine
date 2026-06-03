# Gamma Exposure Engine v2: Portfolio Improvement Design

## Document Purpose

This spec defines improvements to the gamma exposure engine that strengthen
its value as a portfolio project for general quant role interviews. The
improvements balance research depth with engineering polish, targeting the
questions an experienced quant interviewer would ask.

The v1 engine is already well-built: clean architecture, thorough docstrings,
proper type hints, methodological honesty, and lookahead bias avoidance. This
spec extends v1 where it matters most for interview credibility.

## Goals

After this work, the project should additionally demonstrate:

- statistical rigor (confidence intervals, formal tests, not just point
  estimates),
- research robustness (regime conditioning, subperiod stability, alternative
  specifications),
- breadth of analysis (multi-factor summaries, not just one factor-target pair
  at a time),
- predictive discipline (regularized models, expanding-window diagnostics,
  prediction intervals),
- production engineering habits (proper CLI, structured logging, CI pipeline),
- clear communication (README that sells the project, architecture diagram,
  sample output).

## Hiring Narrative (Updated)

> I built a reproducible empirical research system for SPY gamma exposure that
> tests associations with next-day intraday behavior using bootstrap inference,
> regime conditioning, and walk-forward prediction. The pipeline includes
> offline-first caching, a proper CLI, and CI, and the results are communicated
> through self-contained HTML reports with honest limitations.

## Scope

### In Scope

- Bootstrap confidence intervals on quantile summary metrics.
- Non-parametric statistical tests (Kruskal-Wallis, Spearman correlation).
- Regime-conditional analysis using trailing realized volatility.
- Multi-factor rank correlation summary.
- Expanded robustness checks (subperiod stability, alternative moneyness
  bands, leave-one-out month sensitivity).
- Predictive appendix upgrade (Ridge regression, expanding-window error
  diagnostics, residual bootstrap prediction intervals).
- Proper CLI with `typer` and `[project.scripts]` entry point.
- Structured logging via stdlib `logging`.
- GitHub Actions CI workflow.
- Architecture diagram in README (Mermaid).
- README rewrite for portfolio presentation.
- Sample report committed to `outputs/samples/`.

### Out of Scope

- Streamlit app improvements (explicitly excluded by user).
- New data sources or symbols beyond SPY.
- Deep learning or complex ML models.
- Real-time streaming.
- Causal claims about dealer hedging.

## Design

### 1. Bootstrap Confidence Intervals

**New file:** `src/gamma_exposure_engine/research/bootstrap.py`

**Purpose:** Attach nonparametric bootstrap confidence intervals to the
existing quantile summary metrics, transforming point estimates into
statistically grounded ranges.

**Algorithm:**

1. Accept the aligned research frame, factor name, target name, quantile
   count, number of bootstrap iterations, and confidence level.
2. For each iteration, resample the research frame rows with replacement
   (preserving the full row so factor-target pairing is maintained).
3. Run `build_quantile_summary` on each resampled frame to get the target
   mean per bucket.
4. Collect the distribution of target means per bucket across all iterations.
5. Compute percentile-based confidence intervals from the bootstrap
   distribution.
6. Return a DataFrame with one row per bucket: bucket label, point estimate,
   CI lower bound, CI upper bound, observation count.

**Key decisions:**

- Nonparametric bootstrap (no distributional assumptions).
- Resample rows, not residuals. This is appropriate for a descriptive study
  where the question is "are these bucket means reliably different?"
- Default 1000 iterations, 95% confidence level, both configurable.
- The CI captures sampling uncertainty in bucket assignment and target mean
  estimation simultaneously.

**Config additions to `config.toml`:**

```toml
# Number of bootstrap resampling iterations for confidence intervals.
bootstrap_iterations = 1000
# Confidence level for bootstrap percentile intervals (0 to 1).
bootstrap_confidence_level = 0.95
```

**Integration:** `cli.py` calls `build_quantile_summary_with_ci` instead of
`build_quantile_summary`. The HTML report renders CI columns alongside the
existing target mean column.

### 2. Non-Parametric Statistical Tests

**New file:** `src/gamma_exposure_engine/research/statistical_tests.py`

**Purpose:** Provide formal statistical evidence for whether quantile bucket
differences and factor-target associations are statistically significant.

**Tests implemented:**

- **Kruskal-Wallis H-test:** Tests whether the target variable distributions
  differ across quantile buckets. Non-parametric, appropriate for small
  samples and non-normal distributions. Uses `scipy.stats.kruskal`.
- **Spearman rank correlation:** Tests monotonic association between the
  factor and target across the full sample. Reports the correlation
  coefficient and two-sided p-value. Uses `scipy.stats.spearmanr`.

**Output:** A small DataFrame with test name, test statistic, p-value, and
sample size. Rendered as a compact table in the HTML report.

**Why these tests:**

- Kruskal-Wallis is the multi-group generalization of the Mann-Whitney U
  test. It answers "do these buckets come from different distributions?"
  without assuming normality.
- Spearman correlation answers "is there a monotonic relationship?" without
  assuming linearity.
- Both are standard in empirical finance and defensible in interviews.

**Dependency addition:** `scipy` (already a transitive dependency of
scikit-learn, but make it explicit in `pyproject.toml`).

### 3. Regime-Conditional Analysis

**New file:** `src/gamma_exposure_engine/research/regime.py`

**Purpose:** Test whether the factor-target association is state-dependent by
conditioning on market volatility regime.

**Algorithm:**

1. Compute a trailing realized volatility measure from the response metrics
   (using a past-only rolling window to avoid lookahead).
2. Classify each day as "high volatility" or "low volatility" using the
   trailing median as the split point (adaptive, not a fixed threshold).
3. Run the quantile summary within each regime separately.
4. Optionally run the Kruskal-Wallis test within each regime.

**Key decisions:**

- Use a trailing rolling window of *past* realized variance values to
  classify each day's regime. On day `t`, the regime label is computed
  from realized variance on days `t-W` through `t-1` (where `W` is the
  lookback window). This avoids lookahead because day `t`'s own response
  is never used in its own regime classification.
- Median split is simple and robust. No need for terciles or quartiles with
  the sample sizes involved.
- The trailing window length is configurable (default: 20 trading days,
  matching the existing abnormal volume window).

**Config additions:**

```toml
# Trailing window for volatility regime classification (trading days).
regime_lookback_window = 20
```

**Output:** A DataFrame with regime label, bucket, target mean, observation
count. Rendered as a split table in the report.

### 4. Multi-Factor Summary

**New file:** `src/gamma_exposure_engine/research/multi_factor.py`

**Purpose:** Compute and display a rank correlation matrix across all
available factors and targets, showing which factors carry independent
information.

**Algorithm:**

1. Identify all factor columns and all target columns in the research
   dataset.
2. Compute the Spearman rank correlation between every factor-target pair.
3. Compute Spearman correlations among factors to identify redundancy.
4. Return two DataFrames: a factor-target correlation matrix and a
   factor-factor correlation matrix.

**Output:** Two compact tables in the report. The factor-target matrix helps
an interviewer see which factors matter. The factor-factor matrix shows
whether factors are redundant.

**Factor columns** (from `build_daily_gamma_factors`):
- `net_gamma_exposure`
- `absolute_gamma_exposure`
- `near_spot_gamma_share`
- `front_expiry_gamma_share`
- `call_put_gamma_imbalance`
- `exposure_concentration_index`

**Target columns** (from `build_daily_intraday_metrics` + pinning):
- `next_day_realized_variance`
- `next_day_open_to_close_abs_return`
- `next_day_close_to_close_abs_return`
- `next_day_abnormal_volume`
- `next_day_pinning_distance`

### 5. Richer Robustness Checks

**Modified file:** `src/gamma_exposure_engine/research/descriptive.py`

**New functions added to the existing descriptive module** (these are natural
extensions of the existing robustness work):

#### 5a. Subperiod Stability

Split the sample at the temporal midpoint. Run the quantile summary and
Spearman correlation in each half independently. Report whether the
association is stable across subperiods.

**Output:** A DataFrame with subperiod label ("first_half", "second_half"),
Spearman rho, p-value, and observation count.

#### 5b. Alternative Moneyness Bands

Recompute `near_spot_gamma_share` at alternative band widths (1%, 3%, 5% in
addition to the default 2%). This requires calling the exposure aggregation
layer with different `near_spot_band` parameters.

**Important:** This is a substantive robustness check that the v1 design
document explicitly acknowledged was missing. The existing threshold
robustness only splits the already-computed factor; this actually recomputes
it.

**Implementation note:** The CLI must pass the cleaned options DataFrame
through to this robustness step so that `build_daily_gamma_factors` can be
called with alternative `near_spot_band` values. The existing pipeline
already has `cleaned_options` in scope at the orchestration level, so this
is a matter of threading it to the robustness function.

**Output:** A DataFrame with band width, Spearman rho with the target, and
observation count.

#### 5c. Leave-One-Month-Out Sensitivity

For each calendar month in the sample, drop that month and recompute the
Spearman correlation between the default factor and target. Report the range
and stability of correlations.

**Output:** A DataFrame with dropped month, Spearman rho, p-value, and
remaining observation count.

### 6. Predictive Appendix Upgrade

**Modified file:** `src/gamma_exposure_engine/research/predictive.py`

#### 6a. Ridge Regression Baseline

Add a `walk_forward_ridge_baseline` function alongside the existing linear
baseline. Ridge regression adds L2 regularization, which is appropriate for
small samples where OLS coefficients may be unstable.

- Uses `sklearn.linear_model.Ridge` with cross-validated alpha selection
  via `sklearn.linear_model.RidgeCV`.
- The walk-forward structure is identical to the existing OLS baseline:
  train on all rows before the prediction row, predict one step ahead.
- Default alpha candidates: `[0.01, 0.1, 1.0, 10.0, 100.0]`.

**Config additions:**

```toml
# Alpha candidates for Ridge regression cross-validation in the predictive appendix.
ridge_alpha_candidates = [0.01, 0.1, 1.0, 10.0, 100.0]
```

#### 6b. Expanding-Window Error Diagnostic

Add a function that returns the per-step out-of-sample error for each model
(OLS, Ridge, naive baseline) as a time series. This lets the report show how
prediction quality evolves as training data grows.

**Output:** A DataFrame with trade_date, model_name, absolute_error. Rendered
as a line chart in the report.

#### 6c. Residual Bootstrap Prediction Intervals

After the walk-forward loop, compute prediction intervals using the residual
bootstrap method:

1. Collect the OOS residuals from the walk-forward evaluation.
2. For each prediction, add resampled residuals to the point prediction.
3. Report the 5th and 95th percentile as the 90% prediction interval.

**Output:** Prediction interval bounds appended to the existing prediction
DataFrame.

### 7. Proper CLI with `typer`

**Modified file:** `src/gamma_exposure_engine/cli.py`

Replace the current `run_pipeline` function-call invocation pattern with a
`typer` CLI:

```bash
uv run gex report --start 2024-01-02 --end 2024-01-10
uv run gex report --start 2024-01-02 --end 2024-01-10 --factor call_put_gamma_imbalance
```

**Structure:**

- `cli.py` gets a `typer.Typer()` app with a `report` subcommand.
- The `report` subcommand wraps the existing `run_pipeline` function.
- Arguments: `--start` (required), `--end` (required), `--factor`
  (optional, defaults to config), `--target` (optional, defaults to config),
  `--output-dir` (optional).
- The existing `run_pipeline` function remains as the engine; the CLI is a
  thin wrapper.

**pyproject.toml addition:**

```toml
[project.scripts]
gex = "gamma_exposure_engine.cli:app"
```

**Dependency addition:** `typer` in `[project.dependencies]`.

### 8. Structured Logging

**Modified files:** `cli.py`, `data/cache.py`, `data/options_queries.py`,
`data/intraday_queries.py`

Replace any `print()` calls with Python stdlib `logging`:

- `cli.py` configures root logger at the top of `run_pipeline` (INFO by
  default, DEBUG via `--verbose` CLI flag).
- Pipeline stages log at INFO: "Fetching intraday bars for SPY",
  "Cleaning 12,450 option rows", "Building research dataset with 245
  trading days".
- Cache hits/misses log at DEBUG.
- Small sample warnings log at WARNING (e.g., "Only 18 observations in
  quantile bucket 3").

No external logging dependency needed.

### 9. GitHub Actions CI

**New file:** `.github/workflows/ci.yml`

Simple single-job workflow:

```yaml
name: CI
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v6
      - run: uv sync --frozen
      - run: uv run pytest
```

Badge added to the top of README.md.

### 10. README Rewrite

**Modified file:** `README.md`

New structure:

1. **Hook:** One-sentence summary that frames the research question
   compellingly ("What does options dealer positioning tell us about
   tomorrow's price action?").
2. **CI badge** from GitHub Actions.
3. **Architecture diagram** (Mermaid) showing the six-layer data flow.
4. **What it does** (concise bullet list, updated for v2 additions).
5. **Sample output** link to the committed sample report.
6. **Quick start** (CLI commands using the new `gex` entry point).
7. **Key design decisions** section that surfaces interview-worthy choices:
   - Why t+1 alignment (lookahead bias avoidance)
   - Exposure convention choice (spot^2 scaling)
   - Bootstrap inference on small samples
   - Regime conditioning rationale
   - Walk-forward evaluation discipline
8. **Methodology notes** (kept from v1, made scannable).
9. **Configuration** (updated for new settings).
10. **Layout** (updated for new modules).

### 11. Sample Report

Commit one generated HTML report to `outputs/samples/` so GitHub visitors can
see what the pipeline produces without cloning and running. This is committed
as a build artifact, not generated in CI (it requires ClickHouse data).

## New Files Summary

| File | Purpose |
|------|---------|
| `src/gamma_exposure_engine/research/bootstrap.py` | Bootstrap CI computation |
| `src/gamma_exposure_engine/research/statistical_tests.py` | Kruskal-Wallis, Spearman tests |
| `src/gamma_exposure_engine/research/regime.py` | Volatility regime conditioning |
| `src/gamma_exposure_engine/research/multi_factor.py` | Cross-factor rank correlation matrices |
| `tests/research/test_bootstrap.py` | Bootstrap CI tests |
| `tests/research/test_statistical_tests.py` | Statistical test module tests |
| `tests/research/test_regime.py` | Regime conditioning tests |
| `tests/research/test_multi_factor.py` | Multi-factor summary tests |
| `.github/workflows/ci.yml` | CI pipeline |

## Modified Files Summary

| File | Changes |
|------|---------|
| `src/gamma_exposure_engine/research/descriptive.py` | Add subperiod stability, alternative bands, leave-one-month-out |
| `src/gamma_exposure_engine/research/predictive.py` | Add Ridge baseline, expanding-window diagnostics, prediction intervals |
| `src/gamma_exposure_engine/cli.py` | typer CLI wrapper, structured logging, orchestrate new research steps |
| `src/gamma_exposure_engine/reporting/html_report.py` | New report sections for CIs, tests, regimes, multi-factor, diagnostics |
| `src/gamma_exposure_engine/reporting/charts.py` | Error bar support, expanding-window diagnostic chart |
| `src/gamma_exposure_engine/settings.py` | New config fields for bootstrap, regime, ridge |
| `src/gamma_exposure_engine/config.toml` | New configuration entries |
| `pyproject.toml` | Add typer, scipy; add [project.scripts] entry point |
| `README.md` | Full rewrite for portfolio presentation |
| `tests/research/test_descriptive.py` | Tests for new robustness functions |
| `tests/research/test_predictive.py` | Tests for Ridge, diagnostics, prediction intervals |
| `tests/test_cli_smoke.py` | Update for typer CLI |
| `GUIDE_ROOT.md` | Update for new modules |

## Dependency Additions

| Package | Reason |
|---------|--------|
| `typer` | CLI framework |
| `scipy` | Statistical tests (Kruskal-Wallis, Spearman). Already a transitive dep of scikit-learn, but make explicit. |

## Config Additions Summary

```toml
[research]
# Number of bootstrap resampling iterations for confidence intervals.
bootstrap_iterations = 1000
# Confidence level for bootstrap percentile intervals (0 to 1).
bootstrap_confidence_level = 0.95
# Trailing window for volatility regime classification (trading days).
regime_lookback_window = 20
# Alpha candidates for Ridge regression cross-validation.
ridge_alpha_candidates = [0.01, 0.1, 1.0, 10.0, 100.0]
# Alternative near-spot band widths tested in the robustness section.
robustness_band_widths = [0.01, 0.03, 0.05]
```

## Testing Strategy

Each new module gets focused unit tests:

- **bootstrap.py:** Verify CI bounds bracket the point estimate, verify
  reproducibility with fixed seed, verify empty-frame handling.
- **statistical_tests.py:** Verify Kruskal-Wallis detects a known difference
  (synthetic data with separated distributions), verify Spearman detects a
  known monotonic relationship, verify output schema.
- **regime.py:** Verify regime classification uses past-only data (no
  lookahead), verify split produces two non-empty groups, verify output
  schema.
- **multi_factor.py:** Verify correlation matrix shape matches factor/target
  counts, verify perfect correlation on synthetic data, verify NaN handling.
- **descriptive.py (new functions):** Verify subperiod split at midpoint,
  verify alternative bands produce different share values, verify
  leave-one-month-out output row count equals month count.
- **predictive.py (new functions):** Verify Ridge walk-forward produces same
  structure as OLS, verify prediction intervals bracket actuals at expected
  coverage on synthetic data, verify expanding-window diagnostic output
  schema.

## Implementation Order

The recommended build order minimizes integration risk:

1. **Foundation:** Add `scipy` and `typer` dependencies, add new config
   fields to settings and config.toml.
2. **Bootstrap CIs:** New module + tests + integration into descriptive
   summary.
3. **Statistical tests:** New module + tests + integration into CLI.
4. **Regime conditioning:** New module + tests + integration into CLI.
5. **Multi-factor summary:** New module + tests + integration into CLI.
6. **Robustness extensions:** New functions in descriptive.py + tests +
   integration into CLI.
7. **Predictive upgrade:** Ridge, diagnostics, prediction intervals + tests.
8. **Report rendering:** Update html_report.py and charts.py for all new
   sections.
9. **CLI rewrite:** typer wrapper + structured logging.
10. **CI pipeline:** GitHub Actions workflow.
11. **README and sample report:** Final presentation polish.
12. **GUIDE_ROOT.md update:** Reflect new module structure.

## Interview Questions This Work Enables

After this work, the author can additionally answer:

- How confident are you in those quantile bucket differences? (Bootstrap CIs)
- Are the bucket differences statistically significant? (Kruskal-Wallis)
- Is there a monotonic relationship? (Spearman)
- Does the effect hold in volatile vs calm markets? (Regime conditioning)
- Which factors carry independent information? (Multi-factor matrix)
- Is the result stable across time? (Subperiod stability)
- Does it survive alternative specifications? (Alternative bands, LOO months)
- Why Ridge over OLS? (Regularization rationale for small samples)
- How do your predictions evolve as training data grows? (Expanding-window
  diagnostics)
- What are the prediction intervals? (Residual bootstrap)
