# Outline proposal

## Project scan summary

- Project archetype candidate: mixed data pipeline and empirical risk research.
- Supporting evidence from files: `offline_pipeline.py` defines a local, auditable path from two Parquet inputs to aligned day-$t$ factors and day-$t+1$ responses; the research modules add quantile, rank-correlation, subperiod, sensitivity, regime, and walk-forward checks.

## Blueprint selection

- Problem: test whether an open-interest-weighted SPY gamma measure is associated with next-day intraday variance.
- Options considered: data-pipeline, risk-model, and mixed.
- Selected blueprint: mixed.
- Why this blueprint fits this project: the engineering contract matters because it prevents hidden data access and lookahead, while the output is an empirical association study rather than a production forecast or a derivatives pricing model.
- Planned section order: research question; exact gamma convention; time alignment; reproducible evidence; what the null result says; implementation limitations; extensions.
- Verification: reproduce the January 2024 run from committed Parquet files and reconcile every reported number with frozen blog data.

## Planned equations

1. Contract-level gamma exposure
   - Purpose: define the quantity aggregated by the engine.
   - Symbols: open interest $OI_i$ in contracts, contract multiplier $M=100$, spot price $S_t$ in dollars, option gamma $\Gamma_i$ in inverse dollars, and contract index $i$.
   - Delimiter: display.
2. Daily aggregate
   - Purpose: show how contract values become the daily factor $G_t$.
   - Symbols: daily option set $\mathcal{O}_t$ and contract exposure $g_{i,t}$.
   - Delimiter: display.
3. Intraday realized variance
   - Purpose: define the next-day response.
   - Symbols: minute close $P_{t,j}$, minute log return $r_{t,j}$, and number of intraday returns $n_t$.
   - Delimiter: display.
4. Spearman rank correlation
   - Purpose: explain the monotonic association test without assuming linearity.
   - Symbols: rank variables $R(G_t)$ and $R(RV_{t+1})$, covariance, and standard deviations.
   - Delimiter: display.

## Planned code excerpts

1. File: `src/gamma_exposure_engine/exposure/cleaning.py`
   - Function/block: contract gamma-exposure expression.
   - Why include this excerpt: it exposes the exact sign and unit convention behind the headline factor.
2. File: `src/gamma_exposure_engine/research/dataset.py`
   - Function/block: next-observed-date alignment.
   - Why include this excerpt: it demonstrates how the pipeline avoids pairing an exposure with a same-day response.

## Planned technical graphs

1. Graph type: dual-axis daily time series.
   - Source: generate from frozen `gamma_factors.csv` and `research_dataset.parquet`.
   - Expected takeaway: gamma changes materially across days, but next-day variance does not visually co-move in a stable way.
2. Graph type: quantile means with bootstrap confidence intervals.
   - Source: generate from frozen `quantile_summary.csv`.
   - Expected takeaway: bucket means are not monotonic and the intervals overlap heavily.
3. Graph type: scatter plot with rank-correlation annotation.
   - Source: generate from the frozen aligned dataset.
   - Expected takeaway: 20 observations provide no persuasive monotonic relationship.

## Risks, gaps, and assumptions

- Data gaps: the committed demo covers 21 trading days and yields 20 aligned observations. The configured 20-day regime lookback and 20-row minimum training sample leave no out-of-sample regime or predictive results.
- Assumptions: open interest is used as supplied; gamma is aggregated without an inferred dealer-position sign; the analysis is descriptive and associational.
- Critical limitation: every stored contract gamma is non-negative in the demo. Consequently `net_gamma_exposure` equals `absolute_gamma_exposure`, and the negative-gamma-node factor is undefined.
- Validation checks: rerun the offline pipeline, compare row counts and statistics, execute the chart script, resolve every image reference, run the blog validator, and confirm no website files were touched.

## Deployment note

The canonical workspace is `gamma-exposure-engine/blog/`. The normal skill workflow would copy publishable files to `~/projects/website/content/post/<slug>/`, build Hugo, and commit the generated site. The user explicitly deferred publishing, so this task stops at the project-local bilingual package and will not read from, write to, build, commit, or push the website repository.
