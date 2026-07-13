# Outline proposal

## Project scan summary

- Project archetype candidate: mixed data pipeline and empirical risk research.
- Supporting evidence from files: the offline pipeline transforms two committed
  Parquet inputs into day-$t$ option-structure factors and day-$t+1$ responses;
  the research modules add quantiles, rank tests, robustness splits, and
  past-only predictive checks.

## Blueprint selection

- Selected blueprint: mixed.
- Why this blueprint fits this project: the main research result depends on both
  a precise data contract and an empirical association test. It is neither a
  dealer-inventory estimator nor a derivatives pricing exercise.
- Planned section order: market claim; observable data; gamma derivation and
  units; corrected code interface; time alignment; evidence; diagnostics;
  economic interpretation; limitations; references.

## Planned equations

1. Contract gamma mass:
   - Purpose: distinguish observable open-interest-weighted curvature from
     unobserved signed dealer inventory.
   - Symbols: open interest $OI_{i,t}$, contract multiplier $M$, spot $S_t$,
     vanilla gamma $\Gamma_{i,t}$, and contract mass $m_{i,t}$.
   - Delimiter: display.
2. Daily gamma mass and one-percent scaling:
   - Purpose: define the factor and convert its raw dollar unit to a one-percent
     spot-move convention.
   - Symbols: option set $\mathcal O_t$, total $G_t$, and $G_t^{1\%}$.
   - Delimiter: display.
3. Realized variance:
   - Purpose: define the next-day response from minute log returns.
   - Symbols: price $P_{t,j}$, return $r_{t,j}$, and return count $n_t$.
   - Delimiter: display.
4. Spearman rank correlation:
   - Purpose: test monotonic association without assuming a linear relation.
   - Symbols: ranks, covariance, and standard deviations.
   - Delimiter: display.

## Planned code excerpts

1. File: `src/gamma_exposure_engine/exposure/cleaning.py`
   - Function/block: open-interest-weighted gamma expression.
   - Why include this excerpt: it proves that no dealer sign is fabricated.
2. File: `src/gamma_exposure_engine/research/dataset.py`
   - Function/block: next-observed-trading-date join.
   - Why include this excerpt: it proves the factor precedes the response.

## Planned technical graphs

1. Graph type: dual-axis daily time series.
   - Source: regenerated from the corrected frozen research dataset.
   - Expected takeaway: gamma mass varies, but next-day variance does not track it.
2. Graph type: quintile means with bootstrap intervals.
   - Source: regenerated from the corrected quantile table.
   - Expected takeaway: means are non-monotonic and intervals overlap.
3. Graph type: scatter with descriptive fit and rank statistic.
   - Source: regenerated from the corrected aligned dataset.
   - Expected takeaway: 20 observations show no persuasive association.

## Risks, gaps, and assumptions

- Data gaps: 21 exposure dates produce only 20 aligned observations; regime and
  walk-forward outputs are empty under the configured minimum histories.
- Assumptions: open interest is used as supplied; vanilla gamma is non-negative;
  the factor measures unsigned market structure, not dealer positioning.
- Validation checks: run all tests, reproduce the pipeline, regenerate and
  inspect figures, reconcile numerical claims, validate both posts, compare
  protected blocks, resolve image paths, and confirm the website is untouched.

## Deployment note

The canonical workspace is `gamma-exposure-engine/blog/`. The user explicitly
deferred website publishing, so no files will be copied to `~/projects/website`
and no Hugo build or website commit will be made.
