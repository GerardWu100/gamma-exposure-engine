# Gamma Exposure Engine Design

## Document Purpose

This document defines the design for the `Gamma Exposure Engine` project.
The goal is to build an interview-quality quantitative research project that
computes daily `SPY` option gamma exposure, links that positioning state to
next-day intraday market behavior, and presents the results through a thin
exploration interface and a reproducible research report.

The design is intentionally research-first. The frontend exists to support
inspection and explanation, not to be the main artifact.

## Project Goal

The project should demonstrate that the author can:

- query and validate market data from `ClickHouse`,
- define option exposure factors clearly and defensibly,
- avoid common empirical mistakes such as lookahead bias,
- translate daily options positioning into interpretable research features,
- connect those features to intraday realized behavior,
- communicate methodology, limitations, and findings clearly.

The project is optimized for `Quant Research` interviews.

## Hiring Narrative

The project should support this interview narrative:

> I built a reproducible empirical research system for measuring daily `SPY`
> gamma exposure, testing its association with next-day intraday behavior, and
> documenting the timing, scaling, robustness, and interpretation choices that
> matter in practice.

This is a stronger story than "I built a gamma dashboard" because it signals
research judgment, data engineering ability, and methodological discipline.

## Scope

### In Scope

- Daily option positioning analysis for `SPY`.
- Contract-level cleaning and exposure aggregation.
- Strike-level, expiry-level, and strike-by-expiry gamma maps.
- Daily summary factor construction.
- Next-day intraday response metrics from `SPY` ETF bars.
- Descriptive event studies and quantile analyses.
- A small predictive appendix with walk-forward evaluation.
- A self-contained HTML research report.
- A thin `Streamlit` app for exploratory inspection.

### Out of Scope

- Live trading or signal execution.
- Real-time streaming infrastructure.
- Deep learning models.
- Multi-symbol portfolio research in version `1`.
- Heavy frontend engineering.
- Strong causal claims about dealer hedging.

## Data Sources

### Options Data

Source table: `firstrate.options`

Available fields confirmed in local data:

- `symbol`
- `trade_date`
- `strike_price`
- `expiry_date`
- `option_type`
- `last_price`
- `bid`
- `ask`
- `bid_iv`
- `ask_iv`
- `open_interest`
- `volume`
- `delta`
- `gamma`
- `vega`
- `theta`
- `rho`

The project will use `SPY` option rows only in version `1`.

### Intraday Underlying Data

Source table: `firstrate.etfs`

Available fields confirmed in local data:

- `symbol`
- `ts`
- `open`
- `high`
- `low`
- `close`
- `volume`

The project will use intraday `SPY` bars to compute realized behavior metrics.

## Why `SPY` for Version 1

`SPY` is the correct version `1` underlying because:

- the local database contains dense `SPY` options history from `2010-01-04`
  through `2024-12-31`,
- the local database also contains `SPY` intraday ETF bars through
  `2026-01-05`,
- the symbol is familiar to interviewers,
- the data path is cleaner than forcing `SPX` before verifying matching
  intraday index coverage.

`SPX` is the natural extension after the `SPY` pipeline is working.

## Research Question

Primary question:

> Is daily `SPY` option gamma positioning associated with measurable
> differences in next-day intraday realized volatility, abnormal volume, and
> price pinning-style behavior?

Secondary question:

> Do a small set of interpretable gamma exposure factors improve simple
> forecasts of next-day realized behavior beyond naive volatility baselines?

## Claim Style

The project will use a `mechanism-informed` but non-causal narrative.

That means:

- dealer hedging mechanics may be used as economic motivation,
- results will be presented as empirical associations,
- same-day charts may be shown for intuition,
- the main study will avoid claims that gamma exposure caused observed price
  behavior.

This is the most defensible standard for interviews.

## Timing Convention

This is the most important methodological rule in the project.

The local options table is daily. Unless the exact capture timestamp is
verified, the design must assume that the options state for trading day `t`
is only safely usable after day `t` is complete.

Therefore:

- the main descriptive and predictive studies will use options state on day
  `t` to explain behavior on trading day `t + 1`,
- same-day charts will be exploratory only,
- same-day analyses will be clearly labeled as timing-contaminated.

This protects the project from lookahead bias.

Lookahead bias is the use of information that would not have been available at
the time of prediction or explanation.

## Exposure Definition

Version `1` must choose one explicit gamma exposure convention and apply it
consistently throughout the pipeline.

Proposed default:

`gamma_exposure = open_interest * contract_multiplier * spot_price^2 * gamma`

Definitions:

- `open_interest` is the number of open contracts,
- `contract_multiplier` is the number of shares per option contract,
- `spot_price` is the `SPY` end-of-day close on `trade_date`, derived from the
  final intraday bar in `firstrate.etfs` and used as the daily scaling price,
- `gamma` is the option gamma from the data source.

`Contract multiplier` will default to `100` for standard US ETF options in
version `1`.

The project must document this scaling convention clearly because exposure
definitions differ across vendors and practitioners.

The report must include a sensitivity section that compares the default
exposure convention with at least one simpler alternative, such as dropping the
`spot_price^2` term or using absolute rather than signed aggregation.

## Data Cleaning Rules

The contract cleaning layer should apply deterministic rules and preserve
diagnostics about excluded rows.

Initial cleaning rules:

- keep `SPY` rows only,
- drop rows with missing `trade_date`, `expiry_date`, `strike_price`, or
  `option_type`,
- drop rows with non-positive `open_interest` when computing open-interest
  exposure,
- flag zero or suspiciously constant gamma values,
- flag clearly invalid bid-ask states,
- preserve counts of filtered rows by date and filter reason.

The project should avoid excessive cleaning complexity in version `1`.
The purpose is to remove obvious bad inputs without turning the pipeline into a
data-cleaning thesis.

## Core Outputs of the Exposure Engine

The exposure engine must produce three granularities of output.

### Contract-Level Output

Each cleaned option row should include:

- contract identifiers,
- raw Greeks,
- chosen exposure measure,
- moneyness descriptors,
- days-to-expiry descriptors,
- any filter flags needed for diagnostics.

### Aggregated Market-State Outputs

For each `trade_date`, compute:

- strike-level signed and absolute gamma exposure,
- expiry-level signed and absolute gamma exposure,
- strike-by-expiry exposure grid,
- call-side and put-side exposure totals.

### Daily Summary Factors

For each `trade_date`, compute interpretable factors such as:

- `net_gamma_exposure`,
- `absolute_gamma_exposure`,
- `near_spot_gamma_share`,
- `front_expiry_gamma_share`,
- `largest_positive_gamma_strike_distance`,
- `largest_negative_gamma_strike_distance`,
- `call_put_gamma_imbalance`,
- `exposure_concentration_index`.

The factor set should remain small and explainable.

## Definitions for Derived Features

### Near-Spot Concentration

This factor measures how much absolute gamma exposure sits near the current
underlying price.

Version `1` should define a fixed moneyness band around spot, for example a
percentage band. The default version `1` band will be `plus or minus 2
percent`, and the factor will be the share of absolute gamma exposure inside
that band.

### Front-Expiry Share

This factor measures the share of total absolute gamma exposure that belongs to
the nearest expiry bucket.

### Gamma Node Distance

This factor measures the distance between spot and the strike with the largest
local gamma mass. The project may keep separate distances for the strongest
positive node and strongest negative node.

### Exposure Concentration Index

This factor summarizes whether gamma exposure is broadly distributed or highly
concentrated in a few strike-expiry cells.

Version `1` will use the `Herfindahl-Hirschman Index`, abbreviated `HHI`,
computed from strike-expiry cell shares of absolute gamma exposure.

## Intraday Response Metrics

The intraday behavior engine must convert `SPY` intraday bars into daily
response variables aligned to `t + 1`.

### Realized Variance

Realized variance is the sum of squared intraday log returns within a trading
day.

Version `1` should compute:

- next-day realized variance,
- next-day realized volatility as the square root of realized variance,
- next-day open-to-close absolute return,
- next-day close-to-close absolute return.

### Abnormal Volume

Abnormal volume should compare observed intraday volume with a baseline.

A simple first definition is:

- compute average volume by minute-of-day over a trailing window,
- use a trailing window of `20` trading days in version `1`,
- scale next-day minute volume by that baseline,
- aggregate burst behavior into a daily score or indicator.

### Pinning-Style Behavior

Pinning-style behavior describes whether the close finishes unusually near a
strike that has large nearby option positioning.

Version `1` should implement a practical proxy:

- identify the top `5` strikes by absolute same-day gamma exposure,
- find the nearest of those candidate strikes to the next-day close,
- measure end-of-day distance from close to that node,
- normalize by price level or strike spacing,
- summarize whether high-concentration days finish closer to those strikes.

The report must describe this as a proxy rather than as a direct observation of
dealer hedging.

## Architecture

The system should be organized into six layers.

### Layer 1: Data Access

Purpose:
read local `ClickHouse` tables and return typed tabular data.

Responsibilities:

- query `firstrate.options`,
- query `firstrate.etfs`,
- expose date-window and symbol filters,
- centralize connection management,
- support small-sample development runs and larger production runs.

### Layer 2: Canonical Daily Options Snapshot

Purpose:
build the clean option state for a single trading date.

Responsibilities:

- load option contracts for one date,
- attach spot reference values,
- compute contract-level exposure fields,
- apply cleaning rules,
- preserve diagnostics.

### Layer 3: Exposure Engine

Purpose:
convert the canonical option snapshot into aggregated exposure maps and daily
factors.

Responsibilities:

- aggregate by strike,
- aggregate by expiry,
- build strike-expiry grids,
- compute summary factors.

### Layer 4: Intraday Behavior Engine

Purpose:
convert intraday `SPY` bars into next-day response variables.

Responsibilities:

- compute realized variance,
- compute abnormal volume scores,
- compute pinning-style metrics,
- return a daily response table keyed by date.

### Layer 5: Research Layer

Purpose:
run descriptive studies and the predictive appendix.

Responsibilities:

- quantile sorts,
- event studies,
- regime splits,
- baseline comparisons,
- walk-forward predictive evaluation.

### Layer 6: Reporting Layer

Purpose:
turn analytical outputs into interview-ready artifacts.

Responsibilities:

- self-contained HTML report,
- reusable interactive charts,
- thin `Streamlit` app.

## Recommended Technology Stack

- `Python` for the full implementation.
- `ClickHouse` for source data.
- `Polars` for the main analytical pipeline.
- `Plotly` for charts.
- `Streamlit` for the exploration app.
- `pytest` for testing.

`Polars` is preferred over `pandas` because the workload is analytical, the
data is potentially large, and the project benefits from a fast columnar
workflow.

## Proposed Project Structure

```text
gamma_exposure_engine/
    app/
    data/
    exposure/
    intraday/
    research/
    reporting/
    outputs/
    tests/
```

Subdirectory responsibilities:

- `app/`: thin `Streamlit` interface.
- `data/`: ClickHouse connection and query logic.
- `exposure/`: cleaning, exposure formulas, and aggregation.
- `intraday/`: realized behavior metrics.
- `research/`: event studies, quantile analyses, and predictive appendix.
- `reporting/`: HTML report generation and chart assembly.
- `outputs/`: generated reports and local artifacts.
- `tests/`: mathematical, alignment, and smoke tests.

## Analysis Plan

The analysis should proceed in this order.

### Step 1: Data Validation

Validate:

- date coverage,
- symbol coverage,
- field completeness,
- spot alignment,
- trading-day continuity,
- obvious anomalies in Greeks and open interest.

### Step 2: Descriptive Exposure Diagnostics

Produce:

- time series of aggregate gamma measures,
- example strike maps for selected dates,
- expiry term structures,
- summary statistics for the factor set.

### Step 3: Intraday Response Diagnostics

Produce:

- distributions of realized volatility,
- distributions of abnormal volume,
- distributions of pinning metrics,
- regime summaries across calm and volatile periods.

### Step 4: Association Studies

Main descriptive studies:

- quantile sorts by gamma features,
- conditional averages of next-day realized behavior,
- event studies around extreme exposure dates,
- cross-tab studies by concentration and expiry structure.

### Step 5: Predictive Appendix

Use a small and disciplined model set:

- linear regression for continuous responses,
- logistic regression for regime classification.

The appendix must remain secondary to the descriptive core.

## Predictive Evaluation Standard

The predictive appendix should follow quant-research conventions.

Rules:

- use walk-forward splits only,
- never shuffle time order,
- compare against naive baselines,
- report out-of-sample results only,
- favor interpretability over model complexity.

Baseline examples:

- lagged realized volatility,
- rolling historical volatility,
- lagged absolute returns.

The predictive appendix succeeds only if it adds value beyond those simple
benchmarks.

## Reporting Deliverables

### HTML Research Report

The report should be the primary deliverable.

It should include:

- project motivation,
- data description,
- timing convention,
- exact exposure formulas,
- cleaning diagnostics,
- descriptive charts,
- association study results,
- robustness checks,
- predictive appendix,
- limitations,
- extension ideas.

The report must be self-contained and easy to send or show during an interview.

### Thin Exploration App

The app is a secondary deliverable.

It should allow a user to:

- select a date,
- inspect strike-level and expiry-level gamma maps,
- inspect daily summary factors,
- view selected event windows,
- compare exploratory same-day views with the formal next-day framing.

The app should remain simple. It exists to help discussion, not to replace the
report.

## Testing Strategy

Testing should focus on failure modes that matter.

### Unit Tests

Required unit tests:

- exposure aggregation arithmetic,
- signed versus absolute aggregation,
- factor construction edge cases,
- realized variance calculations,
- date alignment and `t -> t + 1` joins.

### Integration Tests

Required integration tests:

- a small `ClickHouse` extraction over a narrow date window,
- a short pipeline run that builds a research dataset,
- a smoke test that generates a report for a small sample.

The tests should be practical rather than exhaustive.

## Robustness Checks

The report should include at least these robustness studies:

- alternative near-spot band widths,
- alternative exposure scaling conventions,
- exclusion of low open-interest contracts,
- exclusion of suspicious zero-gamma rows,
- alternative pinning definitions,
- regime splits by market volatility state.

If the headline effect disappears under all reasonable perturbations, that is an
important result and should be reported honestly.

## Main Risks

### Timing Risk

The daily snapshot may reflect end-of-day state. This is why the project uses
next-day response variables in the main study.

### Exposure Definition Risk

There is no single universal gamma exposure formula in practice.
The design addresses this by fixing one primary definition and explicitly
testing sensitivity.

### Data Quality Risk

Zero or stale Greeks, low open interest, and strike gaps may distort results.
The design addresses this through explicit cleaning diagnostics.

### Narrative Risk

It is easy to claim too much from descriptive evidence.
The design addresses this by using a mechanism-informed but non-causal framing.

## Version 1 Success Criteria

Version `1` is complete when:

- the pipeline reads `SPY` options and intraday bars from `ClickHouse`,
- the pipeline computes daily gamma maps and summary factors,
- the pipeline computes next-day intraday response metrics,
- the report documents formulas, timing, diagnostics, and results,
- the app allows simple date-driven exploration,
- tests confirm the key arithmetic and alignment logic,
- the predictive appendix compares fairly against naive baselines.

## Interview Questions the Project Must Answer Well

The final artifact must let the author answer these questions cleanly:

- What exact gamma exposure formula did you use?
- Why did you use `t + 1` rather than same-day responses?
- How did you avoid lookahead bias?
- How did you define pinning-style behavior?
- Which robustness checks mattered most?
- Did your predictive features beat simple historical-volatility baselines?
- What are the strongest limitations of your result?

If the project cannot answer those questions directly, it is not interview-ready.

## Future Extensions

Good version `2` directions:

- add `SPX` once intraday alignment is verified,
- compare `SPY` and `SPX` exposure behavior,
- extend from gamma-only to a multi-Greek positioning view,
- test cross-sectional behavior on a small equity universe,
- build a more advanced frontend after the research result is credible.

These should only happen after version `1` is clean.

## Final Recommendation

Build the full stack in one implementation cycle, but allocate effort in this
priority order:

1. research dataset and timing discipline,
2. exposure engine and intraday response metrics,
3. descriptive studies and robustness checks,
4. predictive appendix,
5. thin frontend.

This ordering maximizes interview value because it optimizes for research
judgment rather than feature count.
