# Gamma Exposure Engine

[![CI](https://github.com/frenzied-org/gamma-exposure-engine/actions/workflows/ci.yml/badge.svg)](https://github.com/frenzied-org/gamma-exposure-engine/actions/workflows/ci.yml)

**What does options dealer positioning tell us about next-day intraday market behavior?**

This repository is an offline-first, interview-defensible quantitative finance
research project for `SPY`. The main artifact is a teaching notebook that
walks from local raw Parquet inputs to descriptive, inferential, robustness,
regime, and predictive results.

## Hiring Narrative

The project studies whether daily options gamma structure is associated with
next-day realized variance, intraday return magnitude, volume anomalies, and
pinning-like behavior. It is intentionally transparent:

- one symbol (`SPY`)
- one canonical raw-data contract (`data/raw/`)
- one reproducible offline CLI path
- one teaching notebook (`notebooks/gamma_exposure_pipeline_demo.ipynb`)

This is an empirical association project, not a causal inference claim.

## Offline Runtime Contract

Normal usage requires only these committed files:

- `data/raw/SPY_intraday_bars.parquet`
- `data/raw/SPY_options_snapshot.parquet`
- `data/raw/manifest.json`

Offline analysis never connects to ClickHouse, never requires `.env`, and fails
fast if raw files are missing.

See `docs/reference/offline-data-contract.md` for required columns and coverage.

## Quick Start (Offline)

Run the offline pipeline against the committed demo range:

```bash
uv run gex run-offline-analysis --start 2024-01-02 --end 2024-01-31 --output-dir outputs/demo
```

Run tests:

```bash
uv run pytest -v
```

Execute the teaching notebook end-to-end:

```bash
uv run jupyter nbconvert --to notebook --execute notebooks/gamma_exposure_pipeline_demo.ipynb --output gamma_exposure_pipeline_demo.executed.ipynb
```

## Optional Raw Refresh (Maintenance Only)

If you have ClickHouse access and want to refresh or expand the demo payload:

```bash
uv run gex refresh-raw-cache --start 2024-01-02 --end 2024-12-31
```

This command is optional and separate from normal execution.

See `docs/reference/clickhouse-raw-cache-refresh.md` for details.

## Offline Outputs

`gex run-offline-analysis` writes simple, non-HTML artifacts under the chosen
output directory, including:

- aligned research dataset (`research_dataset.parquet`)
- quantile summary (`quantile_summary.csv`)
- statistical tests (`statistical_tests.csv`)
- regime summary (`regime_summary.csv`)
- predictive comparison (`predictive_comparison.csv`)
- robustness, diagnostics, and correlation tables (`.csv`)
- run manifest (`run_manifest.json`)

These files are the backbone for interview walkthroughs and future blog writing.

## Notebook Teaching Flow

`notebooks/gamma_exposure_pipeline_demo.ipynb` is the human-facing centerpiece.
It teaches the full offline pipeline:

1. inspect raw local data contract
2. build spot close and clean options
3. construct gamma factors and response metrics
4. align day `t` exposures with day `t+1` responses
5. run descriptive and inferential summaries
6. run regime, robustness, and predictive checks
7. interpret findings and limitations honestly

## Interview Explanation Tips

A concise interview explanation can follow this structure:

1. **Question:** does gamma positioning relate to next-day market behavior?
2. **Data contract:** two local raw files, fixed schema, fixed date window.
3. **Method:** clean -> aggregate gamma factors -> align `t` to `t+1` -> analyze.
4. **Evidence:** quantile means, non-parametric tests, regime splits,
   robustness checks, and walk-forward prediction baselines.
5. **Limitations:** association only, sample-window dependence, single symbol,
   sensitivity to options cleaning conventions.

## Project Layout

- `src/gamma_exposure_engine/data/`: local raw loaders and optional ClickHouse refresh tooling
- `src/gamma_exposure_engine/exposure/`: options cleaning and gamma factor construction
- `src/gamma_exposure_engine/intraday/`: realized variance and response metrics
- `src/gamma_exposure_engine/research/`: descriptive, inferential, robustness, regime, predictive modules
- `src/gamma_exposure_engine/pipeline/`: linear offline orchestration
- `src/gamma_exposure_engine/cli.py`: Typer commands for offline run and optional refresh
- `config.toml`: runtime defaults for paths, research knobs, and raw-data contract
- `docs/reference/`: hard-contract documentation
- `notebooks/`: teaching notebook
- `tests/`: unit and smoke coverage
