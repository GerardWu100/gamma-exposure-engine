# Gamma Exposure Engine

Offline research project that asks whether the daily options gamma structure
of `SPY` is associated with next-day intraday market behavior. It ships a
fixed demo dataset, a CLI pipeline, and a teaching notebook that walks the
full analysis end to end.

## What it does

The engine takes local Parquet snapshots of SPY intraday bars and SPY options
chains, builds daily open-interest-weighted gamma-mass factors (the raw data
has no dealer/owner position sign, so this is unsigned exposure, not signed
dealer gamma), and aligns day `t` exposure with day `t+1` market response
(realized variance, return magnitude, volume anomalies, pinning-like
behavior near round strikes).

It then runs:

- descriptive statistics by exposure quantile
- non-parametric inferential tests
- volatility-regime splits
- robustness checks across alternate near-spot bands
- a walk-forward Ridge regression predictive baseline

This is an empirical association study, not a causal claim. See
`docs/reference/offline-data-contract.md` for the exact input schema and
`GUIDE_ROOT.md` for the full pipeline walkthrough.

## Requirements

- Python >= 3.13
- No external services for normal use. Offline analysis reads only local
  Parquet files under `data/raw/` and never touches ClickHouse or `.env`.
- Optional: a local ClickHouse instance, only if you want to refresh the raw
  demo data (see Usage below). Credentials for that path are read from `.env`
  as `CLICKHOUSE_USER` and `CLICKHOUSE_PASSWORD`; `CLICKHOUSE_HOST`,
  `CLICKHOUSE_PORT`, `CLICKHOUSE_SECURE`, and `CLICKHOUSE_VERIFY` can override
  the defaults in `config.toml`. See `.env.example`.

## Setup

```bash
uv sync
```

## Usage

Run the offline pipeline against the committed demo range:

```bash
uv run gex run-offline-analysis --start 2024-01-02 --end 2024-01-31 --output-dir outputs/demo
```

Run tests:

```bash
uv run pytest -v
```

Execute the teaching notebook end to end:

```bash
uv run jupyter nbconvert --to notebook --execute notebooks/gamma_exposure_pipeline_demo.ipynb --output gamma_exposure_pipeline_demo.executed.ipynb
```

Optional maintenance-only step, if you have ClickHouse access and want to
refresh or expand the raw demo data:

```bash
uv run gex refresh-raw-cache --start 2024-01-02 --end 2024-12-31
```

See `docs/reference/clickhouse-raw-cache-refresh.md` for details.

## Configuration

`config.toml` holds non-secret runtime defaults. The knobs most worth knowing:

- `research.default_factor_name` / `research.default_target_name`: the
  exposure factor and next-day response column used when the CLI flags are
  omitted.
- `research.near_spot_band_width` and `research.robustness_band_widths`: the
  moneyness band used for the near-spot factor and the alternates checked in
  the robustness table.
- `research.bootstrap_iterations` / `research.bootstrap_confidence_level`:
  bootstrap resampling settings for confidence intervals.
- `clickhouse.*`: non-secret connection defaults; credentials live in `.env`.

## Layout

- `src/gamma_exposure_engine/data/`: local raw loaders and the optional
  ClickHouse refresh tool
- `src/gamma_exposure_engine/exposure/`: options cleaning and gamma factor
  construction
- `src/gamma_exposure_engine/intraday/`: realized variance and response
  metrics
- `src/gamma_exposure_engine/research/`: descriptive, inferential,
  robustness, regime, and predictive modules
- `src/gamma_exposure_engine/pipeline/`: linear offline orchestration
- `src/gamma_exposure_engine/cli.py`: Typer commands for offline run and
  optional refresh
- `notebooks/`: the teaching notebook
- `docs/reference/`: the raw-data contract and refresh procedure
- `tests/`: unit and smoke coverage

## Output

`gex run-offline-analysis` writes plain Parquet/CSV/JSON artifacts to the
chosen output directory:

- `research_dataset.parquet`: aligned day-`t` exposure and day-`t+1` response
- `quantile_summary.csv`, `statistical_tests.csv`: descriptive and inferential
  results
- `regime_summary.csv`: results split by volatility regime
- `predictive_comparison.csv`: walk-forward predictive baseline comparison
- robustness, diagnostics, and correlation tables (`.csv`)
- `run_manifest.json`: run metadata

## License

All rights reserved. See [LICENSE](LICENSE).
