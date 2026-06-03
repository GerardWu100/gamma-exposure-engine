# GUIDE_ROOT

## Part 1: Conceptual Explanation

This repository is an offline-first gamma exposure research project for `SPY`.
Its purpose is to support interview explanation and reproducible local analysis,
not to run a dashboard or HTML reporting app.

The architecture has two explicit data paths:

1. **Required offline path**
   - reads canonical Parquet files from `data/raw/`
   - runs full research pipeline locally
   - writes non-HTML artifacts (Parquet, CSV, JSON)
   - powers the teaching notebook

2. **Optional refresh path**
   - uses ClickHouse one-time to refresh `data/raw/`
   - is isolated to `src/gamma_exposure_engine/data/raw_cache_builder.py`
   - is never required for normal analysis

The command-line boundary is `src/gamma_exposure_engine/cli.py` with two commands:

- `gex run-offline-analysis`: required local runtime
- `gex refresh-raw-cache`: optional maintenance refresh

The core research flow is linear and interview-friendly:

1. load local intraday bars and options snapshots
2. construct daily spot close and clean options rows
3. build daily gamma factors and intraday response metrics
4. align day-`t` exposures with day-`t+1` responses
5. run descriptive, inferential, regime, robustness, and predictive modules
6. write simple artifacts to the chosen output directory

The main human-facing artifact is
`notebooks/gamma_exposure_pipeline_demo.ipynb`, which teaches this full path
from raw Parquet contract to interpretation and limitations.

## Part 2: Code Reference

- `README.md`: user-level project narrative and run instructions.
- `.github/workflows/ci.yml`: CI workflow running `uv sync --frozen` and
  `uv run pytest -v`.
- `pyproject.toml`: dependency declarations and `gex` entrypoint.
- `uv.lock`: reproducible dependency lockfile.
- `config.toml`: non-secret defaults for paths, research knobs, ClickHouse, and
  raw-data contract.
- `src/gamma_exposure_engine/settings.py`: typed settings loader from config and
  `.env`.
- `src/gamma_exposure_engine/cli.py`: Typer commands for offline analysis and
  optional refresh.
- `src/gamma_exposure_engine/data/raw_store.py`: offline-only canonical Parquet
  loaders.
- `src/gamma_exposure_engine/data/raw_cache_builder.py`: optional ClickHouse
  refresh writer for canonical raw files.
- `src/gamma_exposure_engine/data/clickhouse_client.py`: ClickHouse client and
  result conversion.
- `src/gamma_exposure_engine/data/intraday_queries.py`: SQL and extraction helper
  for intraday bars.
- `src/gamma_exposure_engine/data/options_queries.py`: SQL and extraction helper
  for options snapshots.
- `src/gamma_exposure_engine/pipeline/offline_pipeline.py`: linear orchestration
  and artifact writes.
- `src/gamma_exposure_engine/exposure/`: options cleaning and gamma aggregation.
- `src/gamma_exposure_engine/intraday/`: intraday response metrics.
- `src/gamma_exposure_engine/research/`: descriptive, inferential, regime,
  robustness, predictive modules.
- `docs/reference/offline-data-contract.md`: canonical local input contract.
- `docs/reference/clickhouse-raw-cache-refresh.md`: optional refresh procedure.
- `notebooks/gamma_exposure_pipeline_demo.ipynb`: teaching walkthrough notebook.
- `tests/`: regression tests for settings, data paths, pipeline, and CLI.

## Part 3: Short Journal

- 2026-04-19: Refactored repository to offline-first and notebook-led runtime.
  Removed Streamlit and HTML reporting surfaces, promoted `data/raw/` as the
  canonical raw location, added explicit refresh-vs-offline command split,
  introduced linear offline pipeline module, and rewrote docs around interview
  narrative and local reproducibility.
- 2026-05-20: Moved implementation under `src/gamma_exposure_engine/` package
  layout and promoted `config.toml` to the repository root.
