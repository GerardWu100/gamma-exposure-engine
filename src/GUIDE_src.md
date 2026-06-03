# GUIDE_src

`src/` contains the importable Python package for the offline-first gamma exposure
research pipeline.

## Structure

- `src/gamma_exposure_engine/`: importable package root.
- `src/gamma_exposure_engine/cli.py`: command-line entrypoint (`gex`) for offline
  run and optional refresh.
- `src/gamma_exposure_engine/settings.py`: typed configuration and environment
  loading.
- `config.toml` (repo root): non-secret defaults.
- `src/gamma_exposure_engine/data/`: raw-data loaders and optional ClickHouse
  refresh path.
- `src/gamma_exposure_engine/pipeline/`: linear offline orchestration and artifact
  writing.
- `src/gamma_exposure_engine/exposure/`: options cleaning and gamma factor
  construction.
- `src/gamma_exposure_engine/intraday/`: intraday response metrics.
- `src/gamma_exposure_engine/research/`: descriptive, inferential, regime,
  robustness, and predictive analysis helpers.

## Runtime Contract

- Normal execution path is offline and local-only.
- Canonical raw inputs live under `data/raw/`.
- ClickHouse is isolated to optional refresh code in
  `src/gamma_exposure_engine/data/raw_cache_builder.py`.
