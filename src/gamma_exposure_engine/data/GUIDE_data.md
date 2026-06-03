# GUIDE_data

`src/gamma_exposure_engine/data/` defines two explicit data paths.

## Required Offline Path

- `raw_store.py` loads canonical Parquet files from `data/raw/`.
- These loaders are used by normal offline CLI and notebook execution.
- Missing or malformed files fail with actionable local guidance.

## Optional Refresh Path

- `raw_cache_builder.py` is the only module allowed to refresh local raw files
  from ClickHouse.
- `intraday_queries.py` and `options_queries.py` are ClickHouse extract
  helpers used only by the refresh path.
- `clickhouse_client.py` builds a configured client and converts query output
  to Polars.

## Design Rule

If code is part of normal analysis runtime, it must not import or call
ClickHouse helpers.
