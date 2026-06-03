# ClickHouse Raw Cache Refresh (Optional Maintenance)

This workflow is optional and should be run only when you want to refresh or
expand the committed demo raw dataset.

Normal project use does not require this document. The default runtime is
offline-only from `data/raw/`.

## Purpose

- Pull canonical `SPY` intraday and options snapshots from ClickHouse once.
- Write canonical Parquet files into `data/raw/`.
- Regenerate `data/raw/manifest.json` with row counts, file sizes, and range.

## Prerequisites

- ClickHouse network access
- `.env` contains `CLICKHOUSE_USER` and `CLICKHOUSE_PASSWORD`
- Optional: override host, port, secure, and verify in `.env`

## Command

```bash
uv run gex refresh-raw-cache --start 2024-01-02 --end 2024-12-31
```

Optional destination override:

```bash
uv run gex refresh-raw-cache --start 2024-01-02 --end 2024-12-31 --raw-dir data/raw
```

## Post-Refresh Checks

- Confirm these files exist:
  - `data/raw/SPY_intraday_bars.parquet`
  - `data/raw/SPY_options_snapshot.parquet`
  - `data/raw/manifest.json`
- Confirm manifest range and row counts match expectations.
- Keep combined raw payload comfortably under 100 MB for portability.

## Safety Boundary

- Only `src/data/raw_cache_builder.py` may touch ClickHouse for refresh.
- Offline analysis and notebook execution must remain independent from this
  refresh path.
