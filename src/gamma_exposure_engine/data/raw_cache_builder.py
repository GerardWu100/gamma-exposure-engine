"""Optional ClickHouse refresh workflow for canonical offline raw files.

This module is the only place in the codebase that is allowed to pull data
from ClickHouse for the normal project lifecycle. It extracts raw tables and
writes canonical Parquet files into ``data/raw``.
"""

from __future__ import annotations

import json
from pathlib import Path

import polars as pl

from gamma_exposure_engine.data.clickhouse_client import create_clickhouse_client
from gamma_exposure_engine.data.intraday_queries import (
    fetch_intraday_bars_from_clickhouse,
)
from gamma_exposure_engine.data.options_queries import (
    fetch_options_snapshot_from_clickhouse,
)
from gamma_exposure_engine.data.raw_store import resolve_raw_data_paths
from gamma_exposure_engine.settings import load_settings


def refresh_raw_cache_from_clickhouse(
    symbol: str,
    start_date: str,
    end_date: str,
    raw_data_dir: Path | None = None,
) -> dict[str, object]:
    """Refresh canonical raw Parquet files from ClickHouse.

    Parameters
    ----------
    symbol:
        Underlying symbol to refresh, for example ``SPY``.
    start_date:
        Inclusive ISO-8601 start date for extraction.
    end_date:
        Inclusive ISO-8601 end date for extraction.
    raw_data_dir:
        Optional destination override. Defaults to configured ``data/raw``.

    Returns
    -------
    dict[str, object]
        Run manifest with row counts, date range, and file sizes.
    """

    settings = load_settings()
    resolved_raw_dir = raw_data_dir or settings.raw_data.raw_data_dir
    resolved_raw_dir.mkdir(parents=True, exist_ok=True)
    raw_paths = resolve_raw_data_paths(symbol=symbol, raw_data_dir=resolved_raw_dir)

    client = create_clickhouse_client()
    intraday_frame = fetch_intraday_bars_from_clickhouse(
        client=client,
        symbol=symbol,
        start_date=start_date,
        end_date=end_date,
    )
    options_frame = fetch_options_snapshot_from_clickhouse(
        client=client,
        symbol=symbol,
        start_date=start_date,
        end_date=end_date,
    )

    intraday_frame.write_parquet(raw_paths.intraday_path)
    options_frame.write_parquet(raw_paths.options_path)

    manifest = build_raw_manifest(
        symbol=symbol,
        start_date=start_date,
        end_date=end_date,
        intraday_frame=intraday_frame,
        options_frame=options_frame,
        intraday_path=raw_paths.intraday_path,
        options_path=raw_paths.options_path,
        schema_version=settings.raw_data.schema_version,
    )
    manifest_path = resolved_raw_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def build_raw_manifest(
    symbol: str,
    start_date: str,
    end_date: str,
    intraday_frame: pl.DataFrame,
    options_frame: pl.DataFrame,
    intraday_path: Path,
    options_path: Path,
    schema_version: int,
) -> dict[str, object]:
    """Build a canonical raw-data manifest payload."""

    return {
        "schema_version": schema_version,
        "symbol": symbol,
        "date_range": {
            "start_date": start_date,
            "end_date": end_date,
        },
        "datasets": {
            "intraday_bars": {
                "file": intraday_path.name,
                "row_count": intraday_frame.height,
                "size_bytes": intraday_path.stat().st_size,
                "columns": intraday_frame.columns,
            },
            "options_snapshot": {
                "file": options_path.name,
                "row_count": options_frame.height,
                "size_bytes": options_path.stat().st_size,
                "columns": options_frame.columns,
            },
        },
        "note": "These Parquet files are the offline demo source of truth.",
    }
