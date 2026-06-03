"""Offline-only raw-data loaders for canonical Parquet inputs.

This module is the only data entrypoint for normal analysis execution. It reads
local Parquet files from ``data/raw`` and never reaches for ClickHouse.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path

import polars as pl

from gamma_exposure_engine.settings import load_settings

INTRADAY_DATASET_NAME: str = "intraday_bars"
OPTIONS_DATASET_NAME: str = "options_snapshot"

INTRADAY_REQUIRED_COLUMNS: tuple[str, ...] = (
    "symbol",
    "ts",
    "open",
    "high",
    "low",
    "close",
    "volume",
)
OPTIONS_REQUIRED_COLUMNS: tuple[str, ...] = (
    "symbol",
    "trade_date",
    "strike_price",
    "expiry_date",
    "option_type",
    "last_price",
    "bid",
    "ask",
    "bid_iv",
    "ask_iv",
    "open_interest",
    "volume",
    "delta",
    "gamma",
    "vega",
    "theta",
    "rho",
)


@dataclass(frozen=True)
class RawDataPaths:
    """Resolved canonical local paths for intraday and options raw files."""

    intraday_path: Path
    options_path: Path


def resolve_raw_data_paths(
    symbol: str,
    raw_data_dir: Path | None = None,
) -> RawDataPaths:
    """Build canonical raw-data paths for one symbol.

    Parameters
    ----------
    symbol:
        Underlying symbol used in canonical filenames.
    raw_data_dir:
        Optional directory override for tests and local experiments.

    Returns
    -------
    RawDataPaths
        Filesystem paths for intraday bars and options snapshots.
    """

    if raw_data_dir is None:
        settings = load_settings(require_clickhouse_password=False)
        resolved_raw_dir = settings.raw_data.raw_data_dir
    else:
        resolved_raw_dir = raw_data_dir
    return RawDataPaths(
        intraday_path=resolved_raw_dir / f"{symbol}_{INTRADAY_DATASET_NAME}.parquet",
        options_path=resolved_raw_dir / f"{symbol}_{OPTIONS_DATASET_NAME}.parquet",
    )


def load_raw_intraday_bars(
    symbol: str,
    start_date: str,
    end_date: str,
    raw_data_dir: Path | None = None,
) -> pl.DataFrame:
    """Load intraday bars from local canonical raw Parquet.

    Parameters
    ----------
    symbol:
        Underlying symbol used in canonical filenames.
    start_date:
        Inclusive ISO-8601 start date for filtering.
    end_date:
        Inclusive ISO-8601 end date for filtering.
    raw_data_dir:
        Optional directory override for tests and local experiments.

    Returns
    -------
    pl.DataFrame
        Intraday bars filtered to the requested date range.
    """

    paths = resolve_raw_data_paths(symbol=symbol, raw_data_dir=raw_data_dir)
    frame = _load_required_raw_file(
        file_path=paths.intraday_path,
        dataset_name=INTRADAY_DATASET_NAME,
        required_columns=INTRADAY_REQUIRED_COLUMNS,
    )
    return _filter_inclusive_date_range(
        frame=frame,
        date_column=pl.col("ts").dt.date(),
        start_date=start_date,
        end_date=end_date,
    )


def load_raw_options_snapshot(
    symbol: str,
    start_date: str,
    end_date: str,
    raw_data_dir: Path | None = None,
) -> pl.DataFrame:
    """Load options snapshots from local canonical raw Parquet.

    Parameters
    ----------
    symbol:
        Underlying symbol used in canonical filenames.
    start_date:
        Inclusive ISO-8601 start date for filtering.
    end_date:
        Inclusive ISO-8601 end date for filtering.
    raw_data_dir:
        Optional directory override for tests and local experiments.

    Returns
    -------
    pl.DataFrame
        Options snapshots filtered to the requested date range.
    """

    paths = resolve_raw_data_paths(symbol=symbol, raw_data_dir=raw_data_dir)
    frame = _load_required_raw_file(
        file_path=paths.options_path,
        dataset_name=OPTIONS_DATASET_NAME,
        required_columns=OPTIONS_REQUIRED_COLUMNS,
    )
    return _filter_inclusive_date_range(
        frame=frame,
        date_column="trade_date",
        start_date=start_date,
        end_date=end_date,
    )


def _filter_inclusive_date_range(
    frame: pl.DataFrame,
    date_column: str | pl.Expr,
    start_date: str,
    end_date: str,
) -> pl.DataFrame:
    """Keep rows whose date expression falls inside an inclusive ISO range."""

    inclusive_start = date.fromisoformat(start_date)
    inclusive_end = date.fromisoformat(end_date)
    date_expression = pl.col(date_column) if isinstance(date_column, str) else date_column
    return frame.filter(
        date_expression.is_between(
            inclusive_start,
            inclusive_end,
            closed="both",
        )
    )


def _load_required_raw_file(
    file_path: Path,
    dataset_name: str,
    required_columns: tuple[str, ...],
) -> pl.DataFrame:
    """Load one required raw file and validate required columns.

    Raises
    ------
    RuntimeError
        If the file is missing or malformed for offline analysis.
    """

    if not file_path.exists():
        raise RuntimeError(_build_missing_raw_file_message(file_path, dataset_name))

    frame = pl.read_parquet(file_path)
    missing_columns = [
        column_name
        for column_name in required_columns
        if column_name not in frame.columns
    ]
    if missing_columns:
        missing_columns_text = ", ".join(missing_columns)
        raise RuntimeError(
            f"Raw dataset {dataset_name!r} at {file_path} is missing required "
            f"columns: {missing_columns_text}.",
        )

    return frame


def _build_missing_raw_file_message(file_path: Path, dataset_name: str) -> str:
    """Build an actionable missing-file message for offline users."""

    return (
        f"Offline analysis requires {dataset_name!r} at {file_path}, but the "
        "file is missing. Place canonical raw files under data/raw or run "
        "`uv run gex refresh-raw-cache --start YYYY-MM-DD --end YYYY-MM-DD` "
        "once on a machine with ClickHouse access."
    )
