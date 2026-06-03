"""Tests for offline-only raw data loaders."""

from __future__ import annotations

from datetime import date, datetime
from pathlib import Path

import polars as pl
import pytest

from gamma_exposure_engine.data.raw_store import load_raw_intraday_bars
from gamma_exposure_engine.data.raw_store import load_raw_options_snapshot


def test_load_raw_intraday_bars_reads_only_local_parquet_and_filters_range(
    tmp_path: Path,
) -> None:
    """Intraday loader should read local files and filter by requested dates."""

    raw_dir = tmp_path / "data" / "raw"
    raw_dir.mkdir(parents=True)
    intraday_path = raw_dir / "SPY_intraday_bars.parquet"
    pl.DataFrame(
        {
            "symbol": ["SPY", "SPY", "SPY"],
            "ts": [
                datetime(2024, 1, 2, 9, 30),
                datetime(2024, 1, 3, 9, 30),
                datetime(2024, 1, 4, 9, 30),
            ],
            "open": [100.0, 101.0, 102.0],
            "high": [101.0, 102.0, 103.0],
            "low": [99.0, 100.0, 101.0],
            "close": [100.5, 101.5, 102.5],
            "volume": [10.0, 11.0, 12.0],
        }
    ).write_parquet(intraday_path)

    loaded = load_raw_intraday_bars(
        symbol="SPY",
        start_date="2024-01-03",
        end_date="2024-01-04",
        raw_data_dir=raw_dir,
    )

    assert loaded.height == 2
    assert loaded.select(pl.col("ts").dt.date()).to_series().to_list() == [
        date(2024, 1, 3),
        date(2024, 1, 4),
    ]


def test_load_raw_options_snapshot_reads_only_local_parquet_and_filters_range(
    tmp_path: Path,
) -> None:
    """Options loader should read local files and filter by requested dates."""

    raw_dir = tmp_path / "data" / "raw"
    raw_dir.mkdir(parents=True)
    options_path = raw_dir / "SPY_options_snapshot.parquet"
    pl.DataFrame(
        {
            "symbol": ["SPY", "SPY", "SPY"],
            "trade_date": [date(2024, 1, 2), date(2024, 1, 3), date(2024, 1, 4)],
            "strike_price": [100.0, 101.0, 102.0],
            "expiry_date": [date(2024, 1, 19), date(2024, 1, 19), date(2024, 1, 19)],
            "option_type": ["c", "c", "p"],
            "last_price": [2.0, 2.1, 2.2],
            "bid": [1.9, 2.0, 2.1],
            "ask": [2.0, 2.1, 2.2],
            "bid_iv": [0.2, 0.2, 0.2],
            "ask_iv": [0.21, 0.21, 0.21],
            "open_interest": [100, 110, 120],
            "volume": [10, 11, 12],
            "delta": [0.5, 0.5, 0.5],
            "gamma": [0.01, 0.01, 0.01],
            "vega": [0.1, 0.1, 0.1],
            "theta": [-0.01, -0.01, -0.01],
            "rho": [0.01, 0.01, 0.01],
        }
    ).write_parquet(options_path)

    loaded = load_raw_options_snapshot(
        symbol="SPY",
        start_date="2024-01-03",
        end_date="2024-01-04",
        raw_data_dir=raw_dir,
    )

    assert loaded.height == 2
    assert loaded["trade_date"].to_list() == [date(2024, 1, 3), date(2024, 1, 4)]


def test_load_raw_intraday_bars_fails_with_actionable_message_when_missing(
    tmp_path: Path,
) -> None:
    """Missing intraday file should fail with guidance that stays offline-first."""

    raw_dir = tmp_path / "data" / "raw"
    raw_dir.mkdir(parents=True)

    with pytest.raises(RuntimeError, match="SPY_intraday_bars.parquet") as error:
        load_raw_intraday_bars(
            symbol="SPY",
            start_date="2024-01-02",
            end_date="2024-01-03",
            raw_data_dir=raw_dir,
        )

    assert "refresh-raw-cache" in str(error.value)


def test_load_raw_options_snapshot_fails_when_required_column_missing(
    tmp_path: Path,
) -> None:
    """Malformed options parquet should fail fast with missing-column details."""

    raw_dir = tmp_path / "data" / "raw"
    raw_dir.mkdir(parents=True)
    options_path = raw_dir / "SPY_options_snapshot.parquet"
    pl.DataFrame(
        {
            "symbol": ["SPY"],
            "trade_date": [date(2024, 1, 2)],
            "strike_price": [100.0],
        }
    ).write_parquet(options_path)

    with pytest.raises(RuntimeError, match="missing required columns"):
        load_raw_options_snapshot(
            symbol="SPY",
            start_date="2024-01-02",
            end_date="2024-01-02",
            raw_data_dir=raw_dir,
        )
