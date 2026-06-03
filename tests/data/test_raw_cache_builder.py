"""Tests for optional ClickHouse raw-cache refresh workflow."""

from __future__ import annotations

from datetime import date, datetime
from pathlib import Path

import polars as pl

from gamma_exposure_engine.data.raw_cache_builder import refresh_raw_cache_from_clickhouse


def test_refresh_raw_cache_from_clickhouse_writes_parquet_and_manifest(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """Refresh path should write canonical files and manifest to data/raw."""

    raw_dir = tmp_path / "data" / "raw"
    settings = _build_settings(tmp_path=tmp_path, raw_dir=raw_dir)
    fake_client = _FakeClient(
        intraday_frame=pl.DataFrame(
            {
                "symbol": ["SPY"],
                "ts": [datetime(2024, 1, 2, 9, 30)],
                "open": [100.0],
                "high": [101.0],
                "low": [99.0],
                "close": [100.5],
                "volume": [10.0],
            }
        ),
        options_frame=pl.DataFrame(
            {
                "symbol": ["SPY"],
                "trade_date": [date(2024, 1, 2)],
                "strike_price": [100.0],
                "expiry_date": [date(2024, 1, 19)],
                "option_type": ["c"],
                "last_price": [2.0],
                "bid": [1.9],
                "ask": [2.0],
                "bid_iv": [0.2],
                "ask_iv": [0.21],
                "open_interest": [100],
                "volume": [10],
                "delta": [0.5],
                "gamma": [0.01],
                "vega": [0.1],
                "theta": [-0.01],
                "rho": [0.01],
            }
        ),
    )

    monkeypatch.setattr(
        "gamma_exposure_engine.data.raw_cache_builder.load_settings",
        lambda: settings,
    )
    monkeypatch.setattr(
        "gamma_exposure_engine.data.raw_cache_builder.create_clickhouse_client",
        lambda: fake_client,
    )
    monkeypatch.setattr(
        "gamma_exposure_engine.data.raw_cache_builder.fetch_intraday_bars_from_clickhouse",
        lambda client, symbol, start_date, end_date: client.intraday_frame,
    )
    monkeypatch.setattr(
        "gamma_exposure_engine.data.raw_cache_builder.fetch_options_snapshot_from_clickhouse",
        lambda client, symbol, start_date, end_date: client.options_frame,
    )

    manifest = refresh_raw_cache_from_clickhouse(
        symbol="SPY",
        start_date="2024-01-02",
        end_date="2024-01-02",
    )

    intraday_path = raw_dir / "SPY_intraday_bars.parquet"
    options_path = raw_dir / "SPY_options_snapshot.parquet"
    manifest_path = raw_dir / "manifest.json"

    assert intraday_path.exists()
    assert options_path.exists()
    assert manifest_path.exists()
    assert manifest["schema_version"] == 7
    assert manifest["symbol"] == "SPY"
    assert manifest["datasets"]["intraday_bars"]["row_count"] == 1
    assert manifest["datasets"]["options_snapshot"]["row_count"] == 1


class _FakeRawDataSettings:
    def __init__(self, raw_data_dir: Path, schema_version: int) -> None:
        self.raw_data_dir = raw_data_dir
        self.schema_version = schema_version


class _FakeSettings:
    def __init__(self, raw_data_dir: Path, schema_version: int) -> None:
        self.raw_data = _FakeRawDataSettings(raw_data_dir, schema_version)


def _build_settings(tmp_path: Path, raw_dir: Path) -> _FakeSettings:
    _ = tmp_path
    return _FakeSettings(raw_data_dir=raw_dir, schema_version=7)


class _FakeClient:
    def __init__(
        self, intraday_frame: pl.DataFrame, options_frame: pl.DataFrame
    ) -> None:
        self.intraday_frame = intraday_frame
        self.options_frame = options_frame
