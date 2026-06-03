"""Tests for offline pipeline orchestration."""

from __future__ import annotations

from datetime import date, datetime
from pathlib import Path

import polars as pl
import pytest

from gamma_exposure_engine.pipeline.offline_pipeline import run_offline_analysis


def test_run_offline_analysis_writes_expected_non_html_artifacts(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """Offline pipeline should write Parquet, CSV, and JSON artifacts."""

    raw_dir = tmp_path / "data" / "raw"
    raw_dir.mkdir(parents=True)
    _write_intraday_raw(raw_dir)
    _write_options_raw(raw_dir)

    settings = _build_settings(tmp_path=tmp_path, raw_dir=raw_dir)
    monkeypatch.setattr(
        "gamma_exposure_engine.pipeline.offline_pipeline.load_settings",
        lambda require_clickhouse_password=False: settings,
    )

    output_dir = tmp_path / "outputs" / "demo"
    manifest = run_offline_analysis(
        start_date="2024-01-02",
        end_date="2024-01-05",
        output_dir=output_dir,
    )

    expected_files = [
        output_dir / "research_dataset.parquet",
        output_dir / "quantile_summary.csv",
        output_dir / "statistical_tests.csv",
        output_dir / "regime_summary.csv",
        output_dir / "predictive_comparison.csv",
        output_dir / "run_manifest.json",
    ]
    for expected_file in expected_files:
        assert expected_file.exists(), str(expected_file)

    assert manifest["artifacts"]["research_dataset"]["row_count"] > 0
    research_dataset = pl.read_parquet(output_dir / "research_dataset.parquet")
    assert research_dataset.height > 0


def test_run_offline_analysis_fails_clearly_when_raw_inputs_are_missing(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """Offline pipeline should fail with actionable message for missing raw files."""

    raw_dir = tmp_path / "data" / "raw"
    raw_dir.mkdir(parents=True)
    settings = _build_settings(tmp_path=tmp_path, raw_dir=raw_dir)
    monkeypatch.setattr(
        "gamma_exposure_engine.pipeline.offline_pipeline.load_settings",
        lambda require_clickhouse_password=False: settings,
    )

    output_dir = tmp_path / "outputs" / "demo"
    with pytest.raises(RuntimeError, match="SPY_intraday_bars.parquet") as error:
        run_offline_analysis(
            start_date="2024-01-02",
            end_date="2024-01-31",
            output_dir=output_dir,
            raw_data_dir=raw_dir,
        )

    assert "refresh-raw-cache" in str(error.value)


def _write_intraday_raw(raw_dir: Path) -> None:
    intraday = pl.DataFrame(
        {
            "symbol": ["SPY"] * 8,
            "ts": [
                datetime(2024, 1, 2, 9, 30),
                datetime(2024, 1, 2, 9, 31),
                datetime(2024, 1, 3, 9, 30),
                datetime(2024, 1, 3, 9, 31),
                datetime(2024, 1, 4, 9, 30),
                datetime(2024, 1, 4, 9, 31),
                datetime(2024, 1, 5, 9, 30),
                datetime(2024, 1, 5, 9, 31),
            ],
            "open": [100.0, 100.5, 101.0, 101.5, 102.0, 102.5, 103.0, 103.5],
            "high": [100.5, 101.0, 101.5, 102.0, 102.5, 103.0, 103.5, 104.0],
            "low": [99.5, 100.0, 100.5, 101.0, 101.5, 102.0, 102.5, 103.0],
            "close": [100.5, 101.0, 101.5, 102.0, 102.5, 103.0, 103.5, 104.0],
            "volume": [100.0, 110.0, 120.0, 130.0, 140.0, 150.0, 160.0, 170.0],
        }
    )
    intraday.write_parquet(raw_dir / "SPY_intraday_bars.parquet")


def _write_options_raw(raw_dir: Path) -> None:
    options = pl.DataFrame(
        {
            "symbol": ["SPY"] * 10,
            "trade_date": [
                date(2024, 1, 2),
                date(2024, 1, 2),
                date(2024, 1, 2),
                date(2024, 1, 3),
                date(2024, 1, 3),
                date(2024, 1, 3),
                date(2024, 1, 4),
                date(2024, 1, 4),
                date(2024, 1, 4),
                date(2024, 1, 5),
            ],
            "strike_price": [
                100.0,
                101.0,
                102.0,
                101.0,
                102.0,
                103.0,
                102.0,
                103.0,
                104.0,
                105.0,
            ],
            "expiry_date": [date(2024, 1, 19)] * 10,
            "option_type": ["c"] * 10,
            "last_price": [2.4, 2.5, 2.6, 2.5, 2.6, 2.7, 2.6, 2.7, 2.8, 2.9],
            "bid": [2.3, 2.4, 2.5, 2.4, 2.5, 2.6, 2.6, 2.8, 2.7, 2.8],
            "ask": [2.4, 2.5, 2.6, 2.5, 2.6, 2.7, 2.7, 2.7, 2.9, 2.9],
            "bid_iv": [0.2] * 10,
            "ask_iv": [0.21] * 10,
            "open_interest": [80, 100, 120, 90, 110, 130, 95, 115, 135, 140],
            "volume": [10, 11, 12, 11, 12, 13, 12, 13, 14, 15],
            "delta": [0.5] * 10,
            "gamma": [
                0.008,
                0.010,
                0.012,
                0.009,
                0.011,
                0.013,
                0.010,
                0.012,
                0.014,
                0.015,
            ],
            "vega": [0.1] * 10,
            "theta": [-0.01] * 10,
            "rho": [0.01] * 10,
        }
    )
    options.write_parquet(raw_dir / "SPY_options_snapshot.parquet")


class _FakeResearchSettings:
    def __init__(self) -> None:
        self.near_spot_band_width = 0.02
        self.abnormal_volume_window = 2
        self.quantile_count = 5
        self.pinning_candidate_count = 5
        self.predictive_min_train_size = 1
        self.near_spot_share_thresholds = (0.25, 0.5)
        self.default_factor_name = "largest_positive_gamma_strike_distance"
        self.default_target_name = "next_day_realized_variance"
        self.bootstrap_iterations = 200
        self.bootstrap_confidence_level = 0.95
        self.regime_lookback_window = 2
        self.ridge_alpha_candidates = (0.1, 1.0)
        self.robustness_band_widths = (0.01, 0.03)


class _FakeRawDataSettings:
    def __init__(self, raw_data_dir: Path) -> None:
        self.raw_data_dir = raw_data_dir
        self.schema_version = 1


class _FakeSettings:
    def __init__(self, tmp_path: Path, raw_dir: Path) -> None:
        self.project_root = tmp_path
        self.outputs_dir = tmp_path / "outputs"
        self.symbol = "SPY"
        self.research = _FakeResearchSettings()
        self.raw_data = _FakeRawDataSettings(raw_data_dir=raw_dir)


def _build_settings(tmp_path: Path, raw_dir: Path) -> _FakeSettings:
    return _FakeSettings(tmp_path=tmp_path, raw_dir=raw_dir)
