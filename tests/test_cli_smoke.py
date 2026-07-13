"""Smoke tests for the offline-first Typer CLI."""

from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from gamma_exposure_engine.cli import app


def test_run_offline_analysis_command_invokes_pipeline_with_overrides(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """CLI should forward run-offline-analysis options to the pipeline."""

    runner = CliRunner()
    called_with: dict[str, object] = {}

    def fake_run_offline_analysis(
        start_date: str,
        end_date: str,
        output_dir: Path,
        symbol: str | None = None,
        factor_name: str | None = None,
        target_name: str | None = None,
        raw_data_dir: Path | None = None,
    ) -> dict[str, object]:
        called_with.update(
            {
                "start_date": start_date,
                "end_date": end_date,
                "output_dir": output_dir,
                "symbol": symbol,
                "factor_name": factor_name,
                "target_name": target_name,
                "raw_data_dir": raw_data_dir,
            }
        )
        return {"artifacts": {"research_dataset": {"row_count": 10}}}

    monkeypatch.setattr(
        "gamma_exposure_engine.cli.run_offline_analysis",
        fake_run_offline_analysis,
    )

    result = runner.invoke(
        app,
        [
            "run-offline-analysis",
            "--start",
            "2024-01-02",
            "--end",
            "2024-01-31",
            "--output-dir",
            str(tmp_path),
            "--symbol",
            "SPY",
            "--factor",
            "total_open_interest_weighted_gamma",
            "--target",
            "next_day_realized_variance",
            "--raw-dir",
            str(tmp_path / "data" / "raw"),
        ],
    )

    assert result.exit_code == 0
    assert "research_dataset" in result.stdout
    assert called_with == {
        "start_date": "2024-01-02",
        "end_date": "2024-01-31",
        "output_dir": tmp_path,
        "symbol": "SPY",
        "factor_name": "total_open_interest_weighted_gamma",
        "target_name": "next_day_realized_variance",
        "raw_data_dir": tmp_path / "data" / "raw",
    }


def test_refresh_raw_cache_command_invokes_refresh_builder(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """CLI should forward refresh-raw-cache options to refresh builder."""

    runner = CliRunner()
    called_with: dict[str, object] = {}

    class _FakeSettings:
        symbol = "SPY"

    def fake_refresh_raw_cache_from_clickhouse(
        symbol: str,
        start_date: str,
        end_date: str,
        raw_data_dir: Path | None = None,
    ) -> dict[str, object]:
        called_with.update(
            {
                "symbol": symbol,
                "start_date": start_date,
                "end_date": end_date,
                "raw_data_dir": raw_data_dir,
            }
        )
        return {"symbol": symbol, "datasets": {"intraday_bars": {"row_count": 1}}}

    monkeypatch.setattr(
        "gamma_exposure_engine.cli.load_settings", lambda: _FakeSettings()
    )
    monkeypatch.setattr(
        "gamma_exposure_engine.cli.refresh_raw_cache_from_clickhouse",
        fake_refresh_raw_cache_from_clickhouse,
    )

    result = runner.invoke(
        app,
        [
            "refresh-raw-cache",
            "--start",
            "2024-01-02",
            "--end",
            "2024-01-31",
            "--raw-dir",
            str(tmp_path / "data" / "raw"),
        ],
    )

    assert result.exit_code == 0
    assert "intraday_bars" in result.stdout
    assert called_with == {
        "symbol": "SPY",
        "start_date": "2024-01-02",
        "end_date": "2024-01-31",
        "raw_data_dir": tmp_path / "data" / "raw",
    }
