"""Typer command-line interface for offline analysis and optional refresh."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import typer

from gamma_exposure_engine.data.raw_cache_builder import refresh_raw_cache_from_clickhouse
from gamma_exposure_engine.pipeline.offline_pipeline import run_offline_analysis
from gamma_exposure_engine.settings import load_settings

app = typer.Typer(help="Gamma Exposure Engine CLI.")


@app.callback()
def main() -> None:
    """Expose the Typer command group."""


@app.command("refresh-raw-cache")
def refresh_raw_cache_command(
    start: str = typer.Option(..., help="Inclusive ISO-8601 start date."),
    end: str = typer.Option(..., help="Inclusive ISO-8601 end date."),
    symbol: str | None = typer.Option(None, help="Override configured symbol."),
    raw_dir: Path | None = typer.Option(None, help="Override data/raw destination."),
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Enable DEBUG logging."
    ),
) -> None:
    """Refresh canonical raw Parquet files from ClickHouse once."""

    _configure_logging(verbose)
    settings = load_settings()
    resolved_symbol = symbol or settings.symbol
    manifest = refresh_raw_cache_from_clickhouse(
        symbol=resolved_symbol,
        start_date=start,
        end_date=end,
        raw_data_dir=raw_dir,
    )
    typer.echo(json.dumps(manifest, indent=2))


@app.command("run-offline-analysis")
def run_offline_analysis_command(
    start: str = typer.Option(..., help="Inclusive ISO-8601 start date."),
    end: str = typer.Option(..., help="Inclusive ISO-8601 end date."),
    output_dir: Path = typer.Option(
        ..., help="Output directory for non-HTML artifacts."
    ),
    symbol: str | None = typer.Option(None, help="Override configured symbol."),
    factor: str | None = typer.Option(None, help="Override configured factor."),
    target: str | None = typer.Option(None, help="Override configured target."),
    raw_dir: Path | None = typer.Option(None, help="Override data/raw source path."),
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Enable DEBUG logging."
    ),
) -> None:
    """Run the offline-only analysis pipeline from local Parquet files."""

    _configure_logging(verbose)
    manifest = run_offline_analysis(
        start_date=start,
        end_date=end,
        output_dir=output_dir,
        symbol=symbol,
        factor_name=factor,
        target_name=target,
        raw_data_dir=raw_dir,
    )
    typer.echo(json.dumps(manifest, indent=2))


def _configure_logging(verbose: bool) -> None:
    """Initialize process-wide logging with optional DEBUG verbosity."""

    log_level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        force=True,
    )


if __name__ == "__main__":
    app()
