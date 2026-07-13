"""Linear offline analysis pipeline for the gamma exposure project.

This module orchestrates the full local-Parquet workflow used by the resume
project. It never connects to ClickHouse and writes simple, inspectable
non-HTML artifacts to an output directory.
"""

from __future__ import annotations

import json
from pathlib import Path

import polars as pl

from gamma_exposure_engine.data.raw_store import load_raw_intraday_bars
from gamma_exposure_engine.data.raw_store import load_raw_options_snapshot
from gamma_exposure_engine.exposure.aggregation import (
    build_daily_gamma_factors,
    build_strike_gamma_map,
)
from gamma_exposure_engine.exposure.cleaning import (
    clean_options_snapshot,
    summarize_cleaning_diagnostics,
)
from gamma_exposure_engine.intraday.metrics import attach_pinning_distance
from gamma_exposure_engine.intraday.metrics import build_daily_intraday_metrics
from gamma_exposure_engine.research.bootstrap import build_quantile_summary_with_ci
from gamma_exposure_engine.research.dataset import build_research_dataset
from gamma_exposure_engine.research.descriptive import (
    build_alternative_band_sensitivity,
    build_leave_one_month_out_sensitivity,
    build_near_spot_share_threshold_summary,
    build_subperiod_stability,
)
from gamma_exposure_engine.research.multi_factor import (
    build_factor_factor_correlations,
    build_factor_target_correlations,
)
from gamma_exposure_engine.research.predictive import (
    build_expanding_window_diagnostics,
    build_predictive_baseline_comparison,
)
from gamma_exposure_engine.research.regime import build_regime_quantile_summary
from gamma_exposure_engine.research.statistical_tests import (
    build_statistical_test_summary,
)
from gamma_exposure_engine.settings import load_settings

TRADE_DATE_COLUMN: str = "trade_date"
TIMESTAMP_COLUMN: str = "ts"
CLOSE_COLUMN: str = "close"
SPOT_CLOSE_COLUMN: str = "spot_close"
RESPONSE_TRADE_DATE_COLUMN: str = "response_trade_date"
NEXT_DAY_PREFIX: str = "next_day_"
STRIKE_PRICE_COLUMN: str = "strike_price"
STRIKE_GAMMA_MASS_COLUMN: str = "strike_open_interest_weighted_gamma"


def run_offline_analysis(
    start_date: str,
    end_date: str,
    output_dir: Path,
    symbol: str | None = None,
    factor_name: str | None = None,
    target_name: str | None = None,
    raw_data_dir: Path | None = None,
) -> dict[str, object]:
    """Run the full offline pipeline and write non-HTML artifacts.

    Parameters
    ----------
    start_date:
        Inclusive ISO-8601 start date for local raw-data filtering.
    end_date:
        Inclusive ISO-8601 end date for local raw-data filtering.
    output_dir:
        Destination directory for pipeline artifacts.
    symbol:
        Optional symbol override. Defaults to configured project symbol.
    factor_name:
        Optional descriptive factor override.
    target_name:
        Optional target override.
    raw_data_dir:
        Optional raw-data directory override for tests.

    Returns
    -------
    dict[str, object]
        Run manifest with core run metadata and artifact row counts.
    """

    settings = load_settings(require_clickhouse_password=False)
    resolved_symbol = symbol or settings.symbol
    resolved_factor_name = factor_name or settings.research.default_factor_name
    resolved_target_name = target_name or settings.research.default_target_name

    output_dir.mkdir(parents=True, exist_ok=True)

    # Stage 1: load canonical local raw inputs for the requested window.
    intraday_bars = load_raw_intraday_bars(
        symbol=resolved_symbol,
        start_date=start_date,
        end_date=end_date,
        raw_data_dir=raw_data_dir,
    )
    options_snapshot = load_raw_options_snapshot(
        symbol=resolved_symbol,
        start_date=start_date,
        end_date=end_date,
        raw_data_dir=raw_data_dir,
    )

    # Stage 2: attach spot closes, clean contracts, and build factor/response frames.
    spot_close = build_spot_close_frame(intraday_bars)
    options_with_spot = options_snapshot.join(
        spot_close, on=TRADE_DATE_COLUMN, how="inner"
    )
    cleaning_diagnostics = summarize_cleaning_diagnostics(options_with_spot)
    cleaned_options = clean_options_snapshot(options_with_spot)

    gamma_factors = build_daily_gamma_factors(
        cleaned_options,
        near_spot_band=settings.research.near_spot_band_width,
    )
    intraday_metrics = build_daily_intraday_metrics(
        intraday_bars,
        abnormal_volume_window=settings.research.abnormal_volume_window,
    )
    pinning_candidates = select_pinning_candidates(
        cleaned_options=cleaned_options,
        candidate_count=settings.research.pinning_candidate_count,
    )
    intraday_metrics = attach_pinning_distance(intraday_metrics, pinning_candidates)

    # Stage 3: align day-t exposures with next-day responses for research modules.
    research_dataset = build_research_dataset(
        exposures=gamma_factors, responses=intraday_metrics
    )

    # Stage 4: run descriptive, inferential, robustness, and predictive summaries.
    quantile_summary = build_quantile_summary_with_ci(
        frame=research_dataset,
        factor_name=resolved_factor_name,
        target_name=resolved_target_name,
        quantiles=settings.research.quantile_count,
        bootstrap_iterations=settings.research.bootstrap_iterations,
        confidence_level=settings.research.bootstrap_confidence_level,
    )
    statistical_tests = build_statistical_test_summary(
        frame=research_dataset,
        factor_name=resolved_factor_name,
        target_name=resolved_target_name,
        quantiles=settings.research.quantile_count,
    )
    regime_summary = build_regime_quantile_summary(
        frame=research_dataset,
        factor_name=resolved_factor_name,
        target_name=resolved_target_name,
        quantiles=settings.research.quantile_count,
        lookback_window=settings.research.regime_lookback_window,
    )
    predictive_comparison = build_predictive_baseline_comparison(
        frame=research_dataset,
        feature_name=resolved_factor_name,
        target_name=resolved_target_name,
        min_train_size=settings.research.predictive_min_train_size,
        ridge_alpha_candidates=settings.research.ridge_alpha_candidates,
    )

    target_frame = research_dataset.select(TRADE_DATE_COLUMN, resolved_target_name)
    threshold_robustness = build_near_spot_share_threshold_summary(
        frame=research_dataset,
        target_name=resolved_target_name,
        thresholds=settings.research.near_spot_share_thresholds,
    )
    subperiod_stability = build_subperiod_stability(
        frame=research_dataset,
        factor_name=resolved_factor_name,
        target_name=resolved_target_name,
    )
    band_sensitivity = build_alternative_band_sensitivity(
        cleaned_options=cleaned_options,
        targets=target_frame,
        target_name=resolved_target_name,
        band_widths=settings.research.robustness_band_widths,
    )
    loo_month_sensitivity = build_leave_one_month_out_sensitivity(
        frame=research_dataset,
        factor_name=resolved_factor_name,
        target_name=resolved_target_name,
    )

    factor_columns = _factor_columns(research_dataset)
    target_columns = _target_columns(research_dataset)
    factor_target_correlations = build_factor_target_correlations(
        frame=research_dataset,
        factor_names=factor_columns,
        target_names=target_columns,
    )
    factor_factor_correlations = build_factor_factor_correlations(
        frame=research_dataset,
        factor_names=factor_columns,
    )
    expanding_window_diagnostics = build_expanding_window_diagnostics(
        frame=research_dataset,
        feature_name=resolved_factor_name,
        target_name=resolved_target_name,
        min_train_size=settings.research.predictive_min_train_size,
        alpha_candidates=settings.research.ridge_alpha_candidates,
    )

    artifact_frames = {
        "cleaning_diagnostics": cleaning_diagnostics,
        "gamma_factors": gamma_factors,
        "intraday_metrics": intraday_metrics,
        "research_dataset": research_dataset,
        "quantile_summary": quantile_summary,
        "statistical_tests": statistical_tests,
        "regime_summary": regime_summary,
        "predictive_comparison": predictive_comparison,
        "threshold_robustness": threshold_robustness,
        "subperiod_stability": subperiod_stability,
        "band_sensitivity": band_sensitivity,
        "loo_month_sensitivity": loo_month_sensitivity,
        "factor_target_correlations": factor_target_correlations,
        "factor_factor_correlations": factor_factor_correlations,
        "expanding_window_diagnostics": expanding_window_diagnostics,
    }
    artifact_paths = write_artifacts(
        output_dir=output_dir, artifact_frames=artifact_frames
    )

    run_manifest = {
        "symbol": resolved_symbol,
        "start_date": start_date,
        "end_date": end_date,
        "factor_name": resolved_factor_name,
        "target_name": resolved_target_name,
        "raw_data_dir": str(raw_data_dir or settings.raw_data.raw_data_dir),
        "artifacts": {
            artifact_name: {
                "path": str(artifact_paths[artifact_name]),
                "row_count": artifact_frames[artifact_name].height,
            }
            for artifact_name in artifact_frames
        },
    }
    run_manifest_path = output_dir / "run_manifest.json"
    run_manifest_path.write_text(json.dumps(run_manifest, indent=2), encoding="utf-8")
    return run_manifest


def write_artifacts(
    output_dir: Path,
    artifact_frames: dict[str, pl.DataFrame],
) -> dict[str, Path]:
    """Write pipeline artifacts to Parquet and CSV files.

    Parameters
    ----------
    output_dir:
        Destination directory for files.
    artifact_frames:
        Named Polars frames produced by the offline pipeline.

    Returns
    -------
    dict[str, Path]
        Artifact path by artifact name.
    """

    paths: dict[str, Path] = {}
    for artifact_name, frame in artifact_frames.items():
        if artifact_name == "research_dataset":
            artifact_path = output_dir / f"{artifact_name}.parquet"
            frame.write_parquet(artifact_path)
        else:
            artifact_path = output_dir / f"{artifact_name}.csv"
            frame.write_csv(artifact_path)

        paths[artifact_name] = artifact_path

    return paths


def build_spot_close_frame(intraday_bars: pl.DataFrame) -> pl.DataFrame:
    """Reduce minute bars to one daily close row per trade date."""

    dated_bars = intraday_bars.sort(TIMESTAMP_COLUMN).with_columns(
        pl.col(TIMESTAMP_COLUMN).dt.date().alias(TRADE_DATE_COLUMN)
    )
    return (
        dated_bars.group_by(TRADE_DATE_COLUMN)
        .agg(pl.col(CLOSE_COLUMN).last().alias(SPOT_CLOSE_COLUMN))
        .sort(TRADE_DATE_COLUMN)
    )


def select_pinning_candidates(
    cleaned_options: pl.DataFrame,
    candidate_count: int,
) -> pl.DataFrame:
    """Choose the largest strike-level gamma-mass nodes for each trade date."""

    strike_map = build_strike_gamma_map(cleaned_options)
    ordered_strikes = strike_map.sort(
        [TRADE_DATE_COLUMN, STRIKE_GAMMA_MASS_COLUMN, STRIKE_PRICE_COLUMN],
        descending=[False, True, False],
    )
    return (
        ordered_strikes.group_by(TRADE_DATE_COLUMN, maintain_order=True)
        .head(candidate_count)
        .select(TRADE_DATE_COLUMN, STRIKE_PRICE_COLUMN)
    )


def _factor_columns(research_dataset: pl.DataFrame) -> list[str]:
    """Return exposure-factor columns from the aligned research dataset."""

    return [
        column_name
        for column_name in research_dataset.columns
        if column_name not in {TRADE_DATE_COLUMN, RESPONSE_TRADE_DATE_COLUMN}
        and not column_name.startswith(NEXT_DAY_PREFIX)
    ]


def _target_columns(research_dataset: pl.DataFrame) -> list[str]:
    """Return response-target columns from the aligned research dataset."""

    return [
        column_name
        for column_name in research_dataset.columns
        if column_name.startswith(NEXT_DAY_PREFIX)
    ]
