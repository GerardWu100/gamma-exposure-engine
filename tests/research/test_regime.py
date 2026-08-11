"""Tests for volatility regime conditioning."""

from __future__ import annotations

from datetime import date, timedelta

import polars as pl
from gamma_exposure_engine.research.regime import (
    HIGH_VOLATILITY_LABEL,
    LOW_VOLATILITY_LABEL,
    REGIME_COLUMN,
    build_regime_quantile_summary,
    classify_volatility_regime,
)


def _make_research_frame(row_count: int = 40) -> pl.DataFrame:
    """Build a synthetic research frame with a simple volatility shift."""

    dates = [date(2024, 1, 2) + timedelta(days=index) for index in range(row_count)]
    realized_variance = [0.001] * (row_count // 2) + [0.010] * (row_count // 2)
    factor_values = list(range(row_count))
    return pl.DataFrame(
        {
            "trade_date": dates,
            "factor_value": [float(value) for value in factor_values],
            "next_day_realized_variance": realized_variance,
        }
    )


def test_classify_regime_produces_two_labels() -> None:
    """The classifier should eventually emit both high and low labels."""

    frame = _make_research_frame(row_count=40)

    classified = classify_volatility_regime(
        frame=frame,
        realized_variance_column="next_day_realized_variance",
        lookback_window=5,
    )

    regime_values = set(classified[REGIME_COLUMN].drop_nulls().to_list())
    assert regime_values == {HIGH_VOLATILITY_LABEL, LOW_VOLATILITY_LABEL}


def test_classify_regime_uses_past_only_window() -> None:
    """The first lookback window should stay null because history is missing."""

    frame = _make_research_frame(row_count=40)

    classified = classify_volatility_regime(
        frame=frame,
        realized_variance_column="next_day_realized_variance",
        lookback_window=5,
    )

    first_regimes = classified.head(5)[REGIME_COLUMN].to_list()
    assert all(regime is None for regime in first_regimes)


def test_build_regime_quantile_summary_has_both_regimes() -> None:
    """The conditioned summary should include both volatility states."""

    frame = _make_research_frame(row_count=40)

    summary = build_regime_quantile_summary(
        frame=frame,
        factor_name="factor_value",
        target_name="next_day_realized_variance",
        quantiles=2,
        lookback_window=5,
    )

    regimes_in_output = set(summary[REGIME_COLUMN].to_list())
    assert HIGH_VOLATILITY_LABEL in regimes_in_output
    assert LOW_VOLATILITY_LABEL in regimes_in_output
    assert "quantile_bucket" in summary.columns
    assert "target_mean" in summary.columns
    assert "observation_count" in summary.columns


def test_build_regime_quantile_summary_empty_frame() -> None:
    """Empty input should preserve the public regime-summary schema."""

    frame = pl.DataFrame(
        {
            "trade_date": pl.Series([], dtype=pl.Date),
            "factor_value": pl.Series([], dtype=pl.Float64),
            "next_day_realized_variance": pl.Series([], dtype=pl.Float64),
        }
    )

    summary = build_regime_quantile_summary(
        frame=frame,
        factor_name="factor_value",
        target_name="next_day_realized_variance",
        quantiles=2,
        lookback_window=5,
    )

    assert summary.height == 0
    assert REGIME_COLUMN in summary.columns
    assert "quantile_bucket" in summary.columns
