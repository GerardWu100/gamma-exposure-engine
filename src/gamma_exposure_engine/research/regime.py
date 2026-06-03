"""Volatility regime classification and conditioned quantile summaries.

This module asks whether a factor-target association changes across market
states. The state variable is a past-only trailing realized-variance measure:
for day ``t``, the regime signal is computed from earlier days only, so the
classification cannot leak the response of day ``t`` into its own label.
"""

from __future__ import annotations

from typing import Final

import polars as pl

from gamma_exposure_engine.research.descriptive import (
    OBSERVATION_COUNT_COLUMN,
    QUANTILE_BUCKET_COLUMN,
    TARGET_MEAN_COLUMN,
    build_quantile_summary,
)

TRADE_DATE_COLUMN: Final[str] = "trade_date"
REALIZED_VARIANCE_COLUMN: Final[str] = "next_day_realized_variance"
REGIME_COLUMN: Final[str] = "volatility_regime"
TRAILING_REALIZED_VARIANCE_COLUMN: Final[str] = "_trailing_realized_variance"
HIGH_VOLATILITY_LABEL: Final[str] = "high_volatility"
LOW_VOLATILITY_LABEL: Final[str] = "low_volatility"

__all__ = [
    "HIGH_VOLATILITY_LABEL",
    "LOW_VOLATILITY_LABEL",
    "REGIME_COLUMN",
    "build_regime_quantile_summary",
    "classify_volatility_regime",
]


def classify_volatility_regime(
    frame: pl.DataFrame,
    realized_variance_column: str,
    lookback_window: int,
) -> pl.DataFrame:
    """Classify each row into a high or low volatility regime.

    Parameters
    ----------
    frame:
        Research dataset with one row per trade date.
    realized_variance_column:
        Column used to estimate the trailing volatility state.
    lookback_window:
        Number of prior observations used in the trailing average.

    Returns
    -------
    pl.DataFrame
        Input frame plus ``volatility_regime``. The first ``lookback_window``
        rows remain null because there is not enough trailing history.
    """

    if lookback_window < 1:
        msg = "lookback_window must be at least 1"
        raise ValueError(msg)

    sorted_frame = frame.sort(TRADE_DATE_COLUMN)
    # Shift by one row so day t never uses its own realized variance when
    # computing the state that conditions day t.
    trailing_frame = sorted_frame.with_columns(
        pl.col(realized_variance_column)
        .shift(1)
        .rolling_mean(window_size=lookback_window, min_samples=lookback_window)
        .alias(TRAILING_REALIZED_VARIANCE_COLUMN)
    )

    trailing_median = trailing_frame.select(
        pl.col(TRAILING_REALIZED_VARIANCE_COLUMN).median()
    ).item()
    if trailing_median is None:
        return trailing_frame.with_columns(
            pl.lit(None, dtype=pl.String).alias(REGIME_COLUMN)
        ).drop(TRAILING_REALIZED_VARIANCE_COLUMN)

    classified_frame = trailing_frame.with_columns(
        pl.when(pl.col(TRAILING_REALIZED_VARIANCE_COLUMN).is_null())
        .then(pl.lit(None, dtype=pl.String))
        .when(pl.col(TRAILING_REALIZED_VARIANCE_COLUMN) > float(trailing_median))
        .then(pl.lit(HIGH_VOLATILITY_LABEL))
        .otherwise(pl.lit(LOW_VOLATILITY_LABEL))
        .alias(REGIME_COLUMN)
    )
    return classified_frame.drop(TRAILING_REALIZED_VARIANCE_COLUMN)


def build_regime_quantile_summary(
    frame: pl.DataFrame,
    factor_name: str,
    target_name: str,
    quantiles: int,
    lookback_window: int,
    realized_variance_column: str = REALIZED_VARIANCE_COLUMN,
) -> pl.DataFrame:
    """Summarize bucket means separately within low and high volatility states.

    Parameters
    ----------
    frame:
        Research dataset with one row per trade date.
    factor_name:
        Factor column used for the within-regime quantile sort.
    target_name:
        Target column whose bucket means are reported.
    quantiles:
        Requested number of within-regime buckets.
    lookback_window:
        Trailing history length used for regime classification.
    realized_variance_column:
        Column used to define the regime. Defaults to the next-day realized
        variance metric even when another target is summarized.

    Returns
    -------
    pl.DataFrame
        One row per observed ``(regime, quantile_bucket)`` pair.
    """

    if frame.height == 0:
        return _empty_regime_summary()

    classified_frame = classify_volatility_regime(
        frame=frame,
        realized_variance_column=realized_variance_column,
        lookback_window=lookback_window,
    ).drop_nulls([REGIME_COLUMN])
    if classified_frame.height == 0:
        return _empty_regime_summary()

    regime_summaries: list[pl.DataFrame] = []
    for regime_label in [LOW_VOLATILITY_LABEL, HIGH_VOLATILITY_LABEL]:
        regime_frame = classified_frame.filter(pl.col(REGIME_COLUMN) == regime_label)
        if regime_frame.height == 0:
            continue

        regime_summary = build_quantile_summary(
            frame=regime_frame,
            factor_name=factor_name,
            target_name=target_name,
            quantiles=quantiles,
        ).with_columns(pl.lit(regime_label).alias(REGIME_COLUMN))
        regime_summaries.append(regime_summary)

    if not regime_summaries:
        return _empty_regime_summary()

    return pl.concat(regime_summaries, how="vertical").select(
        REGIME_COLUMN,
        QUANTILE_BUCKET_COLUMN,
        TARGET_MEAN_COLUMN,
        OBSERVATION_COUNT_COLUMN,
    )


def _empty_regime_summary() -> pl.DataFrame:
    """Return the public empty schema for regime-conditioned summaries."""

    return pl.DataFrame(
        {
            REGIME_COLUMN: pl.Series([], dtype=pl.String),
            QUANTILE_BUCKET_COLUMN: pl.Series([], dtype=pl.Int64),
            TARGET_MEAN_COLUMN: pl.Series([], dtype=pl.Float64),
            OBSERVATION_COUNT_COLUMN: pl.Series([], dtype=pl.Int64),
        }
    )
