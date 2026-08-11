"""Compute daily intraday response metrics from minute SPY bars.

The intraday layer reduces minute bars to one row per trade date for later
research. It intentionally keeps the metrics simple and auditable:
``realized_variance`` is the sum of squared minute log returns,
``realized_volatility`` is the unannualized square root of that variance,
``open_to_close_abs_return`` and ``close_to_close_abs_return`` are absolute
daily return magnitudes, ``close_to_close_abs_return`` is computed on the
trade-date-sorted daily series and is ``NaN`` for the first available day,
``close_price`` and ``total_volume`` are the daily close and total volume, and
``abnormal_volume_score`` is a trailing past-only volume ratio. The module
also exposes a helper for measuring a normalized pinning-distance proxy from
next-day closes to prior-day candidate gamma strikes.
"""

from __future__ import annotations

import polars as pl

TRADE_DATE_COLUMN: str = "trade_date"
TIMESTAMP_COLUMN: str = "ts"
OPEN_COLUMN: str = "open"
CLOSE_COLUMN: str = "close"
VOLUME_COLUMN: str = "volume"
MINUTE_OF_DAY_COLUMN: str = "minute_of_day"
MINUTE_VOLUME_COLUMN: str = "minute_volume"
LOG_RETURN_COLUMN: str = "log_return"
REALIZED_VARIANCE_COLUMN: str = "realized_variance"
REALIZED_VOLATILITY_COLUMN: str = "realized_volatility"
OPEN_TO_CLOSE_ABS_RETURN_COLUMN: str = "open_to_close_abs_return"
CLOSE_TO_CLOSE_ABS_RETURN_COLUMN: str = "close_to_close_abs_return"
CLOSE_PRICE_COLUMN: str = "close_price"
TOTAL_VOLUME_COLUMN: str = "total_volume"
ABNORMAL_VOLUME_SCORE_COLUMN: str = "abnormal_volume_score"
PINNING_DISTANCE_COLUMN: str = "pinning_distance"
STRIKE_PRICE_COLUMN: str = "strike_price"
EXPECTED_TOTAL_VOLUME_COLUMN: str = "expected_total_volume"
BASELINE_MINUTE_VOLUME_COLUMN: str = "baseline_minute_volume"
MISSING_NUMERIC: float = float("nan")
ZERO_FLOAT: float = 0.0
MINUTES_PER_HOUR: int = 60

__all__ = ["attach_pinning_distance", "build_daily_intraday_metrics"]


def build_daily_intraday_metrics(
    frame: pl.DataFrame,
    abnormal_volume_window: int,
) -> pl.DataFrame:
    """Aggregate minute bars into one daily response row per trade date.

    Args:
        frame:
            Minute SPY bars. Each row represents one intraday bar and is
            expected to include ``ts``, ``open``, ``close``, and ``volume``.
            The frame may contain additional columns, which are ignored.
        abnormal_volume_window:
            Number of prior trade dates to use in the trailing minute-of-day
            baseline. The baseline is computed with past data only; same-day
            and future bars are excluded by construction.

    Returns:
        pl.DataFrame: One row per ``trade_date`` with columns for realized
        variance, realized volatility, absolute open-to-close return,
        absolute close-to-close return, close price, total volume, and
        abnormal volume score. ``realized_volatility`` is the unannualized
        square root of ``realized_variance``. ``close_to_close_abs_return`` is
        computed from the trade-date-sorted daily close series, and the first
        available value is ``NaN`` because there is no prior close.
    """

    minute_bars = _build_minute_bars(frame)
    daily_metrics = _build_daily_bar_metrics(minute_bars).sort(TRADE_DATE_COLUMN)
    # Compute the day-over-day close return on the sorted daily series before
    # any join work so the contract does not depend on join ordering.
    daily_metrics = daily_metrics.with_columns(
        pl.col(REALIZED_VARIANCE_COLUMN).sqrt().alias(REALIZED_VOLATILITY_COLUMN),
        (
            (pl.col(CLOSE_PRICE_COLUMN) / pl.col(CLOSE_PRICE_COLUMN).shift(1))
            - 1.0
        )
        .abs()
        .alias(CLOSE_TO_CLOSE_ABS_RETURN_COLUMN),
    )
    abnormal_volume = _build_abnormal_volume_scores(
        minute_bars=minute_bars,
        abnormal_volume_window=abnormal_volume_window,
    )
    metrics = daily_metrics.join(abnormal_volume, on=TRADE_DATE_COLUMN, how="left")
    metrics = metrics.with_columns(
        pl.col(ABNORMAL_VOLUME_SCORE_COLUMN).fill_null(ZERO_FLOAT)
    )
    metrics = metrics.with_columns(
        pl.col(CLOSE_TO_CLOSE_ABS_RETURN_COLUMN).fill_null(MISSING_NUMERIC)
    )
    return metrics.sort(TRADE_DATE_COLUMN)


def attach_pinning_distance(
    metrics: pl.DataFrame,
    candidate_strikes: pl.DataFrame,
) -> pl.DataFrame:
    """Attach the prior-day nearest candidate-strike distance to response rows.

    Args:
        metrics:
            Daily intraday response metrics keyed by ``trade_date``.
        candidate_strikes:
            Candidate high-gamma strikes keyed by the exposure trade date. The
            helper only needs ``trade_date`` and ``strike_price``; any other
            columns are ignored.

    Returns:
        pl.DataFrame: The input metrics with ``pinning_distance`` added. The
        proxy uses the previous observed trade date's candidate strikes, then
        measures the absolute close-to-strike gap normalized by the current
        close price. When a day has no prior candidate strikes, the output is
        ``NaN`` by explicit numeric contract.
    """

    candidate_calendar = metrics.select(TRADE_DATE_COLUMN).sort(TRADE_DATE_COLUMN)
    candidate_calendar = candidate_calendar.with_columns(
        pl.col(TRADE_DATE_COLUMN).shift(-1).alias("_response_trade_date")
    )
    aligned_candidates = candidate_strikes.select(
        pl.col(TRADE_DATE_COLUMN),
        pl.col(STRIKE_PRICE_COLUMN),
    ).join(
        candidate_calendar.rename(
            {TRADE_DATE_COLUMN: "_candidate_trade_date"}
        ),
        left_on=TRADE_DATE_COLUMN,
        right_on="_candidate_trade_date",
        how="inner",
    ).drop_nulls(["_response_trade_date"]).select(
        pl.col("_response_trade_date").alias(TRADE_DATE_COLUMN),
        pl.col(STRIKE_PRICE_COLUMN),
    )
    close_to_strike = metrics.join(
        aligned_candidates,
        on=TRADE_DATE_COLUMN,
        how="left",
    )
    close_to_strike = close_to_strike.with_columns(
        # Normalize by the current close so the proxy is dimensionless and can
        # be compared across price levels.
        (
            (pl.col(CLOSE_PRICE_COLUMN) - pl.col(STRIKE_PRICE_COLUMN)).abs()
            / pl.col(CLOSE_PRICE_COLUMN)
        )
        .alias(PINNING_DISTANCE_COLUMN)
    )
    nearest_distance = close_to_strike.group_by(TRADE_DATE_COLUMN).agg(
        pl.col(PINNING_DISTANCE_COLUMN).min().alias(PINNING_DISTANCE_COLUMN)
    )
    attached = metrics.join(nearest_distance, on=TRADE_DATE_COLUMN, how="left")
    attached = attached.with_columns(
        pl.col(PINNING_DISTANCE_COLUMN).fill_null(MISSING_NUMERIC)
    )
    return attached.sort(TRADE_DATE_COLUMN)


def _build_minute_bars(frame: pl.DataFrame) -> pl.DataFrame:
    """Normalize raw bars with date and minute-of-day fields."""

    minute_bars = frame.sort(TIMESTAMP_COLUMN).with_columns(
        pl.col(TIMESTAMP_COLUMN).dt.date().alias(TRADE_DATE_COLUMN),
        # Cast the hour and minute parts to Int32 before the arithmetic. Polars
        # returns Int8 for both parts, and an Int8 product wraps around at 127,
        # so `hour * 60` alone would fold 09:00 onto 28 and 15:30 onto -94 and
        # merge unrelated clock minutes into the same bucket.
        (
            pl.col(TIMESTAMP_COLUMN).dt.hour().cast(pl.Int32) * MINUTES_PER_HOUR
            + pl.col(TIMESTAMP_COLUMN).dt.minute().cast(pl.Int32)
        ).alias(MINUTE_OF_DAY_COLUMN),
    )
    return minute_bars


def _build_daily_bar_metrics(minute_bars: pl.DataFrame) -> pl.DataFrame:
    """Compute realized variance and basic daily summary metrics."""

    daily_bars = minute_bars.with_columns(
        pl.col(CLOSE_COLUMN).log().alias("_log_close")
    )
    daily_bars = daily_bars.with_columns(
        (pl.col("_log_close") - pl.col("_log_close").shift(1))
        .over(TRADE_DATE_COLUMN)
        .alias(LOG_RETURN_COLUMN)
    )
    daily_metrics = daily_bars.group_by(TRADE_DATE_COLUMN).agg(
        pl.col(LOG_RETURN_COLUMN).pow(2).sum().alias(REALIZED_VARIANCE_COLUMN),
        (
            (pl.col(CLOSE_COLUMN).last() / pl.col(OPEN_COLUMN).first()) - 1.0
        )
        .abs()
        .alias(OPEN_TO_CLOSE_ABS_RETURN_COLUMN),
        pl.col(CLOSE_COLUMN).last().alias(CLOSE_PRICE_COLUMN),
        pl.col(VOLUME_COLUMN).sum().alias(TOTAL_VOLUME_COLUMN),
    )
    return daily_metrics.sort(TRADE_DATE_COLUMN)


def _build_abnormal_volume_scores(
    minute_bars: pl.DataFrame,
    abnormal_volume_window: int,
) -> pl.DataFrame:
    """Compute the trailing minute-of-day abnormal-volume score.

    The baseline is the rolling mean of each minute-of-day volume series
    across prior trade dates only. The current day is excluded by shifting the
    rolling mean one row forward inside each minute bucket.
    """

    minute_volume = minute_bars.group_by(
        TRADE_DATE_COLUMN,
        MINUTE_OF_DAY_COLUMN,
    ).agg(pl.col(VOLUME_COLUMN).sum().alias(MINUTE_VOLUME_COLUMN))
    minute_volume = minute_volume.sort([MINUTE_OF_DAY_COLUMN, TRADE_DATE_COLUMN])
    minute_volume = minute_volume.with_columns(
        # Roll the minute-by-minute volume series forward one day so the
        # current observation only sees strictly prior trade dates.
        pl.col(MINUTE_VOLUME_COLUMN)
        .rolling_mean(window_size=abnormal_volume_window, min_samples=1)
        .shift(1)
        .over(MINUTE_OF_DAY_COLUMN)
        .alias(BASELINE_MINUTE_VOLUME_COLUMN)
    )
    minute_volume = minute_volume.with_columns(
        # Days with no trailing history get an explicit numeric zero contract
        # rather than a null, which keeps downstream joins simple.
        pl.col(BASELINE_MINUTE_VOLUME_COLUMN).fill_null(ZERO_FLOAT)
    )
    expected_daily_volume = minute_volume.group_by(TRADE_DATE_COLUMN).agg(
        pl.col(BASELINE_MINUTE_VOLUME_COLUMN).sum().alias(
            EXPECTED_TOTAL_VOLUME_COLUMN
        )
    )
    daily_total_volume = minute_volume.group_by(TRADE_DATE_COLUMN).agg(
        pl.col(MINUTE_VOLUME_COLUMN).sum().alias(TOTAL_VOLUME_COLUMN)
    )
    abnormal_volume = daily_total_volume.join(
        expected_daily_volume,
        on=TRADE_DATE_COLUMN,
        how="left",
    )
    abnormal_volume = abnormal_volume.with_columns(
        # Use a ratio so the score is dimensionless and easy to compare across
        # days; a missing or zero baseline maps to the explicit zero contract.
        pl.when(pl.col(EXPECTED_TOTAL_VOLUME_COLUMN) > ZERO_FLOAT)
        .then(pl.col(TOTAL_VOLUME_COLUMN) / pl.col(EXPECTED_TOTAL_VOLUME_COLUMN))
        .otherwise(ZERO_FLOAT)
        .fill_null(ZERO_FLOAT)
        .alias(ABNORMAL_VOLUME_SCORE_COLUMN)
    )
    return abnormal_volume.select(
        TRADE_DATE_COLUMN,
        ABNORMAL_VOLUME_SCORE_COLUMN,
    ).sort(TRADE_DATE_COLUMN)
