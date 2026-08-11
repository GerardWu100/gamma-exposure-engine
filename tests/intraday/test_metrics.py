"""Tests for intraday response metrics and pinning distance joins.

The intraday layer should turn minute bars into one row per trade date with a
simple realized-volatility proxy, a defensible abnormal-volume score that only
uses past data, and a helper for joining close-to-strike pinning distances.
"""

from __future__ import annotations

from datetime import date, datetime
from math import isnan, log, sqrt

import polars as pl
import pytest
from gamma_exposure_engine.intraday.metrics import (
    _build_minute_bars,
    attach_pinning_distance,
    build_daily_intraday_metrics,
)


def test_build_daily_intraday_metrics_computes_daily_bar_metrics() -> None:
    """Daily metrics should collapse minute bars into one row per trade date."""

    frame = pl.DataFrame(
        {
            "symbol": ["SPY", "SPY", "SPY", "SPY", "SPY", "SPY"],
            "ts": [
                datetime(2024, 1, 2, 9, 30),
                datetime(2024, 1, 2, 9, 31),
                datetime(2024, 1, 2, 9, 32),
                datetime(2024, 1, 3, 9, 30),
                datetime(2024, 1, 3, 9, 31),
                datetime(2024, 1, 3, 9, 32),
            ],
            "open": [100.0, 101.0, 100.5, 102.0, 101.0, 100.0],
            "high": [101.0, 101.5, 100.75, 102.5, 101.25, 100.25],
            "low": [99.5, 100.25, 100.0, 101.5, 100.5, 99.5],
            "close": [100.0, 101.0, 100.5, 102.0, 101.0, 100.0],
            "volume": [100.0, 150.0, 250.0, 120.0, 180.0, 300.0],
        }
    )

    metrics = build_daily_intraday_metrics(frame, abnormal_volume_window=2)

    assert metrics["trade_date"].to_list() == [
        date(2024, 1, 2),
        date(2024, 1, 3),
    ]
    assert metrics["close_price"].to_list() == [100.5, 100.0]
    assert metrics["total_volume"].to_list() == [500.0, 600.0]
    assert metrics["realized_volatility"].to_list() == pytest.approx(
        [
            sqrt((log(101.0 / 100.0)) ** 2 + (log(100.5 / 101.0)) ** 2),
            sqrt((log(101.0 / 102.0)) ** 2 + (log(100.0 / 101.0)) ** 2),
        ]
    )
    assert metrics["open_to_close_abs_return"].to_list() == [
        abs((100.5 / 100.0) - 1.0),
        abs((100.0 / 102.0) - 1.0),
    ]
    close_to_close_abs_return = metrics["close_to_close_abs_return"].to_list()
    assert isnan(close_to_close_abs_return[0])
    assert close_to_close_abs_return[1] == abs((100.0 / 100.5) - 1.0)


def test_build_daily_intraday_metrics_sorts_daily_series_before_close_shift() -> None:
    """Close-to-close return should not depend on the incoming minute order."""

    frame = pl.DataFrame(
        {
            "symbol": ["SPY", "SPY", "SPY", "SPY", "SPY", "SPY"],
            "ts": [
                datetime(2024, 1, 3, 9, 30),
                datetime(2024, 1, 3, 9, 31),
                datetime(2024, 1, 3, 9, 32),
                datetime(2024, 1, 2, 9, 30),
                datetime(2024, 1, 2, 9, 31),
                datetime(2024, 1, 2, 9, 32),
            ],
            "open": [102.0, 101.0, 100.0, 100.0, 101.0, 100.5],
            "high": [102.5, 101.25, 100.25, 101.0, 101.5, 100.75],
            "low": [101.5, 100.5, 99.5, 99.5, 100.25, 100.0],
            "close": [102.0, 101.0, 100.0, 100.0, 101.0, 100.5],
            "volume": [120.0, 180.0, 300.0, 100.0, 150.0, 250.0],
        }
    )

    metrics = build_daily_intraday_metrics(frame, abnormal_volume_window=2)

    assert metrics["trade_date"].to_list() == [
        date(2024, 1, 2),
        date(2024, 1, 3),
    ]
    assert metrics["close_price"].to_list() == [100.5, 100.0]
    close_to_close_abs_return = metrics["close_to_close_abs_return"].to_list()
    assert isnan(close_to_close_abs_return[0])
    assert close_to_close_abs_return[1] == abs((100.0 / 100.5) - 1.0)


def test_build_daily_intraday_metrics_uses_trailing_minute_baseline_only() -> None:
    """The abnormal-volume score should only use prior days for the baseline."""

    frame = pl.DataFrame(
        {
            "symbol": ["SPY"] * 12,
            "ts": [
                datetime(2024, 1, 2, 9, 30),
                datetime(2024, 1, 2, 9, 31),
                datetime(2024, 1, 2, 9, 32),
                datetime(2024, 1, 3, 9, 30),
                datetime(2024, 1, 3, 9, 31),
                datetime(2024, 1, 3, 9, 32),
                datetime(2024, 1, 4, 9, 30),
                datetime(2024, 1, 4, 9, 31),
                datetime(2024, 1, 4, 9, 32),
                datetime(2024, 1, 5, 9, 30),
                datetime(2024, 1, 5, 9, 31),
                datetime(2024, 1, 5, 9, 32),
            ],
            "open": [100.0] * 12,
            "high": [100.5] * 12,
            "low": [99.5] * 12,
            "close": [100.0] * 12,
            "volume": [
                10.0,
                10.0,
                10.0,
                20.0,
                20.0,
                20.0,
                40.0,
                40.0,
                40.0,
                60.0,
                80.0,
                80.0,
            ],
        }
    )

    metrics = build_daily_intraday_metrics(frame, abnormal_volume_window=2)

    # Day 1 has no prior history, so the contract is explicitly numeric.
    assert (
        metrics.filter(pl.col("trade_date") == date(2024, 1, 2))[
            "abnormal_volume_score"
        ].item()
        == 0.0
    )

    # Day 2 only sees day 1 in the trailing baseline.
    assert (
        metrics.filter(pl.col("trade_date") == date(2024, 1, 3))[
            "abnormal_volume_score"
        ].item()
        == 2.0
    )

    # Day 3 sees day 1 and day 2, not itself.
    assert (
        metrics.filter(pl.col("trade_date") == date(2024, 1, 4))[
            "abnormal_volume_score"
        ].item()
        == 8.0 / 3.0
    )

    # Day 4 sees only the most recent two prior days because the window is 2.
    assert (
        metrics.filter(pl.col("trade_date") == date(2024, 1, 5))[
            "abnormal_volume_score"
        ].item()
        == 22.0 / 9.0
    )


def test_attach_pinning_distance_uses_prior_day_nearest_candidate_strike() -> None:
    """Pinning distance should use the closest prior-day strike from the set."""

    metrics = pl.DataFrame(
        {
            "trade_date": [date(2024, 1, 3), date(2024, 1, 4)],
            "close_price": [471.5, 480.0],
        }
    )
    candidate_strikes = pl.DataFrame(
        {
            "trade_date": [date(2024, 1, 3), date(2024, 1, 3), date(2024, 1, 4)],
            "strike_price": [470.0, 475.0, 485.0],
            "gamma_rank": [1, 2, 1],
        }
    )

    result = attach_pinning_distance(metrics, candidate_strikes)

    pinning_distance = result["pinning_distance"].to_list()
    assert isnan(pinning_distance[0])
    assert pinning_distance[1] == pytest.approx(5.0 / 480.0)


def test_attach_pinning_distance_uses_nan_when_no_candidate_strike_exists() -> None:
    """Missing candidate strikes should produce a numeric missing-value contract."""

    metrics = pl.DataFrame(
        {
            "trade_date": [date(2024, 1, 3)],
            "close_price": [471.5],
        }
    )
    candidate_strikes = pl.DataFrame(
        {
            "trade_date": [date(2024, 1, 2)],
            "strike_price": [470.0],
        }
    )

    result = attach_pinning_distance(metrics, candidate_strikes)

    assert isnan(result["pinning_distance"].item())


def test_attach_pinning_distance_uses_previous_day_candidates_and_normalizes() -> None:
    """Pinning distance should use prior-day candidates and scale by close."""

    metrics = pl.DataFrame(
        {
            "trade_date": [
                date(2024, 1, 2),
                date(2024, 1, 3),
                date(2024, 1, 4),
            ],
            "close_price": [100.0, 104.0, 96.0],
        }
    )
    candidate_strikes = pl.DataFrame(
        {
            "trade_date": [
                date(2024, 1, 2),
                date(2024, 1, 2),
                date(2024, 1, 2),
                date(2024, 1, 3),
                date(2024, 1, 3),
            ],
            "strike_price": [95.0, 100.0, 110.0, 90.0, 98.0],
        }
    )

    result = attach_pinning_distance(metrics, candidate_strikes)

    pinning_distance = result["pinning_distance"].to_list()
    assert isnan(pinning_distance[0])
    assert pinning_distance[1] == pytest.approx(4.0 / 104.0)
    assert pinning_distance[2] == pytest.approx(2.0 / 96.0)


def test_build_minute_bars_keeps_afternoon_minutes_distinct() -> None:
    """Minute-of-day must not wrap around, so late-session bars stay separate.

    Polars returns Int8 for both ``dt.hour()`` and ``dt.minute()``. Multiplying
    the raw Int8 hour by 60 overflows past 127, which folds 09:00 onto 28 and
    15:30 onto -94 and merges unrelated clock minutes into one bucket.
    """

    frame = pl.DataFrame(
        {
            "ts": [
                datetime(2024, 1, 2, 4, 44),
                datetime(2024, 1, 2, 9, 0),
                datetime(2024, 1, 2, 15, 30),
                datetime(2024, 1, 2, 19, 59),
            ],
            "open": [100.0, 100.0, 100.0, 100.0],
            "close": [100.0, 100.0, 100.0, 100.0],
            "volume": [1.0, 1.0, 1.0, 1.0],
        }
    )

    minute_bars = _build_minute_bars(frame)

    assert minute_bars["minute_of_day"].to_list() == [284, 540, 930, 1199]
