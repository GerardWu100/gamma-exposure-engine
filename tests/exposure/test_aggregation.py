"""Tests for strike maps, expiry maps, and daily gamma factors.

The aggregation layer should consume the cleaned contract frame from Task 3,
preserve structurally valid but flagged quote rows, and produce daily factors
that are meaningful for later research work.
"""

from __future__ import annotations

from datetime import date
from math import isnan

import polars as pl
import pytest

from gamma_exposure_engine.exposure.aggregation import (
    build_daily_gamma_factors,
    build_expiry_gamma_map,
    build_strike_gamma_map,
)


def test_build_strike_gamma_map_aggregates_signed_and_absolute_exposure() -> None:
    """Strike-level aggregation should sum signed and absolute gamma."""

    frame = pl.DataFrame(
        {
            "trade_date": [date(2024, 1, 2), date(2024, 1, 2), date(2024, 1, 2)],
            "expiry_date": [date(2024, 1, 19), date(2024, 1, 19), date(2024, 1, 19)],
            "strike_price": [470.0, 470.0, 475.0],
            "gamma_exposure": [100.0, -20.0, -50.0],
            "spot_close": [472.0, 472.0, 472.0],
            "has_invalid_bid_ask": [False, True, False],
        }
    )

    strike_map = build_strike_gamma_map(frame).sort(["trade_date", "strike_price"])

    assert strike_map["strike_gamma_exposure"].to_list() == [80.0, -50.0]
    assert strike_map["strike_abs_gamma_exposure"].to_list() == [120.0, 50.0]


def test_build_expiry_gamma_map_aggregates_by_trade_date_and_expiry() -> None:
    """Expiry-level aggregation should collapse rows by date and maturity."""

    frame = pl.DataFrame(
        {
            "trade_date": [date(2024, 1, 2), date(2024, 1, 2), date(2024, 1, 2)],
            "expiry_date": [date(2024, 1, 19), date(2024, 1, 26), date(2024, 1, 19)],
            "strike_price": [470.0, 475.0, 480.0],
            "gamma_exposure": [100.0, -20.0, -50.0],
            "spot_close": [472.0, 472.0, 472.0],
            "has_invalid_bid_ask": [False, True, False],
        }
    )

    expiry_map = build_expiry_gamma_map(frame).sort(["trade_date", "expiry_date"])

    assert expiry_map["expiry_gamma_exposure"].to_list() == [50.0, -20.0]
    assert expiry_map["expiry_abs_gamma_exposure"].to_list() == [150.0, 20.0]


def test_build_daily_gamma_factors_uses_cleaned_rows_and_flags() -> None:
    """Daily factors should include flagged rows and derive explainable metrics."""

    frame = pl.DataFrame(
        {
            "trade_date": [
                date(2024, 1, 2),
                date(2024, 1, 2),
                date(2024, 1, 2),
                date(2024, 1, 2),
            ],
            "expiry_date": [
                date(2024, 1, 19),
                date(2024, 1, 19),
                date(2024, 1, 26),
                date(2024, 1, 26),
            ],
            "strike_price": [468.0, 470.0, 475.0, 480.0],
            "option_type": ["c", "p", "c", "p"],
            "gamma_exposure": [120.0, -20.0, -40.0, 10.0],
            "spot_close": [472.0, 472.0, 472.0, 472.0],
            "moneyness": [
                (468.0 / 472.0) - 1.0,
                (470.0 / 472.0) - 1.0,
                (475.0 / 472.0) - 1.0,
                (480.0 / 472.0) - 1.0,
            ],
            "has_invalid_bid_ask": [False, True, False, False],
        }
    )

    factors = build_daily_gamma_factors(frame, near_spot_band=0.01)

    assert factors["trade_date"].to_list() == [date(2024, 1, 2)]
    assert factors["net_gamma_exposure"].to_list() == [70.0]
    assert factors["absolute_gamma_exposure"].to_list() == [190.0]
    assert factors["near_spot_gamma_share"].to_list() == [180.0 / 190.0]
    assert factors["front_expiry_gamma_share"].to_list() == [140.0 / 190.0]
    assert factors["largest_positive_gamma_strike_distance"].to_list() == [
        abs((468.0 / 472.0) - 1.0)
    ]
    assert factors["largest_negative_gamma_strike_distance"].to_list() == [
        abs((475.0 / 472.0) - 1.0)
    ]
    assert factors["call_put_gamma_imbalance"].to_list() == [
        (160.0 - 30.0) / 190.0
    ]
    assert factors["exposure_concentration_index"].to_list() == pytest.approx(
        [
            (120.0 / 190.0) ** 2
            + (20.0 / 190.0) ** 2
            + (40.0 / 190.0) ** 2
            + (10.0 / 190.0) ** 2
        ]
    )


def test_build_daily_gamma_factors_uses_nan_for_missing_node_side() -> None:
    """One-sided books should expose missing node distances as NaN."""

    frame = pl.DataFrame(
        {
            "trade_date": [date(2024, 1, 2), date(2024, 1, 2)],
            "expiry_date": [date(2024, 1, 19), date(2024, 1, 19)],
            "strike_price": [470.0, 475.0],
            "option_type": ["c", "c"],
            "gamma_exposure": [40.0, 70.0],
            "spot_close": [472.0, 472.0],
            "moneyness": [
                (470.0 / 472.0) - 1.0,
                (475.0 / 472.0) - 1.0,
            ],
            "has_invalid_bid_ask": [False, False],
        }
    )

    factors = build_daily_gamma_factors(frame)

    positive_distance = factors["largest_positive_gamma_strike_distance"].to_list()[0]
    negative_distance = factors["largest_negative_gamma_strike_distance"].to_list()[0]

    assert positive_distance == pytest.approx(abs((475.0 / 472.0) - 1.0))
    assert isnan(negative_distance)
    assert factors["call_put_gamma_imbalance"].to_list() == [1.0]


def test_build_daily_gamma_factors_call_put_imbalance_uses_absolute_exposure() -> None:
    """Call-put imbalance should use absolute gamma exposure by option type."""

    frame = pl.DataFrame(
        {
            "trade_date": [date(2024, 1, 2), date(2024, 1, 2), date(2024, 1, 2)],
            "expiry_date": [date(2024, 1, 19), date(2024, 1, 19), date(2024, 1, 19)],
            "strike_price": [470.0, 475.0, 480.0],
            "option_type": ["c", "c", "p"],
            "gamma_exposure": [80.0, -20.0, -30.0],
            "spot_close": [472.0, 472.0, 472.0],
            "moneyness": [
                (470.0 / 472.0) - 1.0,
                (475.0 / 472.0) - 1.0,
                (480.0 / 472.0) - 1.0,
            ],
            "has_invalid_bid_ask": [False, False, False],
        }
    )

    factors = build_daily_gamma_factors(frame)

    assert factors["call_put_gamma_imbalance"].to_list() == [
        ((80.0 + 20.0) - 30.0) / 130.0
    ]


def test_build_daily_gamma_factors_chooses_strike_level_nodes() -> None:
    """Node distances should come from strike-level local gamma mass.

    Multiple rows can share a strike. The aggregation must combine them before
    choosing the strongest positive or negative node for the day.
    """

    frame = pl.DataFrame(
        {
            "trade_date": [
                date(2024, 1, 2),
                date(2024, 1, 2),
                date(2024, 1, 2),
                date(2024, 1, 2),
                date(2024, 1, 2),
            ],
            "expiry_date": [
                date(2024, 1, 19),
                date(2024, 1, 19),
                date(2024, 1, 19),
                date(2024, 1, 26),
                date(2024, 1, 26),
            ],
            "strike_price": [470.0, 470.0, 475.0, 478.0, 480.0],
            "option_type": ["c", "c", "c", "p", "p"],
            "gamma_exposure": [40.0, 70.0, 100.0, -110.0, -30.0],
            "spot_close": [472.0, 472.0, 472.0, 472.0, 472.0],
            "moneyness": [
                (470.0 / 472.0) - 1.0,
                (470.0 / 472.0) - 1.0,
                (475.0 / 472.0) - 1.0,
                (478.0 / 472.0) - 1.0,
                (480.0 / 472.0) - 1.0,
            ],
            "has_invalid_bid_ask": [False, False, False, False, False],
        }
    )

    factors = build_daily_gamma_factors(frame)

    assert factors["largest_positive_gamma_strike_distance"].to_list() == [
        abs((470.0 / 472.0) - 1.0)
    ]
    assert factors["largest_negative_gamma_strike_distance"].to_list() == [
        abs((478.0 / 472.0) - 1.0)
    ]
