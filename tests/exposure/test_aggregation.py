"""Tests for strike, expiry, and daily gamma-mass aggregation."""

from __future__ import annotations

from datetime import date

import polars as pl
import pytest

from gamma_exposure_engine.exposure.aggregation import (
    build_daily_gamma_factors,
    build_expiry_gamma_map,
    build_strike_gamma_map,
)


def _frame() -> pl.DataFrame:
    """Return valid unsigned contract-level gamma mass for one trade date."""

    return pl.DataFrame(
        {
            "trade_date": [date(2024, 1, 2)] * 4,
            "expiry_date": [date(2024, 1, 19)] * 2 + [date(2024, 1, 26)] * 2,
            "strike_price": [468.0, 470.0, 475.0, 480.0],
            "option_type": ["c", "p", "c", "p"],
            "open_interest_weighted_gamma": [120.0, 20.0, 40.0, 10.0],
            "spot_close": [472.0] * 4,
            "moneyness": [
                (468.0 / 472.0) - 1.0,
                (470.0 / 472.0) - 1.0,
                (475.0 / 472.0) - 1.0,
                (480.0 / 472.0) - 1.0,
            ],
            "has_invalid_bid_ask": [False, True, False, False],
        }
    )


def test_build_strike_gamma_map_sums_unsigned_gamma_mass() -> None:
    """Strike aggregation should preserve the financially honest unsigned unit."""

    frame = pl.concat([_frame(), _frame().slice(0, 1)])
    strike_map = build_strike_gamma_map(frame)

    assert strike_map["strike_open_interest_weighted_gamma"].to_list() == [
        240.0,
        20.0,
        40.0,
        10.0,
    ]
    assert "strike_abs_gamma_exposure" not in strike_map.columns


def test_build_expiry_gamma_map_sums_unsigned_gamma_mass() -> None:
    """Expiry aggregation should collapse rows without inventing a sign."""

    expiry_map = build_expiry_gamma_map(_frame())

    assert expiry_map["expiry_open_interest_weighted_gamma"].to_list() == [
        140.0,
        50.0,
    ]


def test_build_daily_gamma_factors_describe_mass_and_composition() -> None:
    """Daily factors should expose mass, location, composition, and concentration."""

    factors = build_daily_gamma_factors(_frame(), near_spot_band=0.01)

    assert factors["total_open_interest_weighted_gamma"].to_list() == [190.0]
    assert factors["near_spot_gamma_mass_share"].to_list() == [180.0 / 190.0]
    assert factors["front_expiry_gamma_mass_share"].to_list() == [140.0 / 190.0]
    assert factors["largest_gamma_mass_strike_distance"].to_list() == [
        abs((468.0 / 472.0) - 1.0)
    ]
    assert factors["call_put_gamma_mass_imbalance"].to_list() == [
        (160.0 - 30.0) / 190.0
    ]
    assert factors["gamma_mass_concentration_index"].to_list() == pytest.approx(
        [
            (120.0 / 190.0) ** 2
            + (20.0 / 190.0) ** 2
            + (40.0 / 190.0) ** 2
            + (10.0 / 190.0) ** 2
        ]
    )
    assert "net_gamma_exposure" not in factors.columns
    assert "absolute_gamma_exposure" not in factors.columns
    assert "largest_negative_gamma_strike_distance" not in factors.columns


def test_build_daily_gamma_factors_chooses_strike_level_mass_node() -> None:
    """Rows sharing a strike should be combined before the largest node is chosen."""

    frame = pl.DataFrame(
        {
            "trade_date": [date(2024, 1, 2)] * 3,
            "expiry_date": [date(2024, 1, 19)] * 3,
            "strike_price": [470.0, 470.0, 475.0],
            "option_type": ["c", "p", "c"],
            "open_interest_weighted_gamma": [40.0, 70.0, 100.0],
            "spot_close": [472.0] * 3,
            "moneyness": [
                (470.0 / 472.0) - 1.0,
                (470.0 / 472.0) - 1.0,
                (475.0 / 472.0) - 1.0,
            ],
            "has_invalid_bid_ask": [False] * 3,
        }
    )

    factors = build_daily_gamma_factors(frame)

    assert factors["largest_gamma_mass_strike_distance"].to_list() == [
        abs((470.0 / 472.0) - 1.0)
    ]
