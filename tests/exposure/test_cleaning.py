"""Tests for option cleaning and open-interest-weighted gamma math.

The exposure layer should remove rows that cannot support exposure math, flag
diagnostic states on surviving rows, and compute deterministic contract-level
fields with the documented scaling convention.
"""

from __future__ import annotations

from datetime import date

import polars as pl

from gamma_exposure_engine.exposure.cleaning import (
    CONTRACT_MULTIPLIER,
    clean_options_snapshot,
    summarize_cleaning_diagnostics,
)


def test_clean_options_snapshot_drops_missing_essentials_and_non_positive_open_interest() -> (
    None
):
    """Rows missing essentials or with non-positive open interest are removed."""

    frame = pl.DataFrame(
        {
            "symbol": ["SPY", "SPY", "SPY"],
            "trade_date": [date(2024, 1, 2), date(2024, 1, 2), date(2024, 1, 2)],
            "expiry_date": [date(2024, 1, 19), date(2024, 1, 19), None],
            "strike_price": [470.0, 475.0, 480.0],
            "option_type": ["c", "p", "c"],
            "bid": [1.0, 1.2, 1.1],
            "ask": [1.1, 1.3, 1.2],
            "open_interest": [10, 0, 20],
            "gamma": [0.02, 0.03, 0.04],
            "spot_close": [472.0, 472.0, 472.0],
        }
    )

    cleaned = clean_options_snapshot(frame)

    assert cleaned.height == 1
    assert cleaned["days_to_expiry"].to_list() == [17]
    assert cleaned["moneyness"].to_list() == [(470.0 / 472.0) - 1.0]
    assert cleaned["open_interest_weighted_gamma"].to_list() == [
        10 * CONTRACT_MULTIPLIER * 472.0 * 472.0 * 0.02
    ]


def test_clean_options_snapshot_drops_structurally_invalid_rows() -> None:
    """Rows with impossible calendar or price structure are removed."""

    frame = pl.DataFrame(
        {
            "symbol": ["SPY", "SPY", "SPY"],
            "trade_date": [date(2024, 1, 2), date(2024, 1, 2), date(2024, 1, 2)],
            "expiry_date": [date(2024, 1, 19), date(2024, 1, 1), date(2024, 1, 19)],
            "strike_price": [470.0, 475.0, 480.0],
            "option_type": ["c", "p", "c"],
            "bid": [1.0, 1.2, 1.1],
            "ask": [1.1, 1.3, 1.2],
            "open_interest": [10, 10, 10],
            "gamma": [0.02, 0.03, 0.04],
            "spot_close": [472.0, 472.0, 0.0],
        }
    )

    cleaned = clean_options_snapshot(frame)

    assert cleaned.height == 1
    assert cleaned["strike_price"].to_list() == [470.0]
    assert cleaned["days_to_expiry"].to_list() == [17]
    assert cleaned["has_invalid_bid_ask"].to_list() == [False]
    assert cleaned["is_zero_gamma"].to_list() == [False]


def test_clean_options_snapshot_drops_non_positive_strike_prices() -> None:
    """Zero or negative strikes are structurally unsafe and must be removed."""

    frame = pl.DataFrame(
        {
            "symbol": ["SPY", "SPY"],
            "trade_date": [date(2024, 1, 2), date(2024, 1, 2)],
            "expiry_date": [date(2024, 1, 19), date(2024, 1, 19)],
            "strike_price": [0.0, 470.0],
            "option_type": ["c", "p"],
            "bid": [1.0, 1.2],
            "ask": [1.1, 1.3],
            "open_interest": [10, 10],
            "gamma": [0.02, 0.03],
            "spot_close": [472.0, 472.0],
        }
    )

    cleaned = clean_options_snapshot(frame)

    assert cleaned.height == 1
    assert cleaned["strike_price"].to_list() == [470.0]


def test_clean_options_snapshot_drops_unexpected_option_types() -> None:
    """Only call and put rows are structurally valid."""

    frame = pl.DataFrame(
        {
            "symbol": ["SPY", "SPY", "SPY"],
            "trade_date": [date(2024, 1, 2), date(2024, 1, 2), date(2024, 1, 2)],
            "expiry_date": [date(2024, 1, 19), date(2024, 1, 19), date(2024, 1, 19)],
            "strike_price": [470.0, 471.0, 472.0],
            "option_type": ["c", "x", "p"],
            "bid": [1.0, 1.1, 1.2],
            "ask": [1.1, 1.2, 1.3],
            "open_interest": [10, 10, 10],
            "gamma": [0.02, 0.03, 0.04],
            "spot_close": [472.0, 472.0, 472.0],
        }
    )

    cleaned = clean_options_snapshot(frame)

    assert cleaned.height == 2
    assert cleaned["option_type"].to_list() == ["c", "p"]


def test_clean_options_snapshot_flags_zero_gamma_rows() -> None:
    """Zero-gamma rows stay in the output and are flagged for diagnostics."""

    frame = pl.DataFrame(
        {
            "symbol": ["SPY"],
            "trade_date": [date(2024, 1, 2)],
            "expiry_date": [date(2024, 1, 19)],
            "strike_price": [470.0],
            "option_type": ["c"],
            "bid": [1.0],
            "ask": [1.2],
            "open_interest": [10],
            "gamma": [0.0],
            "spot_close": [472.0],
        }
    )

    cleaned = clean_options_snapshot(frame)

    assert cleaned.height == 1
    assert cleaned["is_zero_gamma"].to_list() == [True]
    assert cleaned["open_interest_weighted_gamma"].to_list() == [0.0]


def test_clean_options_snapshot_excludes_negative_vendor_gamma() -> None:
    """Negative vanilla gamma is invalid data, not a dealer-position sign."""

    frame = pl.DataFrame(
        {
            "symbol": ["SPY"],
            "trade_date": [date(2024, 1, 2)],
            "expiry_date": [date(2024, 1, 19)],
            "strike_price": [470.0],
            "option_type": ["p"],
            "bid": [1.0],
            "ask": [1.2],
            "open_interest": [10],
            "gamma": [-0.02],
            "spot_close": [472.0],
        }
    )

    assert clean_options_snapshot(frame).is_empty()
    diagnostics = summarize_cleaning_diagnostics(frame)
    counts = dict(
        zip(
            diagnostics["diagnostic_name"].to_list(),
            diagnostics["row_count"].to_list(),
            strict=True,
        )
    )
    assert counts["excluded_negative_gamma_row_count"] == 1


def test_clean_options_snapshot_flags_invalid_bid_ask_rows() -> None:
    """Bid above ask should be retained but marked invalid."""

    frame = pl.DataFrame(
        {
            "symbol": ["SPY"],
            "trade_date": [date(2024, 1, 2)],
            "expiry_date": [date(2024, 1, 19)],
            "strike_price": [470.0],
            "option_type": ["c"],
            "bid": [1.3],
            "ask": [1.2],
            "open_interest": [10],
            "gamma": [0.02],
            "spot_close": [472.0],
        }
    )

    cleaned = clean_options_snapshot(frame)

    assert cleaned.height == 1
    assert cleaned["has_invalid_bid_ask"].to_list() == [True]


def test_summarize_cleaning_diagnostics_counts_exclusions_and_surviving_flags() -> None:
    """Diagnostics should report both dropped rows and flagged survivors."""

    frame = pl.DataFrame(
        {
            "symbol": ["SPY"] * 8,
            "trade_date": [date(2024, 1, 2)] * 8,
            "expiry_date": [
                date(2024, 1, 19),
                date(2024, 1, 19),
                None,
                date(2024, 1, 19),
                date(2024, 1, 19),
                date(2024, 1, 1),
                date(2024, 1, 19),
                date(2024, 1, 19),
            ],
            "strike_price": [470.0, 471.0, 472.0, 0.0, 474.0, 475.0, 476.0, 477.0],
            "option_type": ["c", "c", "c", "c", "p", "p", "x", "c"],
            "bid": [1.0, 1.3, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            "ask": [1.1, 1.2, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1],
            "open_interest": [10, 10, 10, 10, 0, 10, 10, 10],
            "gamma": [0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.0],
            "spot_close": [472.0] * 8,
        }
    )

    diagnostics = summarize_cleaning_diagnostics(frame)
    counts = dict(
        zip(
            diagnostics["diagnostic_name"].to_list(),
            diagnostics["row_count"].to_list(),
            strict=True,
        )
    )

    assert counts["input_row_count"] == 8
    assert counts["cleaned_row_count"] == 3
    assert counts["excluded_missing_essential_row_count"] == 1
    assert counts["excluded_non_positive_open_interest_row_count"] == 1
    assert counts["excluded_non_positive_strike_price_row_count"] == 1
    assert counts["excluded_expired_contract_row_count"] == 1
    assert counts["excluded_invalid_option_type_row_count"] == 1
    assert counts["excluded_negative_gamma_row_count"] == 0
    assert counts["surviving_invalid_bid_ask_row_count"] == 1
    assert counts["surviving_zero_gamma_row_count"] == 1
