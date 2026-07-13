"""Clean option rows and compute open-interest-weighted gamma mass.

The exposure layer expects a pre-enriched frame where each option snapshot row
already carries the daily underlying close in ``spot_close``. It removes rows
that are structurally unsafe for aggregation, preserves diagnostic flags on
the surviving rows, and applies the documented unsigned gamma convention:

``open_interest_weighted_gamma = open_interest * multiplier * spot_close^2 * gamma``

The input has no owner or dealer-position field. This module therefore does
not attach a position sign or claim to estimate dealer gamma exposure.
"""

from __future__ import annotations

import polars as pl

CONTRACT_MULTIPLIER: int = 100
# Standard US option contract multiplier for listed equity and ETF options.

MINIMUM_VALID_STRIKE_PRICE: float = 0.0
# Strike prices must be strictly positive so moneyness and exposure math stay
# meaningful.

MINIMUM_VALID_SPOT_CLOSE: float = 0.0
# Spot closes must be strictly positive so the contract math and moneyness are
# well-defined.

VALID_OPTION_TYPES: tuple[str, ...] = ("c", "p")
# The cleaner only accepts call and put encodings used by the upstream option
# feed.

ESSENTIAL_COLUMNS: tuple[str, ...] = (
    "trade_date",
    "expiry_date",
    "strike_price",
    "option_type",
    "open_interest",
    "gamma",
    "spot_close",
)
# These fields must be present so the contract-level math is well-defined.

MISSING_ESSENTIAL_COLUMN: str = "_excluded_missing_essential"
NON_POSITIVE_OPEN_INTEREST_COLUMN: str = "_excluded_non_positive_open_interest"
NON_POSITIVE_STRIKE_PRICE_COLUMN: str = "_excluded_non_positive_strike_price"
NON_POSITIVE_SPOT_CLOSE_COLUMN: str = "_excluded_non_positive_spot_close"
EXPIRED_CONTRACT_COLUMN: str = "_excluded_expired_contract"
INVALID_OPTION_TYPE_COLUMN: str = "_excluded_invalid_option_type"
NEGATIVE_GAMMA_COLUMN: str = "_excluded_negative_gamma"
KEEP_ROW_COLUMN: str = "_keep_row"
DIAGNOSTIC_NAME_COLUMN: str = "diagnostic_name"
ROW_COUNT_COLUMN: str = "row_count"
INTERNAL_STATE_COLUMNS: tuple[str, ...] = (
    MISSING_ESSENTIAL_COLUMN,
    NON_POSITIVE_OPEN_INTEREST_COLUMN,
    NON_POSITIVE_STRIKE_PRICE_COLUMN,
    NON_POSITIVE_SPOT_CLOSE_COLUMN,
    EXPIRED_CONTRACT_COLUMN,
    INVALID_OPTION_TYPE_COLUMN,
    NEGATIVE_GAMMA_COLUMN,
    KEEP_ROW_COLUMN,
)
EXCLUSION_DIAGNOSTIC_COLUMNS: tuple[tuple[str, str], ...] = (
    ("excluded_missing_essential_row_count", MISSING_ESSENTIAL_COLUMN),
    (
        "excluded_non_positive_open_interest_row_count",
        NON_POSITIVE_OPEN_INTEREST_COLUMN,
    ),
    (
        "excluded_non_positive_strike_price_row_count",
        NON_POSITIVE_STRIKE_PRICE_COLUMN,
    ),
    ("excluded_non_positive_spot_close_row_count", NON_POSITIVE_SPOT_CLOSE_COLUMN),
    ("excluded_expired_contract_row_count", EXPIRED_CONTRACT_COLUMN),
    ("excluded_invalid_option_type_row_count", INVALID_OPTION_TYPE_COLUMN),
    ("excluded_negative_gamma_row_count", NEGATIVE_GAMMA_COLUMN),
)
SURVIVING_WARNING_DIAGNOSTIC_COLUMNS: tuple[tuple[str, str], ...] = (
    ("surviving_invalid_bid_ask_row_count", "has_invalid_bid_ask"),
    ("surviving_zero_gamma_row_count", "is_zero_gamma"),
)

__all__ = [
    "CONTRACT_MULTIPLIER",
    "clean_options_snapshot",
    "summarize_cleaning_diagnostics",
]


def clean_options_snapshot(frame: pl.DataFrame) -> pl.DataFrame:
    """Return cleaned rows with open-interest-weighted gamma mass.

    Args:
        frame:
            Pre-enriched option snapshot rows. Each row represents one listed
            contract and is expected to include bid/ask quotes, open interest,
            gamma, and the daily underlying close already attached in
            ``spot_close``. The cleaner also assumes structurally valid rows
            have positive ``strike_price`` values and call/put encodings in
            ``option_type``.

    Returns:
        pl.DataFrame: Structurally safe rows for aggregation, with
        quote-validity and zero-gamma diagnostics retained on surviving rows
        and contract-level gamma-mass fields added. Negative gamma rows are
        excluded because standard long vanilla call and put gamma is
        non-negative; the raw schema contains no position sign.
    """

    cleaning_state = _build_cleaning_state(frame)
    cleaned = cleaning_state.filter(pl.col(KEEP_ROW_COLUMN)).drop(
        INTERNAL_STATE_COLUMNS
    )
    cleaned = cleaned.with_columns(
        # Days to expiry is a simple calendar-day difference.
        (pl.col("expiry_date") - pl.col("trade_date"))
        .dt.total_days()
        .alias("days_to_expiry"),
        # Moneyness is strike scaled by spot and centered at zero for at-the-
        # money contracts.
        ((pl.col("strike_price") / pl.col("spot_close")) - 1.0).alias("moneyness"),
        # This is unsigned market gamma mass, not dealer inventory exposure:
        # OI-weighted gamma = OI * multiplier * spot^2 * gamma.
        (
            pl.col("open_interest")
            * CONTRACT_MULTIPLIER
            * pl.col("spot_close")
            * pl.col("spot_close")
            * pl.col("gamma")
        ).alias("open_interest_weighted_gamma"),
    )
    return cleaned


def summarize_cleaning_diagnostics(frame: pl.DataFrame) -> pl.DataFrame:
    """Summarize dropped rows and surviving warning flags from the cleaner.

    Args:
        frame:
            Pre-enriched option snapshot rows before cleaning. The frame is
            expected to use the same schema contract as ``clean_options_snapshot``.

    Returns:
        pl.DataFrame: Compact diagnostics with one row per exclusion or
        surviving-flag count. The counts are intentionally aggregated across
        the full input frame so the HTML report can explain what was removed
        and what suspicious rows still survived.
    """

    cleaning_state = _build_cleaning_state(frame)
    surviving_rows = cleaning_state.filter(pl.col(KEEP_ROW_COLUMN))
    diagnostics: list[dict[str, int | str]] = [
        {
            DIAGNOSTIC_NAME_COLUMN: "input_row_count",
            ROW_COUNT_COLUMN: int(frame.height),
        },
        {
            DIAGNOSTIC_NAME_COLUMN: "cleaned_row_count",
            ROW_COUNT_COLUMN: int(surviving_rows.height),
        },
    ]
    # Each exclusion flag is counted on the full input so the report can show
    # why rows were removed before aggregation.
    for diagnostic_name, flag_column in EXCLUSION_DIAGNOSTIC_COLUMNS:
        diagnostics.append(
            {
                DIAGNOSTIC_NAME_COLUMN: diagnostic_name,
                ROW_COUNT_COLUMN: _count_true_rows(cleaning_state, flag_column),
            }
        )
    # Surviving-row warnings are counted only among rows that passed filtering.
    for diagnostic_name, flag_column in SURVIVING_WARNING_DIAGNOSTIC_COLUMNS:
        diagnostics.append(
            {
                DIAGNOSTIC_NAME_COLUMN: diagnostic_name,
                ROW_COUNT_COLUMN: _count_true_rows(surviving_rows, flag_column),
            }
        )
    return pl.DataFrame(diagnostics)


def _build_cleaning_state(frame: pl.DataFrame) -> pl.DataFrame:
    """Attach sequential exclusion flags and surviving warning flags."""

    cleaning_state = frame.with_columns(
        pl.any_horizontal(
            *[pl.col(column_name).is_null() for column_name in ESSENTIAL_COLUMNS]
        ).alias(MISSING_ESSENTIAL_COLUMN)
    )
    cleaning_state = cleaning_state.with_columns(
        ((~pl.col(MISSING_ESSENTIAL_COLUMN)) & (pl.col("open_interest") <= 0)).alias(
            NON_POSITIVE_OPEN_INTEREST_COLUMN
        )
    )
    cleaning_state = cleaning_state.with_columns(
        (
            (
                ~pl.any_horizontal(
                    pl.col(MISSING_ESSENTIAL_COLUMN),
                    pl.col(NON_POSITIVE_OPEN_INTEREST_COLUMN),
                )
            )
            & (pl.col("strike_price") <= MINIMUM_VALID_STRIKE_PRICE)
        ).alias(NON_POSITIVE_STRIKE_PRICE_COLUMN)
    )
    cleaning_state = cleaning_state.with_columns(
        (
            (
                ~pl.any_horizontal(
                    pl.col(MISSING_ESSENTIAL_COLUMN),
                    pl.col(NON_POSITIVE_OPEN_INTEREST_COLUMN),
                    pl.col(NON_POSITIVE_STRIKE_PRICE_COLUMN),
                )
            )
            & (pl.col("spot_close") <= MINIMUM_VALID_SPOT_CLOSE)
        ).alias(NON_POSITIVE_SPOT_CLOSE_COLUMN)
    )
    cleaning_state = cleaning_state.with_columns(
        (
            (
                ~pl.any_horizontal(
                    pl.col(MISSING_ESSENTIAL_COLUMN),
                    pl.col(NON_POSITIVE_OPEN_INTEREST_COLUMN),
                    pl.col(NON_POSITIVE_STRIKE_PRICE_COLUMN),
                    pl.col(NON_POSITIVE_SPOT_CLOSE_COLUMN),
                )
            )
            & (pl.col("expiry_date") < pl.col("trade_date"))
        ).alias(EXPIRED_CONTRACT_COLUMN)
    )
    cleaning_state = cleaning_state.with_columns(
        (
            (
                ~pl.any_horizontal(
                    pl.col(MISSING_ESSENTIAL_COLUMN),
                    pl.col(NON_POSITIVE_OPEN_INTEREST_COLUMN),
                    pl.col(NON_POSITIVE_STRIKE_PRICE_COLUMN),
                    pl.col(NON_POSITIVE_SPOT_CLOSE_COLUMN),
                    pl.col(EXPIRED_CONTRACT_COLUMN),
                )
            )
            & (~pl.col("option_type").is_in(VALID_OPTION_TYPES))
        ).alias(INVALID_OPTION_TYPE_COLUMN)
    )
    cleaning_state = cleaning_state.with_columns(
        (
            (
                ~pl.any_horizontal(
                    pl.col(MISSING_ESSENTIAL_COLUMN),
                    pl.col(NON_POSITIVE_OPEN_INTEREST_COLUMN),
                    pl.col(NON_POSITIVE_STRIKE_PRICE_COLUMN),
                    pl.col(NON_POSITIVE_SPOT_CLOSE_COLUMN),
                    pl.col(EXPIRED_CONTRACT_COLUMN),
                    pl.col(INVALID_OPTION_TYPE_COLUMN),
                )
            )
            & (pl.col("gamma") < 0.0)
        ).alias(NEGATIVE_GAMMA_COLUMN)
    )
    cleaning_state = cleaning_state.with_columns(
        (
            ~pl.any_horizontal(
                pl.col(MISSING_ESSENTIAL_COLUMN),
                pl.col(NON_POSITIVE_OPEN_INTEREST_COLUMN),
                pl.col(NON_POSITIVE_STRIKE_PRICE_COLUMN),
                pl.col(NON_POSITIVE_SPOT_CLOSE_COLUMN),
                pl.col(EXPIRED_CONTRACT_COLUMN),
                pl.col(INVALID_OPTION_TYPE_COLUMN),
                pl.col(NEGATIVE_GAMMA_COLUMN),
            )
        ).alias(KEEP_ROW_COLUMN),
        # Keep diagnostics on surviving rows so the report can separate rows
        # that were dropped from rows that survived with caveats.
        (pl.col("gamma") == 0).alias("is_zero_gamma"),
        (
            pl.col("bid").is_null()
            | pl.col("ask").is_null()
            | (pl.col("ask") < pl.col("bid"))
        ).alias("has_invalid_bid_ask"),
    )
    return cleaning_state


def _count_true_rows(frame: pl.DataFrame, column_name: str) -> int:
    """Count ``True`` values in a boolean diagnostics column."""

    count = frame.select(pl.col(column_name).sum().fill_null(0)).item()
    return int(count)
