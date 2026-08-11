"""Build the research dataset with explicit t to t plus 1 alignment.

The research frame joins exposure features observed on trade day ``t`` to
response variables on the next observed exposure date that actually has a
response row. The exposure calendar is the reference clock, and the output
retains ``response_trade_date`` for auditability while keeping only public
columns in the final schema.
"""

from __future__ import annotations

import polars as pl

TRADE_DATE_COLUMN: str = "trade_date"
NEXT_DAY_PREFIX: str = "next_day_"
RESPONSE_TRADE_DATE_COLUMN: str = "response_trade_date"
RESPONSE_JOIN_DATE_COLUMN: str = "_response_join_trade_date"
NEXT_EXPOSURE_DATE_COLUMN: str = "_next_exposure_trade_date"

__all__ = ["build_research_dataset"]


def build_research_dataset(
    exposures: pl.DataFrame,
    responses: pl.DataFrame,
) -> pl.DataFrame:
    """Join day ``t`` exposures with the next observed response date.

    Args:
        exposures:
            One row per trade date of exposure features. The sorted exposure
            calendar defines the allowed next-day match for each exposure row.
        responses:
            One row per observed response trade date. Every column except
            ``trade_date`` is renamed with a ``next_day_`` prefix, and the
            response date is retained as ``response_trade_date`` for audit
            checks.

    Returns:
        pl.DataFrame: Exposure columns from day ``t`` plus prefixed response
        metrics from the next observed response date. Only rows whose matched
        response date equals the next observed exposure date survive the join.
    """

    exposure_calendar = _build_exposure_calendar(exposures)
    response_payload = _build_response_payload(responses)
    joined = exposure_calendar.join(
        response_payload,
        left_on=NEXT_EXPOSURE_DATE_COLUMN,
        right_on=RESPONSE_JOIN_DATE_COLUMN,
        how="inner",
    )
    public_columns = _public_columns(exposures, responses)
    return joined.select(public_columns).sort(TRADE_DATE_COLUMN)


def _build_exposure_calendar(exposures: pl.DataFrame) -> pl.DataFrame:
    """Sort exposures and attach the next observed exposure date."""

    ordered_exposures = exposures.sort(TRADE_DATE_COLUMN)
    return ordered_exposures.with_columns(
        # The next exposure date defines the only allowed response match.
        pl.col(TRADE_DATE_COLUMN).shift(-1).alias(NEXT_EXPOSURE_DATE_COLUMN)
    )


def _build_response_payload(responses: pl.DataFrame) -> pl.DataFrame:
    """Rename response metrics while preserving their observed trade date."""

    ordered_responses = responses.sort(TRADE_DATE_COLUMN)
    response_columns = [
        column_name
        for column_name in ordered_responses.columns
        if column_name != TRADE_DATE_COLUMN
    ]
    renamed_columns = {
        column_name: f"{NEXT_DAY_PREFIX}{column_name}"
        for column_name in response_columns
    }

    return (
        ordered_responses.rename(renamed_columns)
        .rename({TRADE_DATE_COLUMN: RESPONSE_TRADE_DATE_COLUMN})
        .with_columns(
            pl.col(RESPONSE_TRADE_DATE_COLUMN).alias(RESPONSE_JOIN_DATE_COLUMN)
        )
    )


def _public_columns(exposures: pl.DataFrame, responses: pl.DataFrame) -> list[str]:
    """Build the final public schema without internal helper columns."""

    exposure_columns = [name for name in exposures.columns if name != TRADE_DATE_COLUMN]
    response_columns = [
        f"{NEXT_DAY_PREFIX}{name}"
        for name in responses.columns
        if name != TRADE_DATE_COLUMN
    ]
    # Keep exposure day-t columns first, then the matched response audit date,
    # then prefixed next-day response metrics.
    return [
        TRADE_DATE_COLUMN,
        *exposure_columns,
        RESPONSE_TRADE_DATE_COLUMN,
        *response_columns,
    ]
