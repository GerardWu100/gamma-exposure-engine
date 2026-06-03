"""Aggregate cleaned option rows into strike, expiry, and daily gamma views.

The aggregation layer consumes the cleaned contract frame from
``exposure.cleaning``. It keeps structurally valid rows,
including rows that were flagged with invalid bid/ask quotes, because the
cleaner already decided whether the row is safe to keep. The aggregations in
this module therefore only reshape and summarize gamma exposure; they do not
apply additional quality filters.
"""

from __future__ import annotations

import polars as pl

CALL_OPTION_TYPE: str = "c"
PUT_OPTION_TYPE: str = "p"

DEFAULT_NEAR_SPOT_BAND: float = 0.02
ZERO_FLOAT: float = 0.0

TRADE_DATE_COLUMN: str = "trade_date"
EXPIRY_DATE_COLUMN: str = "expiry_date"
STRIKE_PRICE_COLUMN: str = "strike_price"
OPTION_TYPE_COLUMN: str = "option_type"
MONEYNESS_COLUMN: str = "moneyness"
GAMMA_EXPOSURE_COLUMN: str = "gamma_exposure"
ABSOLUTE_GAMMA_EXPOSURE_COLUMN: str = "absolute_gamma_exposure"
NET_GAMMA_EXPOSURE_COLUMN: str = "net_gamma_exposure"

STRIKE_GAMMA_EXPOSURE_COLUMN: str = "strike_gamma_exposure"
STRIKE_ABS_GAMMA_EXPOSURE_COLUMN: str = "strike_abs_gamma_exposure"
EXPIRY_GAMMA_EXPOSURE_COLUMN: str = "expiry_gamma_exposure"
EXPIRY_ABS_GAMMA_EXPOSURE_COLUMN: str = "expiry_abs_gamma_exposure"
CALL_GAMMA_EXPOSURE_COLUMN: str = "call_gamma_exposure"
PUT_GAMMA_EXPOSURE_COLUMN: str = "put_gamma_exposure"
NEAR_SPOT_GAMMA_SHARE_COLUMN: str = "near_spot_gamma_share"
FRONT_EXPIRY_GAMMA_SHARE_COLUMN: str = "front_expiry_gamma_share"
LARGEST_POSITIVE_GAMMA_STRIKE_DISTANCE_COLUMN: str = (
    "largest_positive_gamma_strike_distance"
)
LARGEST_NEGATIVE_GAMMA_STRIKE_DISTANCE_COLUMN: str = (
    "largest_negative_gamma_strike_distance"
)
CALL_PUT_GAMMA_IMBALANCE_COLUMN: str = "call_put_gamma_imbalance"
EXPOSURE_CONCENTRATION_INDEX_COLUMN: str = "exposure_concentration_index"
SPOT_DISTANCE_COLUMN: str = "spot_distance"
FRONT_EXPIRY_DATE_COLUMN: str = "front_expiry_date"
NEAR_SPOT_ABS_GAMMA_EXPOSURE_COLUMN: str = "near_spot_abs_gamma_exposure"
FRONT_EXPIRY_ABS_GAMMA_EXPOSURE_COLUMN: str = "front_expiry_abs_gamma_exposure"
STRIKE_EXPIRY_ABS_GAMMA_EXPOSURE_COLUMN: str = "strike_expiry_abs_gamma_exposure"
STRIKE_EXPIRY_GAMMA_SHARE_COLUMN: str = "strike_expiry_gamma_share"

__all__ = [
    "DEFAULT_NEAR_SPOT_BAND",
    "build_daily_gamma_factors",
    "build_expiry_gamma_map",
    "build_strike_gamma_map",
]


def build_strike_gamma_map(frame: pl.DataFrame) -> pl.DataFrame:
    """Aggregate cleaned contract rows to strike level.

    Args:
        frame:
            Cleaned contract rows. Each row represents one option contract on
            one trade date and may include a diagnostic ``has_invalid_bid_ask``
            flag from the cleaner. The flag is retained on the row, but it does
            not change aggregation arithmetic.

    Returns:
        pl.DataFrame: One row per ``trade_date`` and ``strike_price`` with
        signed and absolute gamma exposure totals.
    """

    strike_map = frame.group_by([TRADE_DATE_COLUMN, STRIKE_PRICE_COLUMN]).agg(
        pl.col(GAMMA_EXPOSURE_COLUMN).sum().alias(STRIKE_GAMMA_EXPOSURE_COLUMN),
        pl.col(GAMMA_EXPOSURE_COLUMN).abs().sum().alias(
            STRIKE_ABS_GAMMA_EXPOSURE_COLUMN
        ),
    )
    return strike_map.sort([TRADE_DATE_COLUMN, STRIKE_PRICE_COLUMN])


def build_expiry_gamma_map(frame: pl.DataFrame) -> pl.DataFrame:
    """Aggregate cleaned contract rows to expiry level.

    Args:
        frame:
            Cleaned contract rows. Rows that were flagged with invalid bid/ask
            quotes remain in the aggregation because quote validity is a
            diagnostic property, not a filter at this stage.

    Returns:
        pl.DataFrame: One row per ``trade_date`` and ``expiry_date`` with
        signed and absolute gamma exposure totals.
    """

    expiry_map = frame.group_by([TRADE_DATE_COLUMN, EXPIRY_DATE_COLUMN]).agg(
        pl.col(GAMMA_EXPOSURE_COLUMN).sum().alias(EXPIRY_GAMMA_EXPOSURE_COLUMN),
        pl.col(GAMMA_EXPOSURE_COLUMN).abs().sum().alias(
            EXPIRY_ABS_GAMMA_EXPOSURE_COLUMN
        ),
    )
    return expiry_map.sort([TRADE_DATE_COLUMN, EXPIRY_DATE_COLUMN])


def build_daily_gamma_factors(
    frame: pl.DataFrame,
    near_spot_band: float = DEFAULT_NEAR_SPOT_BAND,
) -> pl.DataFrame:
    """Build daily gamma factors from cleaned contract rows.

    Args:
        frame:
            Cleaned contract rows from Task 3. The frame is expected to already
            include ``gamma_exposure``, ``moneyness``, ``trade_date``,
            ``expiry_date``, ``strike_price``, ``spot_close``, and
            ``option_type``. Rows flagged with invalid bid/ask quotes remain in
            the input and are included in the factor math.
        near_spot_band:
            Absolute moneyness band around spot used for the near-spot share.
            A value of ``0.02`` means plus or minus 2 percent around spot.
            ``call_put_gamma_imbalance`` is computed from absolute gamma
            exposure by option type and normalized by total absolute gamma.

    Returns:
        pl.DataFrame: One row per ``trade_date`` with explainable daily gamma
        factors for research. The node-distance factors are non-negative
        magnitudes measured from spot, and a missing side is encoded as
        numeric ``NaN`` so downstream code does not have to infer null
        semantics.
    """

    daily_totals = _build_daily_totals(frame)
    near_spot = _build_near_spot_share(frame, daily_totals, near_spot_band)
    front_expiry = _build_front_expiry_share(frame, daily_totals)
    positive_node = _build_signed_node_distance(
        frame=frame,
        daily_totals=daily_totals,
        sign=1,
        distance_column=LARGEST_POSITIVE_GAMMA_STRIKE_DISTANCE_COLUMN,
    )
    negative_node = _build_signed_node_distance(
        frame=frame,
        daily_totals=daily_totals,
        sign=-1,
        distance_column=LARGEST_NEGATIVE_GAMMA_STRIKE_DISTANCE_COLUMN,
    )
    call_put = _build_call_put_imbalance(frame, daily_totals)
    concentration = _build_exposure_concentration_index(frame, daily_totals)

    factors = daily_totals
    factors = factors.join(near_spot, on=TRADE_DATE_COLUMN, how="left")
    factors = factors.join(front_expiry, on=TRADE_DATE_COLUMN, how="left")
    factors = factors.join(positive_node, on=TRADE_DATE_COLUMN, how="left")
    factors = factors.join(negative_node, on=TRADE_DATE_COLUMN, how="left")
    factors = factors.join(call_put, on=TRADE_DATE_COLUMN, how="left")
    factors = factors.join(concentration, on=TRADE_DATE_COLUMN, how="left")

    return factors.sort(TRADE_DATE_COLUMN)


def _build_daily_totals(frame: pl.DataFrame) -> pl.DataFrame:
    """Summarize signed and absolute gamma exposure by trade date."""

    daily_totals = frame.group_by(TRADE_DATE_COLUMN).agg(
        pl.col(GAMMA_EXPOSURE_COLUMN).sum().alias(NET_GAMMA_EXPOSURE_COLUMN),
        pl.col(GAMMA_EXPOSURE_COLUMN).abs().sum().alias(
            ABSOLUTE_GAMMA_EXPOSURE_COLUMN
        ),
    )
    return daily_totals.sort(TRADE_DATE_COLUMN)


def _build_near_spot_share(
    frame: pl.DataFrame,
    daily_totals: pl.DataFrame,
    near_spot_band: float,
) -> pl.DataFrame:
    """Compute the share of absolute gamma exposure inside the near-spot band."""

    near_spot_abs = (
        frame.filter(pl.col(MONEYNESS_COLUMN).abs() <= near_spot_band)
        .group_by(TRADE_DATE_COLUMN)
        .agg(
            pl.col(GAMMA_EXPOSURE_COLUMN)
            .abs()
            .sum()
            .alias(NEAR_SPOT_ABS_GAMMA_EXPOSURE_COLUMN)
        )
    )
    return _absolute_gamma_share(
        partial_abs_frame=near_spot_abs,
        numerator_column=NEAR_SPOT_ABS_GAMMA_EXPOSURE_COLUMN,
        share_column=NEAR_SPOT_GAMMA_SHARE_COLUMN,
        daily_totals=daily_totals,
    )


def _build_front_expiry_share(
    frame: pl.DataFrame,
    daily_totals: pl.DataFrame,
) -> pl.DataFrame:
    """Compute the share of absolute gamma exposure in the nearest expiry."""

    front_expiry_dates = frame.group_by(TRADE_DATE_COLUMN).agg(
        pl.col(EXPIRY_DATE_COLUMN).min().alias(FRONT_EXPIRY_DATE_COLUMN)
    )
    front_expiry_abs = (
        frame.join(front_expiry_dates, on=TRADE_DATE_COLUMN, how="inner")
        .filter(pl.col(EXPIRY_DATE_COLUMN) == pl.col(FRONT_EXPIRY_DATE_COLUMN))
        .group_by(TRADE_DATE_COLUMN)
        .agg(
            pl.col(GAMMA_EXPOSURE_COLUMN)
            .abs()
            .sum()
            .alias(FRONT_EXPIRY_ABS_GAMMA_EXPOSURE_COLUMN)
        )
    )
    return _absolute_gamma_share(
        partial_abs_frame=front_expiry_abs,
        numerator_column=FRONT_EXPIRY_ABS_GAMMA_EXPOSURE_COLUMN,
        share_column=FRONT_EXPIRY_GAMMA_SHARE_COLUMN,
        daily_totals=daily_totals,
    )


def _absolute_gamma_share(
    partial_abs_frame: pl.DataFrame,
    numerator_column: str,
    share_column: str,
    daily_totals: pl.DataFrame,
) -> pl.DataFrame:
    """Convert a partial absolute-gamma total into a share of the daily total."""

    share_frame = partial_abs_frame.join(
        daily_totals.select(TRADE_DATE_COLUMN, ABSOLUTE_GAMMA_EXPOSURE_COLUMN),
        on=TRADE_DATE_COLUMN,
        how="right",
    )
    share_frame = share_frame.with_columns(
        pl.when(pl.col(ABSOLUTE_GAMMA_EXPOSURE_COLUMN) > ZERO_FLOAT)
        .then(pl.col(numerator_column) / pl.col(ABSOLUTE_GAMMA_EXPOSURE_COLUMN))
        .otherwise(ZERO_FLOAT)
        .fill_null(ZERO_FLOAT)
        .alias(share_column)
    )
    return share_frame.select(TRADE_DATE_COLUMN, share_column)


def _build_signed_node_distance(
    frame: pl.DataFrame,
    daily_totals: pl.DataFrame,
    sign: int,
    distance_column: str,
) -> pl.DataFrame:
    """Select the largest positive or negative gamma node and report distance."""

    node_frame = _build_strike_level_gamma_frame(frame)
    node_frame = node_frame.with_columns(
        pl.col(MONEYNESS_COLUMN).abs().alias(SPOT_DISTANCE_COLUMN)
    )

    if sign > 0:
        node_frame = node_frame.filter(pl.col(GAMMA_EXPOSURE_COLUMN) > ZERO_FLOAT)
        node_frame = node_frame.sort(
            [
                TRADE_DATE_COLUMN,
                GAMMA_EXPOSURE_COLUMN,
                SPOT_DISTANCE_COLUMN,
                STRIKE_PRICE_COLUMN,
            ],
            descending=[False, True, False, False],
        )
    else:
        node_frame = node_frame.filter(pl.col(GAMMA_EXPOSURE_COLUMN) < ZERO_FLOAT)
        node_frame = node_frame.sort(
            [
                TRADE_DATE_COLUMN,
                GAMMA_EXPOSURE_COLUMN,
                SPOT_DISTANCE_COLUMN,
                STRIKE_PRICE_COLUMN,
            ],
            descending=[False, False, False, False],
        )

    node_frame = node_frame.group_by(TRADE_DATE_COLUMN, maintain_order=True).first()
    node_frame = daily_totals.select(TRADE_DATE_COLUMN).join(
        node_frame.select(
            TRADE_DATE_COLUMN,
            pl.col(SPOT_DISTANCE_COLUMN).alias(distance_column),
        ),
        on=TRADE_DATE_COLUMN,
        how="left",
    )
    return node_frame.with_columns(
        pl.col(distance_column).fill_null(float("nan"))
    ).select(
        TRADE_DATE_COLUMN,
        distance_column,
    )


def _build_strike_level_gamma_frame(frame: pl.DataFrame) -> pl.DataFrame:
    """Aggregate contract rows to strike level before node selection.

    The same strike can appear multiple times within a day. The strike-level
    node logic must therefore collapse all rows at that strike first and use
    the resulting local gamma mass when choosing the strongest node.
    """

    strike_level_frame = frame.group_by(
        [TRADE_DATE_COLUMN, STRIKE_PRICE_COLUMN]
    ).agg(
        pl.col(GAMMA_EXPOSURE_COLUMN).sum().alias(GAMMA_EXPOSURE_COLUMN),
        pl.col(GAMMA_EXPOSURE_COLUMN).abs().sum().alias(
            ABSOLUTE_GAMMA_EXPOSURE_COLUMN
        ),
        pl.col(MONEYNESS_COLUMN).first().alias(MONEYNESS_COLUMN),
    )
    return strike_level_frame.sort([TRADE_DATE_COLUMN, STRIKE_PRICE_COLUMN])


def _build_call_put_imbalance(
    frame: pl.DataFrame,
    daily_totals: pl.DataFrame,
) -> pl.DataFrame:
    """Compute absolute call-versus-put gamma imbalance.

    The imbalance uses absolute gamma exposure within each option type so the
    signal measures how concentrated the book is on the call side versus the
    put side, regardless of sign convention.
    """

    call_put = frame.group_by(TRADE_DATE_COLUMN).agg(
        pl.when(pl.col(OPTION_TYPE_COLUMN) == CALL_OPTION_TYPE)
        .then(pl.col(GAMMA_EXPOSURE_COLUMN).abs())
        .otherwise(None)
        .sum()
        .alias(CALL_GAMMA_EXPOSURE_COLUMN),
        pl.when(pl.col(OPTION_TYPE_COLUMN) == PUT_OPTION_TYPE)
        .then(pl.col(GAMMA_EXPOSURE_COLUMN).abs())
        .otherwise(None)
        .sum()
        .alias(PUT_GAMMA_EXPOSURE_COLUMN),
    )
    call_put = call_put.join(
        daily_totals.select(TRADE_DATE_COLUMN, ABSOLUTE_GAMMA_EXPOSURE_COLUMN),
        on=TRADE_DATE_COLUMN,
        how="left",
    )
    call_put = call_put.with_columns(
        pl.when(pl.col(ABSOLUTE_GAMMA_EXPOSURE_COLUMN) > ZERO_FLOAT)
        .then(
            (
                pl.col(CALL_GAMMA_EXPOSURE_COLUMN)
                - pl.col(PUT_GAMMA_EXPOSURE_COLUMN)
            )
            / pl.col(ABSOLUTE_GAMMA_EXPOSURE_COLUMN)
        )
        .otherwise(ZERO_FLOAT)
        .fill_null(ZERO_FLOAT)
        .alias(CALL_PUT_GAMMA_IMBALANCE_COLUMN)
    )
    return call_put.select(TRADE_DATE_COLUMN, CALL_PUT_GAMMA_IMBALANCE_COLUMN)


def _build_exposure_concentration_index(
    frame: pl.DataFrame,
    daily_totals: pl.DataFrame,
) -> pl.DataFrame:
    """Compute the Herfindahl-Hirschman Index over strike-expiry gamma shares."""

    strike_expiry_abs = frame.group_by(
        [TRADE_DATE_COLUMN, STRIKE_PRICE_COLUMN, EXPIRY_DATE_COLUMN]
    ).agg(
        pl.col(GAMMA_EXPOSURE_COLUMN)
        .abs()
        .sum()
        .alias(STRIKE_EXPIRY_ABS_GAMMA_EXPOSURE_COLUMN)
    )
    strike_expiry_share = strike_expiry_abs.join(
        daily_totals.select(TRADE_DATE_COLUMN, ABSOLUTE_GAMMA_EXPOSURE_COLUMN),
        on=TRADE_DATE_COLUMN,
        how="left",
    )
    strike_expiry_share = strike_expiry_share.with_columns(
        pl.when(pl.col(ABSOLUTE_GAMMA_EXPOSURE_COLUMN) > ZERO_FLOAT)
        .then(
            pl.col(STRIKE_EXPIRY_ABS_GAMMA_EXPOSURE_COLUMN)
            / pl.col(ABSOLUTE_GAMMA_EXPOSURE_COLUMN)
        )
        .otherwise(ZERO_FLOAT)
        .fill_null(ZERO_FLOAT)
        .alias(STRIKE_EXPIRY_GAMMA_SHARE_COLUMN)
    )
    concentration = strike_expiry_share.group_by(TRADE_DATE_COLUMN).agg(
        (
            pl.col(STRIKE_EXPIRY_GAMMA_SHARE_COLUMN)
            * pl.col(STRIKE_EXPIRY_GAMMA_SHARE_COLUMN)
        )
        .sum()
        .alias(EXPOSURE_CONCENTRATION_INDEX_COLUMN)
    )
    return concentration.sort(TRADE_DATE_COLUMN)
