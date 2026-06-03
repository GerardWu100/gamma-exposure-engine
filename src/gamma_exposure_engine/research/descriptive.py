"""Descriptive research helpers for aligned exposure-response frames.

The helpers in this module operate on the aligned research dataset and keep
the contract intentionally small:

- deterministic quantile summaries over a factor and target pair
- a threshold-based robustness split for the observed
  ``near_spot_gamma_share`` factor

The robustness helper is intentionally honest about what it measures. It does
not rebuild gamma factors under alternative moneyness bands. Instead, it asks
whether the association with the chosen target looks similar when the already
computed ``near_spot_gamma_share`` factor is split at several configured
thresholds.
"""

from __future__ import annotations

from typing import Final

import polars as pl
from scipy import stats

from gamma_exposure_engine.exposure.aggregation import build_daily_gamma_factors

TRADE_DATE_COLUMN: str = "trade_date"
QUANTILE_BUCKET_COLUMN: str = "quantile_bucket"
TARGET_MEAN_COLUMN: str = "target_mean"
OBSERVATION_COUNT_COLUMN: str = "observation_count"
NEAR_SPOT_SHARE_THRESHOLD_COLUMN: str = "near_spot_share_threshold"
AT_OR_ABOVE_THRESHOLD_COUNT_COLUMN: str = "at_or_above_threshold_count"
BELOW_THRESHOLD_COUNT_COLUMN: str = "below_threshold_count"
AT_OR_ABOVE_THRESHOLD_TARGET_MEAN_COLUMN: str = "at_or_above_threshold_target_mean"
BELOW_THRESHOLD_TARGET_MEAN_COLUMN: str = "below_threshold_target_mean"
TARGET_MEAN_SPREAD_COLUMN: str = "target_mean_spread"
NEAR_SPOT_GAMMA_SHARE_COLUMN: str = "near_spot_gamma_share"
BUCKET_ASSIGNMENT_COLUMN: str = "_quantile_bucket_assignment"
ROW_INDEX_COLUMN: str = "_row_index"
SUBPERIOD_COLUMN: str = "subperiod"
SPEARMAN_RHO_COLUMN: str = "spearman_rho"
P_VALUE_COLUMN: str = "p_value"
BAND_WIDTH_COLUMN: str = "band_width"
DROPPED_MONTH_COLUMN: str = "dropped_month"
YEAR_MONTH_COLUMN: str = "_year_month"
FIRST_HALF_LABEL: str = "first_half"
SECOND_HALF_LABEL: str = "second_half"
ONE_INT: Final[int] = 1

__all__ = [
    "build_alternative_band_sensitivity",
    "build_leave_one_month_out_sensitivity",
    "build_near_spot_share_threshold_summary",
    "build_quantile_summary",
    "build_subperiod_stability",
]


def build_quantile_summary(
    frame: pl.DataFrame,
    factor_name: str,
    target_name: str,
    quantiles: int,
) -> pl.DataFrame:
    """Summarize the target mean inside deterministic quantile buckets.

    Args:
        frame:
            Research frame with one row per trade date. The frame must include
            the factor column and the target column named by ``factor_name``
            and ``target_name``.
        factor_name:
            Name of the factor column used to sort rows before bucket
            assignment.
        target_name:
            Name of the target column whose mean is summarized inside each
            quantile bucket.
        quantiles:
            Number of buckets to assign. Buckets are labeled from ``0`` to
            ``quantiles - 1``.

    Returns:
        pl.DataFrame: One row per observed bucket with the bucket label,
        target mean, and observation count.
    """

    if quantiles < ONE_INT:
        msg = "quantiles must be at least 1"
        raise ValueError(msg)

    ordered_frame = _sort_frame_for_quantiles(frame=frame, factor_name=factor_name)
    if ordered_frame.height == 0:
        return pl.DataFrame(
            {
                QUANTILE_BUCKET_COLUMN: pl.Series([], dtype=pl.Int64),
                TARGET_MEAN_COLUMN: pl.Series([], dtype=pl.Float64),
                OBSERVATION_COUNT_COLUMN: pl.Series([], dtype=pl.Int64),
            }
        )

    bucket_labels = _build_quantile_buckets(
        row_count=ordered_frame.height,
        quantiles=quantiles,
    )
    bucket_labels = _compact_bucket_labels(bucket_labels)
    ordered_frame = ordered_frame.with_columns(
        pl.Series(BUCKET_ASSIGNMENT_COLUMN, bucket_labels)
    )
    summary = ordered_frame.group_by(BUCKET_ASSIGNMENT_COLUMN).agg(
        pl.col(target_name).mean().cast(pl.Float64).alias(TARGET_MEAN_COLUMN),
        pl.len().cast(pl.Int64).alias(OBSERVATION_COUNT_COLUMN),
    )
    return (
        summary.rename({BUCKET_ASSIGNMENT_COLUMN: QUANTILE_BUCKET_COLUMN})
        .with_columns(pl.col(QUANTILE_BUCKET_COLUMN).cast(pl.Int64))
        .sort(QUANTILE_BUCKET_COLUMN)
    )


def build_near_spot_share_threshold_summary(
    frame: pl.DataFrame,
    target_name: str,
    thresholds: list[float] | tuple[float, ...],
) -> pl.DataFrame:
    """Summarize target means above and below observed share thresholds.

    Args:
        frame:
            Research frame with one row per trade date. The frame must include
            the observed ``near_spot_gamma_share`` factor plus the target
            column named by ``target_name``.
        target_name:
            Name of the target column whose mean is summarized on each side of
            every threshold split.
        thresholds:
            Share cutoffs applied to the already-computed
            ``near_spot_gamma_share`` column. A threshold of ``0.20`` means
            rows are split into ``near_spot_gamma_share >= 0.20`` and
            ``near_spot_gamma_share < 0.20``.

    Returns:
        pl.DataFrame: One row per threshold with counts and target means on
        both sides of the split plus the mean spread.
    """

    ordered_thresholds = sorted(thresholds)
    if not ordered_thresholds:
        return pl.DataFrame(
            {
                NEAR_SPOT_SHARE_THRESHOLD_COLUMN: pl.Series([], dtype=pl.Float64),
                AT_OR_ABOVE_THRESHOLD_COUNT_COLUMN: pl.Series([], dtype=pl.Int64),
                BELOW_THRESHOLD_COUNT_COLUMN: pl.Series([], dtype=pl.Int64),
                AT_OR_ABOVE_THRESHOLD_TARGET_MEAN_COLUMN: pl.Series(
                    [],
                    dtype=pl.Float64,
                ),
                BELOW_THRESHOLD_TARGET_MEAN_COLUMN: pl.Series([], dtype=pl.Float64),
                TARGET_MEAN_SPREAD_COLUMN: pl.Series([], dtype=pl.Float64),
            }
        )

    summary_rows: list[dict[str, float | int | None]] = []
    for threshold in ordered_thresholds:
        at_or_above_frame = frame.filter(
            pl.col(NEAR_SPOT_GAMMA_SHARE_COLUMN) >= threshold
        )
        below_frame = frame.filter(pl.col(NEAR_SPOT_GAMMA_SHARE_COLUMN) < threshold)

        at_or_above_target_mean = _mean_or_none(
            frame=at_or_above_frame,
            column_name=target_name,
        )
        below_target_mean = _mean_or_none(
            frame=below_frame,
            column_name=target_name,
        )
        target_mean_spread = _spread_or_none(
            left_value=at_or_above_target_mean,
            right_value=below_target_mean,
        )

        summary_rows.append(
            {
                NEAR_SPOT_SHARE_THRESHOLD_COLUMN: float(threshold),
                AT_OR_ABOVE_THRESHOLD_COUNT_COLUMN: at_or_above_frame.height,
                BELOW_THRESHOLD_COUNT_COLUMN: below_frame.height,
                AT_OR_ABOVE_THRESHOLD_TARGET_MEAN_COLUMN: at_or_above_target_mean,
                BELOW_THRESHOLD_TARGET_MEAN_COLUMN: below_target_mean,
                TARGET_MEAN_SPREAD_COLUMN: target_mean_spread,
            }
        )

    return pl.DataFrame(summary_rows)


def _build_quantile_buckets(row_count: int, quantiles: int) -> list[int]:
    """Assign each row to a deterministic integer bucket."""

    bucket_labels: list[int] = []
    for row_index in range(row_count):
        # Scale the zero-based row index into the integer bucket range
        # ``[0, quantiles - 1]`` without relying on percentile interpolation.
        bucket_index = (row_index * quantiles) // row_count
        bucket_labels.append(bucket_index)
    return bucket_labels


def _compact_bucket_labels(bucket_labels: list[int]) -> list[int]:
    """Map sparse bucket labels onto a contiguous zero-based range."""

    compact_mapping: dict[int, int] = {}
    compact_labels: list[int] = []
    next_compact_label = 0
    for bucket_label in bucket_labels:
        if bucket_label not in compact_mapping:
            compact_mapping[bucket_label] = next_compact_label
            next_compact_label += 1
        compact_labels.append(compact_mapping[bucket_label])
    return compact_labels


def _sort_frame_for_quantiles(frame: pl.DataFrame, factor_name: str) -> pl.DataFrame:
    """Sort factor rows with a deterministic tie-breaker."""

    indexed_frame = frame.with_row_index(name=ROW_INDEX_COLUMN)
    sort_columns = [factor_name]
    if TRADE_DATE_COLUMN in indexed_frame.columns:
        sort_columns.append(TRADE_DATE_COLUMN)
    sort_columns.append(ROW_INDEX_COLUMN)
    return indexed_frame.sort(sort_columns)


def _mean_or_none(frame: pl.DataFrame, column_name: str) -> float | None:
    """Return a scalar mean or ``None`` when the input frame is empty."""

    if frame.height == 0:
        return None

    mean_value = frame.select(pl.col(column_name).mean()).item()
    if mean_value is None:
        return None

    return float(mean_value)


def _spread_or_none(
    left_value: float | None, right_value: float | None
) -> float | None:
    """Return a difference only when both mean inputs are present."""

    if left_value is None or right_value is None:
        return None

    return left_value - right_value


def build_subperiod_stability(
    frame: pl.DataFrame,
    factor_name: str,
    target_name: str,
) -> pl.DataFrame:
    """Split the sample at the temporal midpoint and compare Spearman results.

    Parameters
    ----------
    frame:
        Research dataset with one row per trade date.
    factor_name:
        Factor column used for Spearman rank correlation.
    target_name:
        Target column used for Spearman rank correlation.

    Returns
    -------
    pl.DataFrame
        Two rows, one for the first half and one for the second half.
    """

    sorted_frame = frame.sort(TRADE_DATE_COLUMN)
    midpoint = sorted_frame.height // 2
    first_half = sorted_frame.head(midpoint)
    second_half = sorted_frame.tail(sorted_frame.height - midpoint)

    summary_rows = [
        _build_spearman_summary_row(
            frame=first_half,
            left_column=factor_name,
            right_column=target_name,
            label_column=SUBPERIOD_COLUMN,
            label_value=FIRST_HALF_LABEL,
        ),
        _build_spearman_summary_row(
            frame=second_half,
            left_column=factor_name,
            right_column=target_name,
            label_column=SUBPERIOD_COLUMN,
            label_value=SECOND_HALF_LABEL,
        ),
    ]
    return pl.DataFrame(summary_rows)


def build_alternative_band_sensitivity(
    cleaned_options: pl.DataFrame,
    targets: pl.DataFrame,
    target_name: str,
    band_widths: list[float] | tuple[float, ...],
) -> pl.DataFrame:
    """Recompute near-spot share under alternative band widths.

    Parameters
    ----------
    cleaned_options:
        Cleaned option rows ready for gamma-factor aggregation.
    targets:
        Daily target values keyed by ``trade_date``.
    target_name:
        Target column used in the sensitivity correlation.
    band_widths:
        Alternative near-spot moneyness bands to test.

    Returns
    -------
    pl.DataFrame
        One row per tested band width.
    """

    summary_rows: list[dict[str, object]] = []
    for band_width in sorted(band_widths):
        factors = build_daily_gamma_factors(
            frame=cleaned_options,
            near_spot_band=band_width,
        )
        merged_frame = factors.join(targets, on=TRADE_DATE_COLUMN, how="inner")
        summary_rows.append(
            _build_spearman_summary_row(
                frame=merged_frame,
                left_column=NEAR_SPOT_GAMMA_SHARE_COLUMN,
                right_column=target_name,
                label_column=BAND_WIDTH_COLUMN,
                label_value=float(band_width),
            )
        )

    return pl.DataFrame(summary_rows)


def build_leave_one_month_out_sensitivity(
    frame: pl.DataFrame,
    factor_name: str,
    target_name: str,
) -> pl.DataFrame:
    """Drop one calendar month at a time and recompute Spearman correlation.

    Parameters
    ----------
    frame:
        Research dataset with one row per trade date.
    factor_name:
        Factor column used for Spearman rank correlation.
    target_name:
        Target column used for Spearman rank correlation.

    Returns
    -------
    pl.DataFrame
        One row per dropped calendar month.
    """

    month_frame = frame.sort(TRADE_DATE_COLUMN).with_columns(
        pl.col(TRADE_DATE_COLUMN).dt.strftime("%Y-%m").alias(YEAR_MONTH_COLUMN)
    )
    month_labels = month_frame.get_column(YEAR_MONTH_COLUMN).unique().sort().to_list()

    summary_rows: list[dict[str, object]] = []
    for month_label in month_labels:
        remaining_frame = month_frame.filter(pl.col(YEAR_MONTH_COLUMN) != month_label)
        summary_rows.append(
            _build_spearman_summary_row(
                frame=remaining_frame,
                left_column=factor_name,
                right_column=target_name,
                label_column=DROPPED_MONTH_COLUMN,
                label_value=month_label,
            )
        )

    return pl.DataFrame(summary_rows)


def _build_spearman_summary_row(
    frame: pl.DataFrame,
    left_column: str,
    right_column: str,
    label_column: str,
    label_value: str | float,
) -> dict[str, object]:
    """Return one labeled Spearman summary row after pairwise cleaning.

    Parameters
    ----------
    frame:
        Input rows that may include nulls, non-finite values, or repeated
        values that make rank correlation undefined.
    left_column:
        Left-hand variable passed to Spearman rank correlation.
    right_column:
        Right-hand variable passed to Spearman rank correlation.
    label_column:
        Output column name that identifies the current split or scenario.
    label_value:
        Output label value for the current split or scenario.

    Returns
    -------
    dict[str, object]
        One summary row with the label, Spearman rho, p-value, and the
        observation count used after pairwise cleaning.
    """

    # Apply one pairwise cleaning pass so all validity checks below operate on
    # the exact same sample.
    clean_frame = frame.drop_nulls([left_column, right_column]).filter(
        pl.col(left_column).is_finite(),
        pl.col(right_column).is_finite(),
    )

    # Spearman requires at least three rows and variability in both columns.
    has_enough_rows = clean_frame.height >= 3
    left_has_variation = clean_frame.get_column(left_column).n_unique() > 1
    right_has_variation = clean_frame.get_column(right_column).n_unique() > 1
    if not (has_enough_rows and left_has_variation and right_has_variation):
        return _nan_spearman_row(
            label_column=label_column,
            label_value=label_value,
            observation_count=clean_frame.height,
        )

    result = stats.spearmanr(
        clean_frame.get_column(left_column).to_numpy(),
        clean_frame.get_column(right_column).to_numpy(),
    )
    return {
        label_column: label_value,
        SPEARMAN_RHO_COLUMN: float(result.statistic),
        P_VALUE_COLUMN: float(result.pvalue),
        OBSERVATION_COUNT_COLUMN: clean_frame.height,
    }


def _nan_spearman_row(
    label_column: str,
    label_value: str | float,
    observation_count: int,
) -> dict[str, object]:
    """Build a standardized summary row for undefined Spearman cases."""

    return {
        label_column: label_value,
        SPEARMAN_RHO_COLUMN: float("nan"),
        P_VALUE_COLUMN: float("nan"),
        OBSERVATION_COUNT_COLUMN: observation_count,
    }
