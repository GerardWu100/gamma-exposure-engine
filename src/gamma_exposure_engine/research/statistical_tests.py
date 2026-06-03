"""Non-parametric statistical tests for factor-target relationships.

The descriptive report already shows bucket means, but interview-quality
research also needs formal tests. This module adds two standard non-parametric
statistics:

- Kruskal-Wallis H-test: compares target distributions across quantile buckets
- Spearman rank correlation: measures monotonic association across the sample
"""

from __future__ import annotations

from typing import Final

import polars as pl
from scipy import stats

from gamma_exposure_engine.research.descriptive import (
    BUCKET_ASSIGNMENT_COLUMN,
    QUANTILE_BUCKET_COLUMN,
    _build_quantile_buckets,
    _compact_bucket_labels,
    _sort_frame_for_quantiles,
)

TEST_NAME_COLUMN: Final[str] = "test_name"
TEST_STATISTIC_COLUMN: Final[str] = "test_statistic"
P_VALUE_COLUMN: Final[str] = "p_value"
SAMPLE_SIZE_COLUMN: Final[str] = "sample_size"

STATISTICAL_TEST_SCHEMA: dict[str, pl.DataType] = {
    TEST_NAME_COLUMN: pl.String,
    TEST_STATISTIC_COLUMN: pl.Float64,
    P_VALUE_COLUMN: pl.Float64,
    SAMPLE_SIZE_COLUMN: pl.Int64,
}

__all__ = ["build_statistical_test_summary"]


def build_statistical_test_summary(
    frame: pl.DataFrame,
    factor_name: str,
    target_name: str,
    quantiles: int,
) -> pl.DataFrame:
    """Run the report's non-parametric significance tests.

    Parameters
    ----------
    frame:
        Research dataset with one row per trade date.
    factor_name:
        Factor column used in the descriptive quantile sort.
    target_name:
        Target column whose association with the factor is tested.
    quantiles:
        Requested bucket count used by the Kruskal-Wallis partition.

    Returns
    -------
    pl.DataFrame
        Two rows, one for Spearman and one for Kruskal-Wallis, with explicit
        schema for report rendering.
    """

    if frame.height == 0:
        return _empty_test_summary()

    clean_frame = frame.drop_nulls([factor_name, target_name]).filter(
        pl.col(factor_name).is_finite(),
        pl.col(target_name).is_finite(),
    )
    if clean_frame.height == 0:
        return _empty_test_summary()

    sample_size = clean_frame.height
    factor_has_variation = clean_frame.get_column(factor_name).n_unique() > 1
    target_has_variation = clean_frame.get_column(target_name).n_unique() > 1
    if factor_has_variation and target_has_variation:
        spearman_result = stats.spearmanr(
            clean_frame.get_column(factor_name).to_numpy(),
            clean_frame.get_column(target_name).to_numpy(),
        )
        spearman_statistic = float(spearman_result.statistic)
        spearman_p_value = float(spearman_result.pvalue)
    else:
        # Constant columns make rank correlation undefined; keep explicit NaNs.
        spearman_statistic = float("nan")
        spearman_p_value = float("nan")
    kruskal_result = _run_kruskal_wallis(
        frame=clean_frame,
        factor_name=factor_name,
        target_name=target_name,
        quantiles=quantiles,
        sample_size=sample_size,
    )

    summary_rows = [
        {
            TEST_NAME_COLUMN: "spearman_rank_correlation",
            TEST_STATISTIC_COLUMN: spearman_statistic,
            P_VALUE_COLUMN: spearman_p_value,
            SAMPLE_SIZE_COLUMN: sample_size,
        },
        kruskal_result,
    ]
    return pl.DataFrame(summary_rows, schema=STATISTICAL_TEST_SCHEMA)


def _run_kruskal_wallis(
    frame: pl.DataFrame,
    factor_name: str,
    target_name: str,
    quantiles: int,
    sample_size: int,
) -> dict[str, object]:
    """Partition the sample into quantile buckets, then run Kruskal-Wallis."""

    bucketed_frame = _assign_quantile_buckets(
        frame=frame,
        factor_name=factor_name,
        quantiles=quantiles,
    )

    bucket_labels = bucketed_frame.get_column(QUANTILE_BUCKET_COLUMN).unique().sort()
    group_values: list[object] = []
    for bucket_label in bucket_labels.to_list():
        bucket_targets = (
            bucketed_frame.filter(pl.col(QUANTILE_BUCKET_COLUMN) == bucket_label)
            .get_column(target_name)
            .to_numpy()
        )
        group_values.append(bucket_targets)

    if len(group_values) < 2:
        return {
            TEST_NAME_COLUMN: "kruskal_wallis",
            TEST_STATISTIC_COLUMN: float("nan"),
            P_VALUE_COLUMN: float("nan"),
            SAMPLE_SIZE_COLUMN: sample_size,
        }

    kruskal = stats.kruskal(*group_values)
    return {
        TEST_NAME_COLUMN: "kruskal_wallis",
        TEST_STATISTIC_COLUMN: float(kruskal.statistic),
        P_VALUE_COLUMN: float(kruskal.pvalue),
        SAMPLE_SIZE_COLUMN: sample_size,
    }


def _assign_quantile_buckets(
    frame: pl.DataFrame,
    factor_name: str,
    quantiles: int,
) -> pl.DataFrame:
    """Apply the descriptive layer's deterministic bucket assignment."""

    ordered_frame = _sort_frame_for_quantiles(frame=frame, factor_name=factor_name)
    bucket_labels = _build_quantile_buckets(
        row_count=ordered_frame.height,
        quantiles=quantiles,
    )
    compact_bucket_labels = _compact_bucket_labels(bucket_labels)
    return ordered_frame.with_columns(
        pl.Series(
            name=BUCKET_ASSIGNMENT_COLUMN,
            values=compact_bucket_labels,
            dtype=pl.Int64,
        )
    ).rename({BUCKET_ASSIGNMENT_COLUMN: QUANTILE_BUCKET_COLUMN})


def _empty_test_summary() -> pl.DataFrame:
    """Return the public empty statistical-summary schema."""

    return pl.DataFrame(
        {
            TEST_NAME_COLUMN: pl.Series([], dtype=pl.String),
            TEST_STATISTIC_COLUMN: pl.Series([], dtype=pl.Float64),
            P_VALUE_COLUMN: pl.Series([], dtype=pl.Float64),
            SAMPLE_SIZE_COLUMN: pl.Series([], dtype=pl.Int64),
        }
    )
