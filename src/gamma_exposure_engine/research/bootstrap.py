"""Bootstrap confidence intervals for descriptive quantile summaries.

This module extends the deterministic quantile summary with nonparametric
row-resampling. Each bootstrap iteration samples full research rows with
replacement, recomputes the bucket means, and stores the resulting target
means by quantile bucket. The final interval is the percentile band from that
bootstrap distribution.
"""

from __future__ import annotations

from typing import Final

import numpy as np
import polars as pl

from gamma_exposure_engine.research.descriptive import (
    OBSERVATION_COUNT_COLUMN,
    QUANTILE_BUCKET_COLUMN,
    TARGET_MEAN_COLUMN,
    build_quantile_summary,
)

CI_LOWER_COLUMN: Final[str] = "ci_lower"
CI_UPPER_COLUMN: Final[str] = "ci_upper"

__all__ = ["build_quantile_summary_with_ci"]


def build_quantile_summary_with_ci(
    frame: pl.DataFrame,
    factor_name: str,
    target_name: str,
    quantiles: int,
    bootstrap_iterations: int,
    confidence_level: float,
    random_seed: int = 42,
) -> pl.DataFrame:
    """Build the quantile summary with percentile bootstrap intervals.

    Parameters
    ----------
    frame:
        Research dataset with one row per trade date.
    factor_name:
        Column used to sort observations into deterministic quantile buckets.
    target_name:
        Column whose bucket mean is summarized and bootstrapped.
    quantiles:
        Requested number of quantile buckets.
    bootstrap_iterations:
        Number of bootstrap resamples. Each resample draws ``frame.height``
        rows with replacement.
    confidence_level:
        Two-sided confidence level between ``0`` and ``1``.
    random_seed:
        Seed for NumPy's random number generator so test runs and report
        builds stay reproducible.

    Returns
    -------
    pl.DataFrame
        Quantile summary with the deterministic point estimate columns plus
        ``ci_lower`` and ``ci_upper``.
    """

    if bootstrap_iterations < 1:
        msg = "bootstrap_iterations must be at least 1"
        raise ValueError(msg)

    if confidence_level <= 0.0 or confidence_level >= 1.0:
        msg = "confidence_level must be strictly between 0 and 1"
        raise ValueError(msg)

    # The point estimate remains the existing deterministic descriptive
    # summary. Bootstrap only adds uncertainty bands around that estimate.
    point_summary = build_quantile_summary(
        frame=frame,
        factor_name=factor_name,
        target_name=target_name,
        quantiles=quantiles,
    )

    if point_summary.height == 0:
        return _build_empty_summary_with_ci()

    bucket_labels = point_summary.get_column(QUANTILE_BUCKET_COLUMN).to_list()
    bootstrap_means = _collect_bootstrap_means(
        frame=frame,
        factor_name=factor_name,
        target_name=target_name,
        quantiles=quantiles,
        bootstrap_iterations=bootstrap_iterations,
        bucket_labels=bucket_labels,
        random_seed=random_seed,
    )

    lower_percentile, upper_percentile = _two_sided_percentile_bounds(
        confidence_level=confidence_level,
    )

    ci_lower_values: list[float] = []
    ci_upper_values: list[float] = []
    for bucket_label in bucket_labels:
        bucket_distribution = np.array(
            bootstrap_means[bucket_label],
            dtype=float,
        )
        # Some resamples can collapse a bucket on very small samples. Using
        # NaN-aware percentiles preserves the available draws without
        # pretending the missing bucket had a value of zero.
        ci_lower_values.append(
            float(np.nanpercentile(bucket_distribution, lower_percentile))
        )
        ci_upper_values.append(
            float(np.nanpercentile(bucket_distribution, upper_percentile))
        )

    return point_summary.with_columns(
        pl.Series(CI_LOWER_COLUMN, ci_lower_values, dtype=pl.Float64),
        pl.Series(CI_UPPER_COLUMN, ci_upper_values, dtype=pl.Float64),
    )


def _collect_bootstrap_means(
    frame: pl.DataFrame,
    factor_name: str,
    target_name: str,
    quantiles: int,
    bootstrap_iterations: int,
    bucket_labels: list[int],
    random_seed: int,
) -> dict[int, list[float]]:
    """Collect one bootstrap target-mean draw per observed bucket.

    Each bootstrap sample preserves the factor-target row pairing by
    resampling full rows instead of marginal columns.
    """

    random_number_generator = np.random.default_rng(random_seed)
    row_count = frame.height
    bootstrap_means: dict[int, list[float]] = {
        bucket_label: [] for bucket_label in bucket_labels
    }

    for _ in range(bootstrap_iterations):
        sampled_indices = random_number_generator.integers(
            low=0,
            high=row_count,
            size=row_count,
        )
        resampled_frame = frame[sampled_indices.tolist()]
        resampled_summary = build_quantile_summary(
            frame=resampled_frame,
            factor_name=factor_name,
            target_name=target_name,
            quantiles=quantiles,
        )
        resampled_means_by_bucket = dict(
            zip(
                resampled_summary.get_column(QUANTILE_BUCKET_COLUMN).to_list(),
                resampled_summary.get_column(TARGET_MEAN_COLUMN).to_list(),
                strict=True,
            )
        )

        for bucket_label in bucket_labels:
            bootstrap_means[bucket_label].append(
                float(resampled_means_by_bucket.get(bucket_label, float("nan")))
            )

    return bootstrap_means


def _two_sided_percentile_bounds(
    confidence_level: float,
) -> tuple[float, float]:
    """Return lower and upper percentile cutoffs for a two-sided interval."""

    alpha = 1.0 - confidence_level
    lower_percentile = 100.0 * (alpha / 2.0)
    upper_percentile = 100.0 * (1.0 - (alpha / 2.0))
    return lower_percentile, upper_percentile


def _build_empty_summary_with_ci() -> pl.DataFrame:
    """Return the public empty output schema for bootstrap summaries."""

    return pl.DataFrame(
        {
            QUANTILE_BUCKET_COLUMN: pl.Series([], dtype=pl.Int64),
            TARGET_MEAN_COLUMN: pl.Series([], dtype=pl.Float64),
            OBSERVATION_COUNT_COLUMN: pl.Series([], dtype=pl.Int64),
            CI_LOWER_COLUMN: pl.Series([], dtype=pl.Float64),
            CI_UPPER_COLUMN: pl.Series([], dtype=pl.Float64),
        }
    )
