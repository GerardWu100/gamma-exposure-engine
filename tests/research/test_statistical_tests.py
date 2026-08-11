"""Tests for non-parametric statistical test helpers."""

from __future__ import annotations

import polars as pl
import pytest
from gamma_exposure_engine.research.statistical_tests import (
    build_statistical_test_summary,
)


def test_detects_known_monotonic_relationship() -> None:
    """Spearman rho should detect a perfectly monotonic relationship."""

    frame = pl.DataFrame(
        {
            "factor_value": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            "target_value": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0],
        }
    )

    summary = build_statistical_test_summary(
        frame=frame,
        factor_name="factor_value",
        target_name="target_value",
        quantiles=2,
    )

    spearman_row = summary.filter(pl.col("test_name") == "spearman_rank_correlation")
    assert spearman_row.height == 1
    assert spearman_row["test_statistic"].item() == pytest.approx(1.0)
    assert spearman_row["p_value"].item() < 0.05


def test_detects_bucket_difference_with_separated_distributions() -> None:
    """Kruskal-Wallis should reject when bucket targets are far apart."""

    frame = pl.DataFrame(
        {
            "factor_value": [1.0, 2.0, 3.0, 4.0, 100.0, 200.0, 300.0, 400.0],
            "target_value": [1.0, 1.5, 2.0, 2.5, 100.0, 100.5, 101.0, 101.5],
        }
    )

    summary = build_statistical_test_summary(
        frame=frame,
        factor_name="factor_value",
        target_name="target_value",
        quantiles=2,
    )

    kruskal_row = summary.filter(pl.col("test_name") == "kruskal_wallis")
    assert kruskal_row.height == 1
    assert kruskal_row["p_value"].item() < 0.05


def test_output_schema() -> None:
    """Output should expose a compact four-column test table."""

    frame = pl.DataFrame(
        {
            "factor_value": [1.0, 2.0, 3.0, 4.0],
            "target_value": [10.0, 20.0, 30.0, 40.0],
        }
    )

    summary = build_statistical_test_summary(
        frame=frame,
        factor_name="factor_value",
        target_name="target_value",
        quantiles=2,
    )

    assert summary.columns == ["test_name", "test_statistic", "p_value", "sample_size"]
    assert summary.schema["test_name"] == pl.String
    assert summary.schema["test_statistic"] == pl.Float64
    assert summary.schema["p_value"] == pl.Float64
    assert summary.schema["sample_size"] == pl.Int64
    assert summary.height == 2


def test_empty_frame_returns_empty_with_schema() -> None:
    """Empty input should return the same schema without any rows."""

    frame = pl.DataFrame(
        {
            "factor_value": pl.Series([], dtype=pl.Float64),
            "target_value": pl.Series([], dtype=pl.Float64),
        }
    )

    summary = build_statistical_test_summary(
        frame=frame,
        factor_name="factor_value",
        target_name="target_value",
        quantiles=2,
    )

    assert summary.height == 0
    assert summary.columns == ["test_name", "test_statistic", "p_value", "sample_size"]
