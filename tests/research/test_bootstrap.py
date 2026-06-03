"""Tests for bootstrap confidence interval computation.

The bootstrap helper should preserve the deterministic quantile summary while
adding reproducible percentile confidence intervals around each bucket mean.
"""

from __future__ import annotations

import polars as pl

from gamma_exposure_engine.research.bootstrap import build_quantile_summary_with_ci


def test_ci_bounds_bracket_point_estimate() -> None:
    """Each percentile interval should contain its point estimate."""

    frame = pl.DataFrame(
        {
            "factor_value": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
            "target_value": [
                10.0,
                20.0,
                30.0,
                40.0,
                50.0,
                60.0,
                70.0,
                80.0,
                90.0,
                100.0,
            ],
        }
    )

    summary = build_quantile_summary_with_ci(
        frame=frame,
        factor_name="factor_value",
        target_name="target_value",
        quantiles=2,
        bootstrap_iterations=500,
        confidence_level=0.95,
        random_seed=42,
    )

    assert "ci_lower" in summary.columns
    assert "ci_upper" in summary.columns
    assert "target_mean" in summary.columns
    for row in summary.to_dicts():
        assert row["ci_lower"] <= row["target_mean"] <= row["ci_upper"]


def test_ci_reproducible_with_fixed_seed() -> None:
    """The same random seed should reproduce identical interval bounds."""

    frame = pl.DataFrame(
        {
            "factor_value": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "target_value": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
        }
    )

    summary_a = build_quantile_summary_with_ci(
        frame=frame,
        factor_name="factor_value",
        target_name="target_value",
        quantiles=2,
        bootstrap_iterations=200,
        confidence_level=0.95,
        random_seed=123,
    )
    summary_b = build_quantile_summary_with_ci(
        frame=frame,
        factor_name="factor_value",
        target_name="target_value",
        quantiles=2,
        bootstrap_iterations=200,
        confidence_level=0.95,
        random_seed=123,
    )

    assert summary_a["ci_lower"].to_list() == summary_b["ci_lower"].to_list()
    assert summary_a["ci_upper"].to_list() == summary_b["ci_upper"].to_list()


def test_ci_empty_frame_returns_empty_with_ci_columns() -> None:
    """Empty input should keep the public schema including CI columns."""

    frame = pl.DataFrame(
        {
            "factor_value": pl.Series([], dtype=pl.Float64),
            "target_value": pl.Series([], dtype=pl.Float64),
        }
    )

    summary = build_quantile_summary_with_ci(
        frame=frame,
        factor_name="factor_value",
        target_name="target_value",
        quantiles=3,
        bootstrap_iterations=100,
        confidence_level=0.95,
        random_seed=42,
    )

    assert summary.height == 0
    assert "ci_lower" in summary.columns
    assert "ci_upper" in summary.columns
    assert "quantile_bucket" in summary.columns
    assert "target_mean" in summary.columns
    assert "observation_count" in summary.columns


def test_ci_output_schema_matches_expected_types() -> None:
    """The bootstrap summary should preserve explicit numeric dtypes."""

    frame = pl.DataFrame(
        {
            "factor_value": [1.0, 2.0, 3.0, 4.0],
            "target_value": [10.0, 20.0, 30.0, 40.0],
        }
    )

    summary = build_quantile_summary_with_ci(
        frame=frame,
        factor_name="factor_value",
        target_name="target_value",
        quantiles=2,
        bootstrap_iterations=100,
        confidence_level=0.90,
        random_seed=42,
    )

    assert summary.schema["ci_lower"] == pl.Float64
    assert summary.schema["ci_upper"] == pl.Float64
    assert summary.schema["quantile_bucket"] == pl.Int64
    assert summary.schema["target_mean"] == pl.Float64
    assert summary.schema["observation_count"] == pl.Int64
