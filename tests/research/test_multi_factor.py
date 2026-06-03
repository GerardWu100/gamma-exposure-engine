"""Tests for multi-factor rank correlation summaries."""

from __future__ import annotations

import polars as pl
import pytest

from gamma_exposure_engine.research.multi_factor import (
    build_factor_factor_correlations,
    build_factor_target_correlations,
)


def test_factor_target_correlations_shape() -> None:
    """Output should have one row per factor and one column per target."""

    frame = pl.DataFrame(
        {
            "factor_a": [1.0, 2.0, 3.0, 4.0, 5.0],
            "factor_b": [5.0, 4.0, 3.0, 2.0, 1.0],
            "target_x": [10.0, 20.0, 30.0, 40.0, 50.0],
            "target_y": [50.0, 40.0, 30.0, 20.0, 10.0],
        }
    )

    result = build_factor_target_correlations(
        frame=frame,
        factor_names=["factor_a", "factor_b"],
        target_names=["target_x", "target_y"],
    )

    assert result.height == 2
    assert result.columns == ["factor", "target_x", "target_y"]


def test_factor_target_detects_perfect_positive_correlation() -> None:
    """A perfectly monotonic factor-target pair should have rho 1.0."""

    frame = pl.DataFrame(
        {
            "factor_a": [1.0, 2.0, 3.0, 4.0, 5.0],
            "target_x": [10.0, 20.0, 30.0, 40.0, 50.0],
        }
    )

    result = build_factor_target_correlations(
        frame=frame,
        factor_names=["factor_a"],
        target_names=["target_x"],
    )

    assert result["target_x"].item() == pytest.approx(1.0)


def test_factor_factor_correlations_shape() -> None:
    """The factor-factor matrix should be square plus the row label column."""

    frame = pl.DataFrame(
        {
            "factor_a": [1.0, 2.0, 3.0, 4.0],
            "factor_b": [4.0, 3.0, 2.0, 1.0],
            "factor_c": [2.0, 4.0, 1.0, 3.0],
        }
    )

    result = build_factor_factor_correlations(
        frame=frame,
        factor_names=["factor_a", "factor_b", "factor_c"],
    )

    assert result.height == 3
    assert result.columns == ["factor", "factor_a", "factor_b", "factor_c"]


def test_factor_factor_diagonal_is_one() -> None:
    """Every factor should correlate perfectly with itself."""

    frame = pl.DataFrame(
        {
            "factor_a": [1.0, 2.0, 3.0, 4.0, 5.0],
            "factor_b": [5.0, 4.0, 3.0, 2.0, 1.0],
        }
    )

    result = build_factor_factor_correlations(
        frame=frame,
        factor_names=["factor_a", "factor_b"],
    )

    factor_a_row = result.filter(pl.col("factor") == "factor_a")
    assert factor_a_row["factor_a"].item() == pytest.approx(1.0)
    factor_b_row = result.filter(pl.col("factor") == "factor_b")
    assert factor_b_row["factor_b"].item() == pytest.approx(1.0)


def test_empty_frame_returns_empty() -> None:
    """Empty input should preserve the output column structure."""

    frame = pl.DataFrame(
        {
            "factor_a": pl.Series([], dtype=pl.Float64),
            "target_x": pl.Series([], dtype=pl.Float64),
        }
    )

    result = build_factor_target_correlations(
        frame=frame,
        factor_names=["factor_a"],
        target_names=["target_x"],
    )

    assert result.height == 0
    assert "factor" in result.columns
