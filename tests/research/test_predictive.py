"""Tests for the research predictive helper functions.

The predictive layer should stay time-series safe: it must sort by trade date,
train only on strictly prior rows, and expose a simple lagged-target baseline.
"""

from __future__ import annotations

from datetime import date

import polars as pl
import pytest

from gamma_exposure_engine.research.predictive import (
    add_naive_lagged_target_baseline,
    add_naive_volatility_baseline,
    build_expanding_window_diagnostics,
    build_prediction_intervals,
    build_predictive_baseline_comparison,
    walk_forward_linear_baseline,
    walk_forward_ridge_baseline,
)


def test_walk_forward_linear_baseline_emits_only_out_of_sample_predictions() -> None:
    """The first prediction should use only the initial training window."""

    frame = pl.DataFrame(
        {
            "trade_date": [
                date(2024, 1, 3),
                date(2024, 1, 1),
                date(2024, 1, 2),
                date(2024, 1, 4),
            ],
            "feature_value": [3.0, 1.0, 2.0, 4.0],
            "target_value": [3.0, 1.0, 2.0, 200.0],
        }
    )

    prediction_frame = walk_forward_linear_baseline(
        frame=frame,
        feature_names=["feature_value"],
        target_name="target_value",
        min_train_size=2,
    )

    assert prediction_frame.columns == ["trade_date", "actual", "prediction"]
    assert prediction_frame.height == 2
    assert prediction_frame["trade_date"].to_list() == [
        date(2024, 1, 3),
        date(2024, 1, 4),
    ]
    assert prediction_frame["actual"].to_list() == [3.0, 200.0]
    assert prediction_frame["prediction"].to_list()[0] == pytest.approx(3.0)


def test_add_naive_volatility_baseline_uses_one_day_lag() -> None:
    """The naive baseline should be a one-row lag after sorting by trade date."""

    frame = pl.DataFrame(
        {
            "trade_date": [
                date(2024, 1, 3),
                date(2024, 1, 1),
                date(2024, 1, 2),
            ],
            "target_value": [30.0, 10.0, 20.0],
        }
    )

    baseline_frame = add_naive_volatility_baseline(
        frame=frame,
        target_name="target_value",
    )

    assert baseline_frame.columns == [
        "trade_date",
        "target_value",
        "naive_lagged_target",
    ]
    assert baseline_frame["trade_date"].to_list() == [
        date(2024, 1, 1),
        date(2024, 1, 2),
        date(2024, 1, 3),
    ]
    assert baseline_frame["naive_lagged_target"].to_list() == [
        None,
        10.0,
        20.0,
    ]


def test_add_naive_lagged_target_baseline_matches_legacy_alias() -> None:
    """The clearer baseline name should preserve the legacy function behavior."""

    frame = pl.DataFrame(
        {
            "trade_date": [
                date(2024, 1, 3),
                date(2024, 1, 1),
                date(2024, 1, 2),
            ],
            "target_value": [30.0, 10.0, 20.0],
        }
    )

    legacy_result = add_naive_volatility_baseline(
        frame=frame,
        target_name="target_value",
    )
    renamed_result = add_naive_lagged_target_baseline(
        frame=frame,
        target_name="target_value",
    )

    assert renamed_result.schema == legacy_result.schema
    assert renamed_result.to_dicts() == legacy_result.to_dicts()


def test_walk_forward_linear_baseline_empty_output_uses_explicit_schema() -> None:
    """Empty walk-forward output should keep the same typed schema contract."""

    frame = pl.DataFrame(
        {
            "trade_date": [date(2024, 1, 1), date(2024, 1, 2)],
            "feature_value": [1.0, 2.0],
            "target_value": [1.0, 2.0],
        }
    )

    prediction_frame = walk_forward_linear_baseline(
        frame=frame,
        feature_names=["feature_value"],
        target_name="target_value",
        min_train_size=2,
    )

    assert prediction_frame.columns == ["trade_date", "actual", "prediction"]
    assert prediction_frame.schema == {
        "trade_date": pl.Date,
        "actual": pl.Float64,
        "prediction": pl.Float64,
    }
    assert prediction_frame.height == 0


def test_build_predictive_baseline_comparison_uses_same_oos_window() -> None:
    """Model comparison should evaluate both baselines on the same dates."""

    frame = pl.DataFrame(
        {
            "trade_date": [
                date(2024, 1, 1),
                date(2024, 1, 2),
                date(2024, 1, 3),
                date(2024, 1, 4),
                date(2024, 1, 5),
            ],
            "feature_value": [1.0, 2.0, 3.0, 4.0, 5.0],
            "target_value": [1.0, 2.0, 3.0, 4.0, 5.0],
        }
    )

    comparison = build_predictive_baseline_comparison(
        frame=frame,
        feature_name="feature_value",
        target_name="target_value",
        min_train_size=2,
    )

    assert comparison["model_name"].to_list() == [
        "feature_linear_baseline",
        "feature_ridge_baseline",
        "naive_lagged_target_baseline",
    ]
    assert comparison["observation_count"].to_list() == [3, 3, 3]
    linear_error = comparison.filter(pl.col("model_name") == "feature_linear_baseline")[
        "mean_absolute_error"
    ].item()
    ridge_error = comparison.filter(pl.col("model_name") == "feature_ridge_baseline")[
        "mean_absolute_error"
    ].item()
    naive_error = comparison.filter(
        pl.col("model_name") == "naive_lagged_target_baseline"
    )["mean_absolute_error"].item()
    assert linear_error == pytest.approx(0.0)
    assert ridge_error >= 0.0
    assert naive_error == pytest.approx(1.0)


def test_walk_forward_ridge_baseline_same_structure_as_ols() -> None:
    """Ridge should emit the same public prediction schema as OLS."""

    frame = pl.DataFrame(
        {
            "trade_date": [
                date(2024, 1, 1),
                date(2024, 1, 2),
                date(2024, 1, 3),
                date(2024, 1, 4),
                date(2024, 1, 5),
            ],
            "feature_value": [1.0, 2.0, 3.0, 4.0, 5.0],
            "target_value": [1.0, 2.0, 3.0, 4.0, 5.0],
        }
    )

    predictions = walk_forward_ridge_baseline(
        frame=frame,
        feature_names=["feature_value"],
        target_name="target_value",
        min_train_size=2,
        alpha_candidates=[0.1, 1.0, 10.0],
    )

    assert predictions.columns == ["trade_date", "actual", "prediction"]
    assert predictions.height == 3


def test_walk_forward_ridge_empty_output() -> None:
    """Ridge should preserve the explicit empty prediction schema."""

    frame = pl.DataFrame(
        {
            "trade_date": [date(2024, 1, 1), date(2024, 1, 2)],
            "feature_value": [1.0, 2.0],
            "target_value": [1.0, 2.0],
        }
    )

    predictions = walk_forward_ridge_baseline(
        frame=frame,
        feature_names=["feature_value"],
        target_name="target_value",
        min_train_size=2,
        alpha_candidates=[1.0],
    )

    assert predictions.height == 0
    assert predictions.schema == {
        "trade_date": pl.Date,
        "actual": pl.Float64,
        "prediction": pl.Float64,
    }


def test_expanding_window_diagnostics_has_per_step_errors() -> None:
    """Diagnostics should emit one row per model and prediction date."""

    frame = pl.DataFrame(
        {
            "trade_date": [
                date(2024, 1, 1),
                date(2024, 1, 2),
                date(2024, 1, 3),
                date(2024, 1, 4),
                date(2024, 1, 5),
            ],
            "feature_value": [1.0, 2.0, 3.0, 4.0, 5.0],
            "target_value": [1.0, 2.0, 3.0, 4.0, 5.0],
        }
    )

    diagnostics = build_expanding_window_diagnostics(
        frame=frame,
        feature_name="feature_value",
        target_name="target_value",
        min_train_size=2,
        alpha_candidates=[1.0],
    )

    assert "trade_date" in diagnostics.columns
    assert "model_name" in diagnostics.columns
    assert "absolute_error" in diagnostics.columns
    model_names = set(diagnostics["model_name"].to_list())
    assert "feature_linear_baseline" in model_names
    assert "feature_ridge_baseline" in model_names


def test_prediction_intervals_bracket_actuals_roughly() -> None:
    """Residual-bootstrap intervals should add lower and upper bounds."""

    frame = pl.DataFrame(
        {
            "trade_date": [date(2024, 1, index + 1) for index in range(20)],
            "feature_value": [float(index) for index in range(20)],
            "target_value": [float(index) for index in range(20)],
        }
    )

    predictions = walk_forward_linear_baseline(
        frame=frame,
        feature_names=["feature_value"],
        target_name="target_value",
        min_train_size=5,
    )
    with_intervals = build_prediction_intervals(
        prediction_frame=predictions,
        confidence_level=0.90,
        bootstrap_iterations=200,
        random_seed=42,
    )

    assert "pi_lower" in with_intervals.columns
    assert "pi_upper" in with_intervals.columns
    for row in with_intervals.to_dicts():
        assert row["pi_lower"] <= row["pi_upper"]
