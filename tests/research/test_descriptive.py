"""Tests for the research descriptive summary helpers.

The descriptive layer should provide a deterministic quantile summary over a
factor and target pair. The contract stays intentionally small so it is easy to
reason about on aligned trade-date research frames.
"""

from __future__ import annotations

from datetime import date

import polars as pl

from gamma_exposure_engine.research.descriptive import (
    build_alternative_band_sensitivity,
    build_leave_one_month_out_sensitivity,
    build_near_spot_share_threshold_summary,
    build_quantile_summary,
    build_subperiod_stability,
)


def test_build_quantile_summary_groups_sorted_factor_into_integer_buckets() -> None:
    """Rows should be sorted by factor and assigned deterministic bucket IDs."""

    frame = pl.DataFrame(
        {
            "factor_value": [30.0, 10.0, 20.0, 50.0, 40.0],
            "target_value": [3.0, 1.0, 2.0, 5.0, 4.0],
        }
    )

    summary = build_quantile_summary(
        frame=frame,
        factor_name="factor_value",
        target_name="target_value",
        quantiles=3,
    )

    assert summary.columns == [
        "quantile_bucket",
        "target_mean",
        "observation_count",
    ]
    assert summary["quantile_bucket"].to_list() == [0, 1, 2]
    assert summary["observation_count"].to_list() == [2, 2, 1]
    assert summary["target_mean"].to_list() == [1.5, 3.5, 5.0]


def test_build_quantile_summary_uses_trade_date_to_break_ties() -> None:
    """Equal factor values should stay in a deterministic trade-date order."""

    frame = pl.DataFrame(
        {
            "trade_date": [
                date(2024, 1, 2),
                date(2024, 1, 1),
                date(2024, 1, 3),
            ],
            "factor_value": [1.0, 1.0, 2.0],
            "target_value": [20.0, 10.0, 30.0],
        }
    )

    summary = build_quantile_summary(
        frame=frame,
        factor_name="factor_value",
        target_name="target_value",
        quantiles=3,
    )

    assert summary["quantile_bucket"].to_list() == [0, 1, 2]
    assert summary["observation_count"].to_list() == [1, 1, 1]
    assert summary["target_mean"].to_list() == [10.0, 20.0, 30.0]


def test_build_quantile_summary_compacts_buckets_when_quantiles_exceed_rows() -> None:
    """Observed bucket labels should remain contiguous on small frames."""

    frame = pl.DataFrame(
        {
            "factor_value": [2.0, 1.0],
            "target_value": [20.0, 10.0],
        }
    )

    summary = build_quantile_summary(
        frame=frame,
        factor_name="factor_value",
        target_name="target_value",
        quantiles=5,
    )

    assert summary["quantile_bucket"].to_list() == [0, 1]
    assert summary["observation_count"].to_list() == [1, 1]
    assert summary["target_mean"].to_list() == [10.0, 20.0]


def test_build_quantile_summary_empty_schema_matches_non_empty_schema() -> None:
    """Empty output should keep the same column names and dtypes contract."""

    frame = pl.DataFrame(
        {
            "factor_value": pl.Series([], dtype=pl.Float64),
            "target_value": pl.Series([], dtype=pl.Float64),
        }
    )

    summary = build_quantile_summary(
        frame=frame,
        factor_name="factor_value",
        target_name="target_value",
        quantiles=3,
    )

    assert summary.columns == [
        "quantile_bucket",
        "target_mean",
        "observation_count",
    ]
    assert summary.schema == {
        "quantile_bucket": pl.Int64,
        "target_mean": pl.Float64,
        "observation_count": pl.Int64,
    }
    assert summary.height == 0


def test_build_near_spot_share_threshold_summary_splits_observed_share_levels() -> None:
    """Threshold splits should summarize the observed near-spot share factor."""

    frame = pl.DataFrame(
        {
            "trade_date": [
                date(2024, 1, 2),
                date(2024, 1, 3),
                date(2024, 1, 4),
                date(2024, 1, 5),
            ],
            "near_spot_gamma_share": [0.10, 0.25, 0.45, 0.70],
            "next_day_realized_variance": [1.0, 2.0, 4.0, 8.0],
        }
    )

    summary = build_near_spot_share_threshold_summary(
        frame=frame,
        target_name="next_day_realized_variance",
        thresholds=[0.20, 0.50],
    )

    assert summary.columns == [
        "near_spot_share_threshold",
        "at_or_above_threshold_count",
        "below_threshold_count",
        "at_or_above_threshold_target_mean",
        "below_threshold_target_mean",
        "target_mean_spread",
    ]
    assert summary["near_spot_share_threshold"].to_list() == [0.2, 0.5]
    assert summary["at_or_above_threshold_count"].to_list() == [3, 1]
    assert summary["below_threshold_count"].to_list() == [1, 3]
    assert summary["at_or_above_threshold_target_mean"].to_list() == [
        (2.0 + 4.0 + 8.0) / 3.0,
        8.0,
    ]
    assert summary["below_threshold_target_mean"].to_list() == [
        1.0,
        (1.0 + 2.0 + 4.0) / 3.0,
    ]
    assert summary["target_mean_spread"].to_list() == [
        ((2.0 + 4.0 + 8.0) / 3.0) - 1.0,
        8.0 - ((1.0 + 2.0 + 4.0) / 3.0),
    ]


def test_subperiod_stability_splits_at_midpoint() -> None:
    """The temporal midpoint split should divide the sample into two halves."""

    frame = pl.DataFrame(
        {
            "trade_date": [date(2024, 1, index + 1) for index in range(20)],
            "factor_value": [float(index) for index in range(20)],
            "target_value": [float(index * 2) for index in range(20)],
        }
    )

    result = build_subperiod_stability(
        frame=frame,
        factor_name="factor_value",
        target_name="target_value",
    )

    assert result.columns == [
        "subperiod",
        "spearman_rho",
        "p_value",
        "observation_count",
    ]
    assert result.height == 2
    assert result["subperiod"].to_list() == ["first_half", "second_half"]
    assert result["observation_count"].to_list() == [10, 10]


def test_alternative_band_sensitivity_returns_one_row_per_band() -> None:
    """Each alternative near-spot band should emit one sensitivity row."""

    cleaned_options = pl.DataFrame(
        {
            "trade_date": [date(2024, 1, 2)] * 6,
            "strike_price": [99.0, 100.0, 101.0, 95.0, 105.0, 110.0],
            "expiry_date": [date(2024, 1, 19)] * 6,
            "option_type": ["c"] * 6,
            "gamma_exposure": [10.0, 20.0, 15.0, 5.0, 8.0, 3.0],
            "moneyness": [-0.01, 0.0, 0.01, -0.05, 0.05, 0.10],
            "spot_close": [100.0] * 6,
        }
    )
    targets = pl.DataFrame(
        {
            "trade_date": [date(2024, 1, 2)],
            "target_value": [0.05],
        }
    )

    result = build_alternative_band_sensitivity(
        cleaned_options=cleaned_options,
        targets=targets,
        target_name="target_value",
        band_widths=[0.01, 0.03, 0.05],
    )

    assert result.columns == [
        "band_width",
        "spearman_rho",
        "p_value",
        "observation_count",
    ]
    assert result.height == 3
    assert result["band_width"].to_list() == [0.01, 0.03, 0.05]


def test_leave_one_month_out_returns_one_row_per_month() -> None:
    """Each distinct calendar month should produce one leave-out row."""

    dates = (
        [date(2024, 1, index + 1) for index in range(10)]
        + [date(2024, 2, index + 1) for index in range(10)]
        + [date(2024, 3, index + 1) for index in range(10)]
    )
    frame = pl.DataFrame(
        {
            "trade_date": dates,
            "factor_value": [float(index) for index in range(30)],
            "target_value": [float(index * 2) for index in range(30)],
        }
    )

    result = build_leave_one_month_out_sensitivity(
        frame=frame,
        factor_name="factor_value",
        target_name="target_value",
    )

    assert result.columns == [
        "dropped_month",
        "spearman_rho",
        "p_value",
        "observation_count",
    ]
    assert result.height == 3
