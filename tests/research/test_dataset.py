"""Tests for the research dataset time-alignment contract.

The research layer should join day ``t`` exposure features to day ``t + 1``
response variables only. Same-day response rows are not allowed to leak into
the joined research frame because that would introduce lookahead bias.
"""

from __future__ import annotations

from datetime import date

import polars as pl

from gamma_exposure_engine.research.dataset import build_research_dataset


def test_build_research_dataset_joins_next_observed_exposure_day() -> None:
    """Responses should align to the next exposure date and preserve audit columns."""

    exposures = pl.DataFrame(
        {
            "trade_date": [date(2024, 1, 5), date(2024, 1, 8)],
            "net_gamma_exposure": [100.0, 200.0],
            "absolute_gamma_exposure": [150.0, 250.0],
        }
    )
    responses = pl.DataFrame(
        {
            "trade_date": [date(2024, 1, 8), date(2024, 1, 9)],
            "realized_variance": [0.2, 0.3],
            "realized_volatility": [0.4, 0.5],
            "abnormal_volume_score": [1.7, 1.9],
            "close_price": [501.0, 502.0],
        }
    )

    dataset = build_research_dataset(exposures, responses)

    assert dataset.columns == [
        "trade_date",
        "net_gamma_exposure",
        "absolute_gamma_exposure",
        "response_trade_date",
        "next_day_realized_variance",
        "next_day_realized_volatility",
        "next_day_abnormal_volume_score",
        "next_day_close_price",
    ]
    assert dataset["trade_date"].to_list() == [date(2024, 1, 5)]
    assert dataset["response_trade_date"].to_list() == [date(2024, 1, 8)]
    assert dataset["net_gamma_exposure"].to_list() == [100.0]
    assert dataset["absolute_gamma_exposure"].to_list() == [150.0]
    assert dataset["next_day_realized_variance"].to_list() == [0.2]
    assert dataset["next_day_realized_volatility"].to_list() == [0.4]
    assert dataset["next_day_abnormal_volume_score"].to_list() == [1.7]
    assert dataset["next_day_close_price"].to_list() == [501.0]


def test_build_research_dataset_does_not_join_same_day_response() -> None:
    """Same-day responses must not appear in the no-lookahead research frame."""

    exposures = pl.DataFrame(
        {
            "trade_date": [date(2024, 1, 2)],
            "net_gamma_exposure": [100.0],
        }
    )
    responses = pl.DataFrame(
        {
            "trade_date": [date(2024, 1, 2)],
            "realized_variance": [0.1],
        }
    )

    dataset = build_research_dataset(exposures, responses)

    assert dataset.columns == [
        "trade_date",
        "net_gamma_exposure",
        "response_trade_date",
        "next_day_realized_variance",
    ]
    assert dataset.height == 0


def test_build_research_dataset_drops_over_jump_on_sparse_response_dates() -> None:
    """Sparse response dates must not skip over a missing exposure day."""

    exposures = pl.DataFrame(
        {
            "trade_date": [
                date(2024, 1, 2),
                date(2024, 1, 3),
                date(2024, 1, 4),
            ],
            "net_gamma_exposure": [10.0, 20.0, 30.0],
        }
    )
    responses = pl.DataFrame(
        {
            "trade_date": [date(2024, 1, 4)],
            "realized_variance": [0.4],
        }
    )

    dataset = build_research_dataset(exposures, responses)

    assert dataset.columns == [
        "trade_date",
        "net_gamma_exposure",
        "response_trade_date",
        "next_day_realized_variance",
    ]
    assert dataset["trade_date"].to_list() == [date(2024, 1, 3)]
    assert dataset["response_trade_date"].to_list() == [date(2024, 1, 4)]
    assert dataset["next_day_realized_variance"].to_list() == [0.4]
