"""Tests for ClickHouse extraction query helpers.

These tests verify that query modules remain pure ClickHouse extractors for the
optional raw refresh workflow.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import date, datetime

import polars as pl
import pytest
from gamma_exposure_engine.data.intraday_queries import (
    build_intraday_query,
    fetch_intraday_bars_from_clickhouse,
)
from gamma_exposure_engine.data.options_queries import (
    build_options_snapshot_query,
    fetch_options_snapshot_from_clickhouse,
)


def test_build_options_snapshot_query_targets_canonical_table() -> None:
    """The options query builder should keep ClickHouse placeholders in SQL."""

    query = build_options_snapshot_query()

    assert "FROM firstrate.options" in query
    assert "symbol = {symbol:String}" in query
    assert "trade_date BETWEEN {start_date:Date} AND {end_date:Date}" in query


def test_build_intraday_query_targets_canonical_table() -> None:
    """The intraday query builder should keep ClickHouse placeholders in SQL."""

    query = build_intraday_query()

    assert "FROM firstrate.etfs" in query
    assert "symbol = {symbol:String}" in query
    assert "toDate(ts) BETWEEN {start_date:Date} AND {end_date:Date}" in query


def test_fetch_options_snapshot_forwards_parameters_and_converts_rows() -> None:
    """The options extractor should pass query parameters through unchanged."""

    fake_client = FakeClient(
        result=FakeQueryResult(
            column_names=["symbol", "trade_date", "strike_price", "expiry_date"],
            result_rows=[
                ("SPY", date(2024, 1, 2), 470.0, date(2024, 1, 19)),
            ],
        )
    )

    frame = fetch_options_snapshot_from_clickhouse(
        client=fake_client,
        symbol="SPY",
        start_date="2024-01-01",
        end_date="2024-01-31",
    )

    assert fake_client.queries == [
        (
            build_options_snapshot_query(),
            {
                "symbol": "SPY",
                "start_date": "2024-01-01",
                "end_date": "2024-01-31",
            },
        )
    ]
    assert isinstance(frame, pl.DataFrame)
    assert frame.to_dicts() == [
        {
            "symbol": "SPY",
            "trade_date": date(2024, 1, 2),
            "strike_price": 470.0,
            "expiry_date": date(2024, 1, 19),
        }
    ]


def test_fetch_intraday_bars_forwards_parameters_and_converts_rows() -> None:
    """The intraday extractor should pass query parameters through unchanged."""

    fake_client = FakeClient(
        result=FakeQueryResult(
            column_names=["symbol", "ts", "open", "high", "low", "close", "volume"],
            result_rows=[
                (
                    "SPY",
                    datetime(2024, 1, 2, 9, 30),
                    470.0,
                    471.0,
                    469.5,
                    470.5,
                    123456,
                ),
            ],
        )
    )

    frame = fetch_intraday_bars_from_clickhouse(
        client=fake_client,
        symbol="SPY",
        start_date="2024-01-01",
        end_date="2024-01-31",
    )

    assert fake_client.queries == [
        (
            build_intraday_query(),
            {
                "symbol": "SPY",
                "start_date": "2024-01-01",
                "end_date": "2024-01-31",
            },
        )
    ]
    assert isinstance(frame, pl.DataFrame)
    assert frame.to_dicts() == [
        {
            "symbol": "SPY",
            "ts": datetime(2024, 1, 2, 9, 30),
            "open": 470.0,
            "high": 471.0,
            "low": 469.5,
            "close": 470.5,
            "volume": 123456,
        }
    ]


def test_live_clickhouse_smoke() -> None:
    """Run a real ClickHouse smoke check only when explicitly requested."""

    if os.environ.get("RUN_CLICKHOUSE_SMOKE_TESTS") != "1":
        pytest.skip(
            "Set RUN_CLICKHOUSE_SMOKE_TESTS=1 to run the ClickHouse smoke test."
        )

    from gamma_exposure_engine.data.clickhouse_client import create_clickhouse_client

    client = create_clickhouse_client()
    frame = fetch_options_snapshot_from_clickhouse(
        client=client,
        symbol="SPY",
        start_date=_latest_options_trade_date(),
        end_date=_latest_options_trade_date(),
    )

    assert isinstance(frame, pl.DataFrame)
    assert frame.height > 0


@dataclass(frozen=True)
class FakeQueryResult:
    """Minimal stand-in for the ClickHouse query result object."""

    column_names: list[str]
    result_rows: list[tuple[object, ...]]


@dataclass
class FakeClient:
    """Capture ClickHouse queries and return a fixed fake result."""

    result: FakeQueryResult
    queries: list[tuple[str, dict[str, object]]] = None

    def __post_init__(self) -> None:
        self.queries = []

    def query(self, query: str, parameters: dict[str, object]) -> FakeQueryResult:
        self.queries.append((query, parameters))
        return self.result


def _latest_options_trade_date() -> str:
    """Return the most recent SPY options trade date available in ClickHouse."""

    from gamma_exposure_engine.data.clickhouse_client import create_clickhouse_client

    client = create_clickhouse_client()
    result = client.query(
        """
        SELECT max(trade_date)
        FROM firstrate.options
        WHERE symbol = {symbol:String}
        """,
        parameters={"symbol": "SPY"},
    )
    return result.result_rows[0][0].isoformat()
