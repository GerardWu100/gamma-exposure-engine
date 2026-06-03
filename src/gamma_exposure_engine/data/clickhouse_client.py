"""Create configured ClickHouse clients and convert query results to Polars.

The data layer stays intentionally thin: settings are loaded from the existing
project bootstrap, a client is created for the local ClickHouse instance, and
query results are normalized into ``polars.DataFrame`` objects.
"""

from __future__ import annotations

from typing import Any

import clickhouse_connect
import polars as pl
from clickhouse_connect.driver.client import Client

from gamma_exposure_engine.settings import load_settings


def create_clickhouse_client() -> Client:
    """Return a ClickHouse client configured from local project settings."""

    settings = load_settings()
    return clickhouse_connect.get_client(
        host=settings.clickhouse.host,
        port=settings.clickhouse.port,
        username=settings.clickhouse.user,
        password=settings.clickhouse.password,
        secure=settings.clickhouse.secure,
        verify=settings.clickhouse.verify,
    )


def execute_symbol_date_range_query(
    client: Client,
    query: str,
    symbol: str,
    start_date: str,
    end_date: str,
) -> pl.DataFrame:
    """Run one parameterized symbol/date query and return rows as Polars.

    Parameters
    ----------
    client:
        Connected ClickHouse client.
    query:
        SQL text with ``{symbol:String}``, ``{start_date:Date}``, and
        ``{end_date:Date}`` placeholders.
    symbol:
        Underlying symbol passed to the query.
    start_date:
        Inclusive ISO-8601 start date.
    end_date:
        Inclusive ISO-8601 end date.

    Returns
    -------
    pl.DataFrame
        Query result with ClickHouse column names preserved.
    """

    result = client.query(
        query,
        parameters={
            "symbol": symbol,
            "start_date": start_date,
            "end_date": end_date,
        },
    )
    return query_result_to_frame(result)


def query_result_to_frame(result: Any) -> pl.DataFrame:
    """Convert a ClickHouse query result into a Polars DataFrame.

    Args:
        result:
            The `clickhouse-connect` query result object returned by
            `Client.query`.

    Returns:
        pl.DataFrame: A frame with one row per query result row.
    """

    return pl.DataFrame(
        result.result_rows,
        schema=result.column_names,
        orient="row",
    )
