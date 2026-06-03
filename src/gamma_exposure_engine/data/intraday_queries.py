"""Build and execute canonical SPY intraday bar ClickHouse extract queries.

This module is intentionally narrow: it only defines SQL and extraction helpers
for the optional raw-cache refresh workflow. Offline analysis never imports or
calls this module.
"""

from __future__ import annotations

import logging
import textwrap

import polars as pl
from clickhouse_connect.driver.client import Client

from gamma_exposure_engine.data.clickhouse_client import execute_symbol_date_range_query

INTRADAY_DATASET_NAME: str = "intraday_bars"
logger = logging.getLogger(__name__)


def build_intraday_query() -> str:
    """Return parameterized SQL for the canonical intraday ETF bar query.

    Returns
    -------
    str
        ClickHouse SQL text that keeps placeholder parameters for symbol and
        date boundaries.
    """

    return textwrap.dedent(
        """
        SELECT
            symbol,
            ts,
            open,
            high,
            low,
            close,
            volume
        FROM firstrate.etfs
        WHERE symbol = {symbol:String}
          AND toDate(ts) BETWEEN {start_date:Date} AND {end_date:Date}
        ORDER BY ts
        """
    ).strip()


def fetch_intraday_bars_from_clickhouse(
    client: Client,
    symbol: str,
    start_date: str,
    end_date: str,
) -> pl.DataFrame:
    """Fetch intraday bars directly from ClickHouse.

    Parameters
    ----------
    client:
        Connected ClickHouse client used to execute the query.
    symbol:
        Underlying symbol, for example ``SPY``.
    start_date:
        Inclusive ISO-8601 date string (YYYY-MM-DD).
    end_date:
        Inclusive ISO-8601 date string (YYYY-MM-DD).

    Returns
    -------
    pl.DataFrame
        Intraday bars sorted by timestamp.
    """

    logger.info(
        "Querying ClickHouse for %s rows for %s from %s to %s",
        INTRADAY_DATASET_NAME,
        symbol,
        start_date,
        end_date,
    )
    return execute_symbol_date_range_query(
        client=client,
        query=build_intraday_query(),
        symbol=symbol,
        start_date=start_date,
        end_date=end_date,
    )
