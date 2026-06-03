"""Build and execute canonical SPY options snapshot ClickHouse queries.

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

OPTIONS_DATASET_NAME: str = "options_snapshot"
logger = logging.getLogger(__name__)


def build_options_snapshot_query() -> str:
    """Return parameterized SQL for the canonical options snapshot query.

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
            trade_date,
            strike_price,
            expiry_date,
            option_type,
            last_price,
            bid,
            ask,
            bid_iv,
            ask_iv,
            open_interest,
            volume,
            delta,
            gamma,
            vega,
            theta,
            rho
        FROM firstrate.options
        WHERE symbol = {symbol:String}
          AND trade_date BETWEEN {start_date:Date} AND {end_date:Date}
        ORDER BY trade_date, expiry_date, strike_price, option_type
        """
    ).strip()


def fetch_options_snapshot_from_clickhouse(
    client: Client,
    symbol: str,
    start_date: str,
    end_date: str,
) -> pl.DataFrame:
    """Fetch options snapshots directly from ClickHouse.

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
        Options snapshots sorted by date, expiry, strike, and option type.
    """

    logger.info(
        "Querying ClickHouse for %s rows for %s from %s to %s",
        OPTIONS_DATASET_NAME,
        symbol,
        start_date,
        end_date,
    )
    return execute_symbol_date_range_query(
        client=client,
        query=build_options_snapshot_query(),
        symbol=symbol,
        start_date=start_date,
        end_date=end_date,
    )
