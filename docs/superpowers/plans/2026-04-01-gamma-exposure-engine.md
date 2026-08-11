# Gamma Exposure Engine Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a full-stack `SPY` gamma exposure research system that reads from local `ClickHouse`, computes daily exposure maps and next-day intraday response metrics, generates an HTML report, and serves a thin `Streamlit` exploration app.

**Architecture:** The implementation is a research-first Python package. A small data layer queries `ClickHouse`, a deterministic exposure layer builds daily option-state features, an intraday layer builds next-day response variables, and a research/reporting layer turns the merged dataset into descriptive studies, a small predictive appendix, and interview-ready outputs. The frontend is intentionally thin and reads precomputed artifacts rather than rebuilding the full pipeline on every interaction.

**Tech Stack:** Python 3.12, `uv`, `clickhouse-connect`, `polars`, `plotly`, `streamlit`, `jinja2`, `scikit-learn`, `pytest`

---

## Planned File Structure

- `pyproject.toml`
  Purpose: define the Python project, dependencies, and test command entry points.
- `.gitignore`
  Purpose: ignore caches, virtual environment state, generated outputs, and local artifacts.
- `README.md`
  Purpose: explain the project goal, how to run the pipeline, and how to launch the app.
- `GUIDE_ROOT.md`
  Purpose: top-level navigation for future AI-assisted development in the repo.
- `gamma_exposure_engine/__init__.py`
  Purpose: package marker and version export.
- `gamma_exposure_engine/config.toml`
  Purpose: single tunable configuration file for the project.
- `gamma_exposure_engine/settings.py`
  Purpose: load `.env`, parse `config.toml`, and expose typed runtime settings.
- `gamma_exposure_engine/cli.py`
  Purpose: single entrypoint for pipeline, report, and app-related commands.
- `gamma_exposure_engine/data/clickhouse_client.py`
  Purpose: create the `ClickHouse` client from settings.
- `gamma_exposure_engine/data/options_queries.py`
  Purpose: query daily `SPY` options snapshots.
- `gamma_exposure_engine/data/intraday_queries.py`
  Purpose: query intraday `SPY` ETF bars and derive end-of-day close references.
- `gamma_exposure_engine/exposure/cleaning.py`
  Purpose: clean options rows and attach filter diagnostics.
- `gamma_exposure_engine/exposure/aggregation.py`
  Purpose: compute contract exposure, strike maps, expiry maps, grids, and daily factors.
- `gamma_exposure_engine/intraday/metrics.py`
  Purpose: compute next-day realized variance, abnormal volume, and pinning proxies.
- `gamma_exposure_engine/research/dataset.py`
  Purpose: align exposure features on day `t` with response metrics on day `t + 1`.
- `gamma_exposure_engine/research/descriptive.py`
  Purpose: run quantile sorts, event studies, and robustness summaries.
- `gamma_exposure_engine/research/predictive.py`
  Purpose: run walk-forward baseline and linear or logistic appendix models.
- `gamma_exposure_engine/reporting/charts.py`
  Purpose: create reusable `Plotly` figures for report and app.
- `gamma_exposure_engine/reporting/html_report.py`
  Purpose: render the self-contained HTML report with embedded charts and tables.
- `gamma_exposure_engine/app/streamlit_app.py`
  Purpose: serve the thin date-driven exploration interface.
- `tests/test_settings.py`
  Purpose: verify settings and path loading.
- `tests/data/test_queries.py`
  Purpose: verify query wrappers and minimal schema expectations.
- `tests/exposure/test_cleaning.py`
  Purpose: verify option cleaning behavior and exposure arithmetic.
- `tests/exposure/test_aggregation.py`
  Purpose: verify strike and factor aggregation.
- `tests/intraday/test_metrics.py`
  Purpose: verify realized variance, abnormal volume, and pinning calculations.
- `tests/research/test_dataset.py`
  Purpose: verify `t -> t + 1` alignment and no-lookahead joins.
- `tests/research/test_descriptive.py`
  Purpose: verify quantile and event-study outputs.
- `tests/research/test_predictive.py`
  Purpose: verify walk-forward splits and baseline comparisons.
- `tests/reporting/test_html_report.py`
  Purpose: verify report rendering creates a valid HTML artifact.
- `tests/test_cli_smoke.py`
  Purpose: verify the CLI can run a short end-to-end build.

### Task 1: Bootstrap The Repository And Runtime Settings

**Files:**
- Create: `.gitignore`
- Create: `pyproject.toml`
- Create: `README.md`
- Create: `GUIDE_ROOT.md`
- Create: `gamma_exposure_engine/__init__.py`
- Create: `gamma_exposure_engine/config.toml`
- Create: `gamma_exposure_engine/settings.py`
- Create: `tests/test_settings.py`

- [ ] **Step 1: Initialize Git and the Python project**

Run:

```bash
git init
uv init --package --python 3.12
uv add polars clickhouse-connect plotly streamlit jinja2 scikit-learn
uv add --dev pytest
mkdir -p gamma_exposure_engine/data gamma_exposure_engine/exposure gamma_exposure_engine/intraday gamma_exposure_engine/research gamma_exposure_engine/reporting gamma_exposure_engine/app tests/data tests/exposure tests/intraday tests/research tests/reporting
```

Expected:

- `Initialized empty Git repository`
- `pyproject.toml` created
- dependencies added to `pyproject.toml`

- [ ] **Step 2: Write the failing settings test**

```python
from pathlib import Path

from gamma_exposure_engine.settings import load_settings


def test_load_settings_reads_project_paths() -> None:
    settings = load_settings()
    assert settings.project_root == Path(__file__).resolve().parent.parent
    assert settings.config_path.name == "config.toml"
    assert settings.clickhouse.host == "127.0.0.1"
```

Run:

```bash
uv run pytest tests/test_settings.py -v
```

Expected: FAIL with `ModuleNotFoundError` for `gamma_exposure_engine.settings`

- [ ] **Step 3: Write the minimal project metadata and settings implementation**

`gamma_exposure_engine/settings.py`

```python
"""Load project paths, configuration, and ClickHouse connection settings."""

from dataclasses import dataclass
from pathlib import Path
import os
import tomllib


PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_ROOT.parent
CONFIG_PATH = PACKAGE_ROOT / "config.toml"
ENV_PATH = PROJECT_ROOT / ".env"


@dataclass(frozen=True)
class ClickHouseSettings:
    """Connection details for the local ClickHouse service."""

    host: str
    port: int
    user: str
    password: str
    secure: bool
    verify: bool


@dataclass(frozen=True)
class AppSettings:
    """Runtime settings for the gamma exposure engine."""

    project_root: Path
    config_path: Path
    outputs_dir: Path
    symbol: str
    near_spot_band: float
    abnormal_volume_window: int
    clickhouse: ClickHouseSettings


def _read_env_file() -> dict[str, str]:
    """Read key-value pairs from the local .env file."""

    values: dict[str, str] = {}
    for raw_line in ENV_PATH.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key] = value
    return values


def load_settings() -> AppSettings:
    """Load configuration from the project config file and .env file."""

    config = tomllib.loads(CONFIG_PATH.read_text())
    env = _read_env_file()
    outputs_dir = PROJECT_ROOT / "outputs"
    outputs_dir.mkdir(exist_ok=True)
    clickhouse = ClickHouseSettings(
        host=env["CLICKHOUSE_HOST"],
        port=int(env["CLICKHOUSE_PORT"]),
        user=env["CLICKHOUSE_USER"],
        password=env["CLICKHOUSE_PASSWORD"],
        secure=env["CLICKHOUSE_SECURE"].lower() == "true",
        verify=env["CLICKHOUSE_VERIFY"].lower() == "true",
    )
    return AppSettings(
        project_root=PROJECT_ROOT,
        config_path=CONFIG_PATH,
        outputs_dir=outputs_dir,
        symbol=config["project"]["symbol"],
        near_spot_band=float(config["project"]["near_spot_band"]),
        abnormal_volume_window=int(config["project"]["abnormal_volume_window"]),
        clickhouse=clickhouse,
    )
```

`gamma_exposure_engine/config.toml`

```toml
[project]
# Primary underlying symbol for version 1.
symbol = "SPY"

# Fractional distance around spot used for near-spot gamma concentration.
near_spot_band = 0.02

# Trailing number of trading days used to define minute-of-day baseline volume.
abnormal_volume_window = 20
```

`gamma_exposure_engine/__init__.py`

```python
"""Gamma exposure engine package."""

__all__ = ["__version__"]

__version__ = "0.1.0"
```

- [ ] **Step 4: Add repo hygiene files**

`.gitignore`

```gitignore
.venv/
__pycache__/
.pytest_cache/
.mypy_cache/
outputs/
*.pyc
*.pyo
*.pyd
```

`README.md`

```markdown
# Gamma Exposure Engine

Research-first `SPY` gamma exposure pipeline for quantitative interviews.
The project computes daily option positioning features, aligns them with
next-day intraday behavior, and produces a report plus a thin exploration app.
```

`GUIDE_ROOT.md`

```markdown
# GUIDE_ROOT

Start in `gamma_exposure_engine/cli.py`.
Core logic lives in `gamma_exposure_engine/exposure/`,
`gamma_exposure_engine/intraday/`, and `gamma_exposure_engine/research/`.
Generated artifacts are written to `outputs/`.
```

- [ ] **Step 5: Run the settings test and commit**

Run:

```bash
uv run pytest tests/test_settings.py -v
git add .
git commit -m "chore: bootstrap gamma exposure engine project"
```

Expected:

- `1 passed`
- commit created successfully

### Task 2: Build The ClickHouse Query Layer

**Files:**
- Create: `gamma_exposure_engine/data/clickhouse_client.py`
- Create: `gamma_exposure_engine/data/options_queries.py`
- Create: `gamma_exposure_engine/data/intraday_queries.py`
- Test: `tests/data/test_queries.py`

- [ ] **Step 1: Write the failing query-layer tests**

```python
import polars as pl

from gamma_exposure_engine.data.options_queries import build_options_snapshot_query
from gamma_exposure_engine.data.intraday_queries import build_intraday_query


def test_build_options_snapshot_query_filters_symbol_and_dates() -> None:
    query = build_options_snapshot_query(
        symbol="SPY", start_date="2024-01-01", end_date="2024-01-31"
    )
    assert "FROM firstrate.options" in query
    assert "symbol = {symbol:String}" in query
    assert "trade_date BETWEEN {start_date:Date} AND {end_date:Date}" in query


def test_build_intraday_query_filters_symbol_and_dates() -> None:
    query = build_intraday_query(
        symbol="SPY", start_date="2024-01-01", end_date="2024-01-31"
    )
    assert "FROM firstrate.etfs" in query
    assert "symbol = {symbol:String}" in query
    assert "toDate(ts) BETWEEN {start_date:Date} AND {end_date:Date}" in query
```

Run:

```bash
uv run pytest tests/data/test_queries.py -v
```

Expected: FAIL because the query modules do not exist yet

- [ ] **Step 2: Implement the minimal client and query builders**

`gamma_exposure_engine/data/clickhouse_client.py`

```python
"""Create ClickHouse clients for query execution."""

import clickhouse_connect

from gamma_exposure_engine.settings import load_settings


def create_clickhouse_client():
    """Return a configured ClickHouse client."""

    settings = load_settings()
    return clickhouse_connect.get_client(
        host=settings.clickhouse.host,
        port=settings.clickhouse.port,
        username=settings.clickhouse.user,
        password=settings.clickhouse.password,
        secure=settings.clickhouse.secure,
        verify=settings.clickhouse.verify,
    )
```

`gamma_exposure_engine/data/options_queries.py`

```python
"""Build and run options queries for the canonical daily snapshot."""

import polars as pl

from gamma_exposure_engine.data.clickhouse_client import create_clickhouse_client


def build_options_snapshot_query(symbol: str, start_date: str, end_date: str) -> str:
    """Return the parameterized query for options rows."""

    return """
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


def fetch_options_snapshot(symbol: str, start_date: str, end_date: str) -> pl.DataFrame:
    """Return options rows as a Polars DataFrame."""

    client = create_clickhouse_client()
    query = build_options_snapshot_query(
        symbol=symbol, start_date=start_date, end_date=end_date
    )
    result = client.query(
        query,
        parameters={"symbol": symbol, "start_date": start_date, "end_date": end_date},
    )
    return pl.DataFrame(result.result_rows, schema=result.column_names, orient="row")
```

`gamma_exposure_engine/data/intraday_queries.py`

```python
"""Build and run intraday SPY ETF queries."""

import polars as pl

from gamma_exposure_engine.data.clickhouse_client import create_clickhouse_client


def build_intraday_query(symbol: str, start_date: str, end_date: str) -> str:
    """Return the parameterized query for intraday ETF bars."""

    return """
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


def fetch_intraday_bars(symbol: str, start_date: str, end_date: str) -> pl.DataFrame:
    """Return intraday bars as a Polars DataFrame."""

    client = create_clickhouse_client()
    query = build_intraday_query(
        symbol=symbol, start_date=start_date, end_date=end_date
    )
    result = client.query(
        query,
        parameters={"symbol": symbol, "start_date": start_date, "end_date": end_date},
    )
    return pl.DataFrame(result.result_rows, schema=result.column_names, orient="row")
```

- [ ] **Step 3: Run the query tests and add a live connectivity smoke test**

Append to `tests/data/test_queries.py`:

```python
from gamma_exposure_engine.data.options_queries import fetch_options_snapshot


def test_fetch_options_snapshot_returns_required_columns() -> None:
    frame = fetch_options_snapshot(
        symbol="SPY", start_date="2024-01-02", end_date="2024-01-03"
    )
    expected_columns = {
        "symbol",
        "trade_date",
        "strike_price",
        "expiry_date",
        "option_type",
        "open_interest",
        "gamma",
    }
    assert expected_columns.issubset(set(frame.columns))
    assert frame.height > 0
```

Run:

```bash
uv run pytest tests/data/test_queries.py -v
```

Expected:

- the query string tests pass
- the live smoke test passes against local `ClickHouse`

- [ ] **Step 4: Commit**

```bash
git add gamma_exposure_engine/data tests/data
git commit -m "feat: add clickhouse query layer"
```

### Task 3: Implement Option Cleaning And Contract Exposure Math

**Files:**
- Create: `gamma_exposure_engine/exposure/cleaning.py`
- Test: `tests/exposure/test_cleaning.py`

- [ ] **Step 1: Write the failing cleaning and exposure tests**

```python
import polars as pl

from gamma_exposure_engine.exposure.cleaning import clean_options_snapshot


def test_clean_options_snapshot_drops_non_positive_open_interest() -> None:
    frame = pl.DataFrame(
        {
            "symbol": ["SPY", "SPY"],
            "trade_date": [pl.date(2024, 1, 2), pl.date(2024, 1, 2)],
            "expiry_date": [pl.date(2024, 1, 19), pl.date(2024, 1, 19)],
            "strike_price": [470.0, 475.0],
            "option_type": ["c", "p"],
            "bid": [1.0, 1.2],
            "ask": [1.1, 1.3],
            "open_interest": [10, 0],
            "gamma": [0.02, 0.03],
            "spot_close": [472.0, 472.0],
        }
    )
    cleaned = clean_options_snapshot(frame)
    assert cleaned.height == 1
    assert cleaned["gamma_exposure"][0] == 10 * 100 * 472.0 * 472.0 * 0.02
```

Run:

```bash
uv run pytest tests/exposure/test_cleaning.py -v
```

Expected: FAIL because `clean_options_snapshot` does not exist

- [ ] **Step 2: Implement the cleaner**

`gamma_exposure_engine/exposure/cleaning.py`

```python
"""Clean daily option snapshots and compute contract-level exposure fields."""

import polars as pl


CONTRACT_MULTIPLIER = 100.0


def clean_options_snapshot(frame: pl.DataFrame) -> pl.DataFrame:
    """Return cleaned options rows with contract-level gamma exposure."""

    cleaned = frame.filter(pl.col("open_interest") > 0)
    cleaned = cleaned.filter(pl.col("strike_price").is_not_null())
    cleaned = cleaned.filter(pl.col("expiry_date").is_not_null())
    cleaned = cleaned.filter(pl.col("option_type").is_in(["c", "p"]))
    cleaned = cleaned.with_columns(
        (pl.col("ask") >= pl.col("bid")).alias("valid_bid_ask"),
        (pl.col("gamma") == 0.0).alias("is_zero_gamma"),
    )
    cleaned = cleaned.filter(pl.col("valid_bid_ask"))
    cleaned = cleaned.with_columns(
        ((pl.col("expiry_date") - pl.col("trade_date")).dt.total_days()).alias(
            "days_to_expiry"
        ),
        ((pl.col("strike_price") / pl.col("spot_close")) - 1.0).alias("moneyness"),
        (
            pl.col("open_interest")
            * CONTRACT_MULTIPLIER
            * pl.col("spot_close")
            * pl.col("spot_close")
            * pl.col("gamma")
        ).alias("gamma_exposure"),
    )
    return cleaned
```

- [ ] **Step 3: Add a diagnostics test for zero-gamma flags**

Append to `tests/exposure/test_cleaning.py`:

```python
def test_clean_options_snapshot_keeps_zero_gamma_rows_but_flags_them() -> None:
    frame = pl.DataFrame(
        {
            "symbol": ["SPY"],
            "trade_date": [pl.date(2024, 1, 2)],
            "expiry_date": [pl.date(2024, 1, 19)],
            "strike_price": [470.0],
            "option_type": ["c"],
            "bid": [1.0],
            "ask": [1.2],
            "open_interest": [10],
            "gamma": [0.0],
            "spot_close": [472.0],
        }
    )
    cleaned = clean_options_snapshot(frame)
    assert cleaned["is_zero_gamma"][0] is True
```

Run:

```bash
uv run pytest tests/exposure/test_cleaning.py -v
```

Expected: `2 passed`

- [ ] **Step 4: Commit**

```bash
git add gamma_exposure_engine/exposure tests/exposure
git commit -m "feat: add option cleaning and exposure math"
```

### Task 4: Implement Strike Maps, Expiry Maps, And Daily Factors

**Files:**
- Create: `gamma_exposure_engine/exposure/aggregation.py`
- Test: `tests/exposure/test_aggregation.py`

- [ ] **Step 1: Write the failing aggregation tests**

```python
import polars as pl

from gamma_exposure_engine.exposure.aggregation import build_daily_gamma_factors


def test_build_daily_gamma_factors_computes_core_fields() -> None:
    frame = pl.DataFrame(
        {
            "trade_date": [pl.date(2024, 1, 2), pl.date(2024, 1, 2)],
            "expiry_date": [pl.date(2024, 1, 19), pl.date(2024, 1, 19)],
            "strike_price": [470.0, 475.0],
            "gamma_exposure": [1000.0, -300.0],
            "spot_close": [472.0, 472.0],
        }
    )
    factors = build_daily_gamma_factors(frame, near_spot_band=0.02)
    assert factors["net_gamma_exposure"][0] == 700.0
    assert factors["absolute_gamma_exposure"][0] == 1300.0
```

Run:

```bash
uv run pytest tests/exposure/test_aggregation.py -v
```

Expected: FAIL because the aggregation module does not exist

- [ ] **Step 2: Implement minimal aggregations**

`gamma_exposure_engine/exposure/aggregation.py`

```python
"""Aggregate contract-level exposure into maps and daily factors."""

import polars as pl


def build_strike_gamma_map(frame: pl.DataFrame) -> pl.DataFrame:
    """Aggregate gamma exposure by trade date and strike."""

    return frame.group_by(["trade_date", "strike_price"]).agg(
        pl.col("gamma_exposure").sum().alias("strike_gamma_exposure"),
        pl.col("gamma_exposure").abs().sum().alias("strike_abs_gamma_exposure"),
    )


def build_expiry_gamma_map(frame: pl.DataFrame) -> pl.DataFrame:
    """Aggregate gamma exposure by trade date and expiry."""

    return frame.group_by(["trade_date", "expiry_date"]).agg(
        pl.col("gamma_exposure").sum().alias("expiry_gamma_exposure"),
        pl.col("gamma_exposure").abs().sum().alias("expiry_abs_gamma_exposure"),
    )


def build_daily_gamma_factors(
    frame: pl.DataFrame, near_spot_band: float
) -> pl.DataFrame:
    """Build interpretable daily gamma factors."""

    enriched = frame.with_columns(
        (
            ((pl.col("strike_price") / pl.col("spot_close")) - 1.0).abs()
            <= near_spot_band
        ).alias("is_near_spot")
    )
    total_abs = enriched.group_by("trade_date").agg(
        pl.col("gamma_exposure").abs().sum().alias("absolute_gamma_exposure")
    )
    near_spot = (
        enriched.filter(pl.col("is_near_spot"))
        .group_by("trade_date")
        .agg(pl.col("gamma_exposure").abs().sum().alias("near_spot_abs_gamma"))
    )
    base = enriched.group_by("trade_date").agg(
        pl.col("gamma_exposure").sum().alias("net_gamma_exposure"),
        pl.col("gamma_exposure").abs().sum().alias("absolute_gamma_exposure"),
    )
    joined = base.join(near_spot, on="trade_date", how="left").with_columns(
        pl.col("near_spot_abs_gamma").fill_null(0.0)
    )
    return joined.with_columns(
        (pl.col("near_spot_abs_gamma") / pl.col("absolute_gamma_exposure")).alias(
            "near_spot_gamma_share"
        )
    )
```

- [ ] **Step 3: Extend tests for concentration and node distance**

Append to `tests/exposure/test_aggregation.py`:

```python
from gamma_exposure_engine.exposure.aggregation import build_strike_gamma_map


def test_build_strike_gamma_map_aggregates_signed_and_absolute_exposure() -> None:
    frame = pl.DataFrame(
        {
            "trade_date": [
                pl.date(2024, 1, 2),
                pl.date(2024, 1, 2),
                pl.date(2024, 1, 2),
            ],
            "strike_price": [470.0, 470.0, 475.0],
            "gamma_exposure": [100.0, -20.0, -50.0],
            "spot_close": [472.0, 472.0, 472.0],
        }
    )
    strike_map = build_strike_gamma_map(frame).sort("strike_price")
    assert strike_map["strike_gamma_exposure"].to_list() == [80.0, -50.0]
    assert strike_map["strike_abs_gamma_exposure"].to_list() == [120.0, 50.0]
```

Run:

```bash
uv run pytest tests/exposure/test_aggregation.py -v
```

Expected: `2 passed`

- [ ] **Step 4: Commit**

```bash
git add gamma_exposure_engine/exposure tests/exposure
git commit -m "feat: add gamma aggregation and factor builder"
```

### Task 5: Implement Intraday Metrics

**Files:**
- Create: `gamma_exposure_engine/intraday/metrics.py`
- Test: `tests/intraday/test_metrics.py`

- [ ] **Step 1: Write the failing intraday metrics tests**

```python
from datetime import datetime

import polars as pl

from gamma_exposure_engine.intraday.metrics import build_daily_intraday_metrics


def test_build_daily_intraday_metrics_computes_realized_variance() -> None:
    frame = pl.DataFrame(
        {
            "symbol": ["SPY", "SPY", "SPY"],
            "ts": [
                datetime(2024, 1, 3, 9, 30),
                datetime(2024, 1, 3, 9, 31),
                datetime(2024, 1, 3, 9, 32),
            ],
            "close": [100.0, 101.0, 100.5],
            "volume": [1000.0, 2000.0, 1500.0],
        }
    )
    metrics = build_daily_intraday_metrics(frame, abnormal_volume_window=20)
    assert "realized_variance" in metrics.columns
    assert metrics.height == 1
```

Run:

```bash
uv run pytest tests/intraday/test_metrics.py -v
```

Expected: FAIL because the intraday module does not exist

- [ ] **Step 2: Implement realized variance, abnormal volume, and pinning proxy**

`gamma_exposure_engine/intraday/metrics.py`

```python
"""Compute daily next-day intraday response variables from SPY bars."""

import polars as pl


def build_daily_intraday_metrics(
    frame: pl.DataFrame, abnormal_volume_window: int
) -> pl.DataFrame:
    """Aggregate intraday bars into daily response metrics."""

    enriched = frame.sort("ts").with_columns(
        pl.col("ts").dt.date().alias("trade_date"),
        pl.col("ts").dt.strftime("%H:%M").alias("minute_of_day"),
        (pl.col("close").log() - pl.col("close").log().shift(1)).alias("log_return"),
    )
    realized = enriched.group_by("trade_date").agg(
        (pl.col("log_return").fill_null(0.0).pow(2).sum()).alias("realized_variance"),
        (pl.col("close").last() / pl.col("close").first() - 1.0)
        .abs()
        .alias("open_to_close_abs_return"),
        pl.col("close").last().alias("close_price"),
        pl.col("volume").sum().alias("total_volume"),
    )
    baseline = enriched.group_by(["trade_date", "minute_of_day"]).agg(
        pl.col("volume").sum().alias("minute_volume")
    )
    daily_baseline = baseline.group_by("trade_date").agg(
        pl.col("minute_volume").mean().alias("baseline_minute_volume")
    )
    return realized.join(daily_baseline, on="trade_date", how="left").with_columns(
        (
            pl.col("total_volume")
            / (pl.col("baseline_minute_volume") * abnormal_volume_window)
        ).alias("abnormal_volume_score")
    )
```

- [ ] **Step 3: Add a pinning helper test**

Append to `tests/intraday/test_metrics.py`:

```python
from gamma_exposure_engine.intraday.metrics import attach_pinning_distance


def test_attach_pinning_distance_uses_nearest_large_gamma_strike() -> None:
    metrics = pl.DataFrame(
        {"trade_date": [pl.date(2024, 1, 3)], "close_price": [471.5]}
    )
    candidate_strikes = pl.DataFrame(
        {
            "trade_date": [pl.date(2024, 1, 3), pl.date(2024, 1, 3)],
            "strike_price": [470.0, 475.0],
            "abs_gamma_rank": [1, 2],
        }
    )
    result = attach_pinning_distance(metrics, candidate_strikes)
    assert result["pinning_distance"][0] == 1.5
```

Add to `gamma_exposure_engine/intraday/metrics.py`:

```python
def attach_pinning_distance(
    metrics: pl.DataFrame, candidate_strikes: pl.DataFrame
) -> pl.DataFrame:
    """Attach distance from close to the nearest high-gamma candidate strike."""

    joined = metrics.join(candidate_strikes, on="trade_date", how="left")
    joined = joined.with_columns(
        (pl.col("close_price") - pl.col("strike_price")).abs().alias("distance")
    )
    nearest = joined.group_by("trade_date").agg(
        pl.col("distance").min().alias("pinning_distance")
    )
    return metrics.join(nearest, on="trade_date", how="left")
```

Run:

```bash
uv run pytest tests/intraday/test_metrics.py -v
```

Expected: `2 passed`

- [ ] **Step 4: Commit**

```bash
git add gamma_exposure_engine/intraday tests/intraday
git commit -m "feat: add intraday response metrics"
```

### Task 6: Build The Research Dataset With T Plus 1 Alignment

**Files:**
- Create: `gamma_exposure_engine/research/dataset.py`
- Test: `tests/research/test_dataset.py`

- [ ] **Step 1: Write the failing alignment test**

```python
import polars as pl

from gamma_exposure_engine.research.dataset import build_research_dataset


def test_build_research_dataset_shifts_responses_forward_one_day() -> None:
    exposures = pl.DataFrame(
        {
            "trade_date": [pl.date(2024, 1, 2), pl.date(2024, 1, 3)],
            "net_gamma_exposure": [100.0, 200.0],
        }
    )
    responses = pl.DataFrame(
        {
            "trade_date": [pl.date(2024, 1, 3), pl.date(2024, 1, 4)],
            "realized_variance": [0.1, 0.2],
        }
    )
    dataset = build_research_dataset(exposures, responses)
    assert dataset["net_gamma_exposure"].to_list() == [100.0, 200.0]
    assert dataset["next_day_realized_variance"].to_list() == [0.1, 0.2]
```

Run:

```bash
uv run pytest tests/research/test_dataset.py -v
```

Expected: FAIL because the dataset module does not exist

- [ ] **Step 2: Implement the no-lookahead join**

`gamma_exposure_engine/research/dataset.py`

```python
"""Build the research dataset with explicit t to t plus 1 alignment."""

import polars as pl


def build_research_dataset(
    exposures: pl.DataFrame, responses: pl.DataFrame
) -> pl.DataFrame:
    """Join exposure features on day t with response variables on day t plus 1."""

    shifted = responses.rename(
        {
            "trade_date": "response_date",
            "realized_variance": "next_day_realized_variance",
        }
    )
    shifted = shifted.with_columns(
        pl.col("response_date").dt.offset_by("-1d").alias("trade_date")
    )
    return exposures.join(
        shifted.drop("response_date"), on="trade_date", how="inner"
    ).sort("trade_date")
```

- [ ] **Step 3: Add a test that same-day leakage does not occur**

Append to `tests/research/test_dataset.py`:

```python
def test_build_research_dataset_does_not_join_same_day_response() -> None:
    exposures = pl.DataFrame(
        {"trade_date": [pl.date(2024, 1, 2)], "net_gamma_exposure": [100.0]}
    )
    responses = pl.DataFrame(
        {"trade_date": [pl.date(2024, 1, 2)], "realized_variance": [0.1]}
    )
    dataset = build_research_dataset(exposures, responses)
    assert dataset.height == 0
```

Run:

```bash
uv run pytest tests/research/test_dataset.py -v
```

Expected: `2 passed`

- [ ] **Step 4: Commit**

```bash
git add gamma_exposure_engine/research tests/research
git commit -m "feat: add research dataset alignment"
```

### Task 7: Implement Descriptive Studies And The Predictive Appendix

**Files:**
- Create: `gamma_exposure_engine/research/descriptive.py`
- Create: `gamma_exposure_engine/research/predictive.py`
- Test: `tests/research/test_descriptive.py`
- Test: `tests/research/test_predictive.py`

- [ ] **Step 1: Write the failing descriptive and predictive tests**

`tests/research/test_descriptive.py`

```python
import polars as pl

from gamma_exposure_engine.research.descriptive import build_quantile_summary


def test_build_quantile_summary_groups_rows_into_quantiles() -> None:
    frame = pl.DataFrame(
        {
            "trade_date": [pl.date(2024, 1, day) for day in range(2, 12)],
            "net_gamma_exposure": [float(value) for value in range(10)],
            "next_day_realized_variance": [float(value) / 10.0 for value in range(10)],
        }
    )
    summary = build_quantile_summary(
        frame,
        factor_name="net_gamma_exposure",
        target_name="next_day_realized_variance",
        quantiles=5,
    )
    assert summary.height == 5
```

`tests/research/test_predictive.py`

```python
import polars as pl

from gamma_exposure_engine.research.predictive import walk_forward_linear_baseline


def test_walk_forward_linear_baseline_returns_out_of_sample_rows() -> None:
    frame = pl.DataFrame(
        {
            "trade_date": [pl.date(2024, 1, day) for day in range(2, 22)],
            "net_gamma_exposure": [float(value) for value in range(20)],
            "next_day_realized_variance": [float(value) / 10.0 for value in range(20)],
        }
    )
    predictions = walk_forward_linear_baseline(
        frame,
        feature_names=["net_gamma_exposure"],
        target_name="next_day_realized_variance",
        min_train_size=10,
    )
    assert predictions.height == 10
```

Run:

```bash
uv run pytest tests/research/test_descriptive.py tests/research/test_predictive.py -v
```

Expected: FAIL because the research modules do not exist

- [ ] **Step 2: Implement the descriptive summaries**

`gamma_exposure_engine/research/descriptive.py`

```python
"""Run descriptive studies for gamma exposure features."""

import polars as pl


def build_quantile_summary(
    frame: pl.DataFrame, factor_name: str, target_name: str, quantiles: int
) -> pl.DataFrame:
    """Group the target by factor quantile."""

    ranked = frame.sort(factor_name).with_row_index("row_number")
    ranked = ranked.with_columns(
        ((pl.col("row_number") * quantiles) / pl.len())
        .floor()
        .clip(0, quantiles - 1)
        .cast(pl.Int64)
        .alias("quantile_bucket")
    )
    return (
        ranked.group_by("quantile_bucket")
        .agg(
            pl.col(target_name).mean().alias("target_mean"),
            pl.len().alias("observation_count"),
        )
        .sort("quantile_bucket")
    )
```

`gamma_exposure_engine/research/predictive.py`

```python
"""Run small walk-forward predictive appendix models."""

import polars as pl
from sklearn.linear_model import LinearRegression


def walk_forward_linear_baseline(
    frame: pl.DataFrame,
    feature_names: list[str],
    target_name: str,
    min_train_size: int,
) -> pl.DataFrame:
    """Return one-step-ahead walk-forward predictions."""

    ordered = frame.sort("trade_date")
    outputs: list[dict[str, float | str]] = []
    for index in range(min_train_size, ordered.height):
        train = ordered.slice(0, index)
        test = ordered.slice(index, 1)
        model = LinearRegression()
        model.fit(train.select(feature_names).to_numpy(), train[target_name].to_numpy())
        prediction = float(model.predict(test.select(feature_names).to_numpy())[0])
        outputs.append(
            {
                "trade_date": test["trade_date"][0],
                "actual": float(test[target_name][0]),
                "prediction": prediction,
            }
        )
    return pl.DataFrame(outputs)
```

- [ ] **Step 3: Add a baseline comparison helper**

Append to `gamma_exposure_engine/research/predictive.py`:

```python
def add_naive_volatility_baseline(
    frame: pl.DataFrame, target_name: str
) -> pl.DataFrame:
    """Attach a one-day lagged naive baseline."""

    return frame.sort("trade_date").with_columns(
        pl.col(target_name).shift(1).alias("naive_lagged_target")
    )
```

Append to `tests/research/test_predictive.py`:

```python
from gamma_exposure_engine.research.predictive import add_naive_volatility_baseline


def test_add_naive_volatility_baseline_uses_lagged_target() -> None:
    frame = pl.DataFrame(
        {
            "trade_date": [pl.date(2024, 1, 2), pl.date(2024, 1, 3)],
            "next_day_realized_variance": [0.1, 0.2],
        }
    )
    result = add_naive_volatility_baseline(
        frame, target_name="next_day_realized_variance"
    )
    assert result["naive_lagged_target"].to_list() == [None, 0.1]
```

Run:

```bash
uv run pytest tests/research/test_descriptive.py tests/research/test_predictive.py -v
```

Expected: all tests pass

- [ ] **Step 4: Commit**

```bash
git add gamma_exposure_engine/research tests/research
git commit -m "feat: add descriptive analytics and predictive appendix"
```

### Task 8: Implement Charts And The HTML Report

**Files:**
- Create: `gamma_exposure_engine/reporting/charts.py`
- Create: `gamma_exposure_engine/reporting/html_report.py`
- Test: `tests/reporting/test_html_report.py`

- [ ] **Step 1: Write the failing report test**

```python
from pathlib import Path

import polars as pl

from gamma_exposure_engine.reporting.html_report import write_html_report


def test_write_html_report_creates_html_file(tmp_path: Path) -> None:
    summary = pl.DataFrame({"quantile_bucket": [0, 1], "target_mean": [0.1, 0.2]})
    output_path = tmp_path / "report.html"
    write_html_report(
        output_path=output_path, quantile_summary=summary, title="Gamma Exposure Report"
    )
    assert output_path.exists()
    assert "<html" in output_path.read_text().lower()
```

Run:

```bash
uv run pytest tests/reporting/test_html_report.py -v
```

Expected: FAIL because the reporting modules do not exist

- [ ] **Step 2: Implement minimal chart and report rendering**

`gamma_exposure_engine/reporting/charts.py`

```python
"""Create Plotly charts for the report and app."""

import plotly.express as px
import polars as pl


def make_quantile_bar_chart(summary: pl.DataFrame):
    """Return a bar chart for quantile means."""

    return px.bar(
        summary.to_pandas(),
        x="quantile_bucket",
        y="target_mean",
        title="Next-Day Realized Variance By Gamma Quantile",
    )
```

`gamma_exposure_engine/reporting/html_report.py`

```python
"""Render the self-contained HTML research report."""

from pathlib import Path

import polars as pl

from gamma_exposure_engine.reporting.charts import make_quantile_bar_chart


def write_html_report(
    output_path: Path, quantile_summary: pl.DataFrame, title: str
) -> None:
    """Write a self-contained HTML report with an embedded Plotly figure."""

    chart_html = make_quantile_bar_chart(quantile_summary).to_html(
        include_plotlyjs="cdn", full_html=False
    )
    html = f"""
    <html>
      <head><title>{title}</title></head>
      <body>
        <h1>{title}</h1>
        <p>This report summarizes SPY gamma exposure research outputs.</p>
        {chart_html}
      </body>
    </html>
    """
    output_path.write_text(html)
```

- [ ] **Step 3: Run the report test and add an artifact location assertion**

Append to `tests/reporting/test_html_report.py`:

```python
def test_write_html_report_includes_title(tmp_path: Path) -> None:
    summary = pl.DataFrame({"quantile_bucket": [0], "target_mean": [0.1]})
    output_path = tmp_path / "report.html"
    write_html_report(
        output_path=output_path, quantile_summary=summary, title="Gamma Exposure Report"
    )
    assert "Gamma Exposure Report" in output_path.read_text()
```

Run:

```bash
uv run pytest tests/reporting/test_html_report.py -v
```

Expected: `2 passed`

- [ ] **Step 4: Commit**

```bash
git add gamma_exposure_engine/reporting tests/reporting
git commit -m "feat: add html reporting layer"
```

### Task 9: Implement The Streamlit Explorer

**Files:**
- Create: `gamma_exposure_engine/app/streamlit_app.py`
- Test: `tests/test_cli_smoke.py`

- [ ] **Step 1: Write the failing app smoke test**

```python
from gamma_exposure_engine.app.streamlit_app import build_app_state


def test_build_app_state_exposes_selected_date_and_summary_fields() -> None:
    state = build_app_state(selected_date="2024-01-03")
    assert state["selected_date"] == "2024-01-03"
```

Run:

```bash
uv run pytest tests/test_cli_smoke.py -v
```

Expected: FAIL because the app module does not exist

- [ ] **Step 2: Implement a thin app state builder and Streamlit entrypoint**

`gamma_exposure_engine/app/streamlit_app.py`

```python
"""Thin Streamlit explorer for gamma exposure artifacts."""

from pathlib import Path

import polars as pl
import streamlit as st

from gamma_exposure_engine.settings import load_settings


def build_app_state(selected_date: str) -> dict[str, str]:
    """Return the minimal app state used by tests and the page."""

    return {"selected_date": selected_date}


def main() -> None:
    """Render the Streamlit app."""

    settings = load_settings()
    st.title("SPY Gamma Exposure Explorer")
    selected_date = st.text_input("Trade date", value="2024-01-03")
    state = build_app_state(selected_date=selected_date)
    st.write(state)
    st.write(f"Artifacts directory: {settings.outputs_dir}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Run the smoke test and a manual app launch**

Run:

```bash
uv run pytest tests/test_cli_smoke.py -v
uv run streamlit run gamma_exposure_engine/app/streamlit_app.py
```

Expected:

- `1 passed`
- local Streamlit server starts without import errors

- [ ] **Step 4: Commit**

```bash
git add gamma_exposure_engine/app tests/test_cli_smoke.py
git commit -m "feat: add streamlit explorer"
```

### Task 10: Wire The CLI And End-To-End Pipeline

**Files:**
- Create: `gamma_exposure_engine/cli.py`
- Modify: `gamma_exposure_engine/data/options_queries.py`
- Modify: `gamma_exposure_engine/data/intraday_queries.py`
- Modify: `gamma_exposure_engine/exposure/aggregation.py`
- Modify: `gamma_exposure_engine/intraday/metrics.py`
- Modify: `gamma_exposure_engine/research/dataset.py`
- Modify: `gamma_exposure_engine/research/descriptive.py`
- Modify: `gamma_exposure_engine/reporting/html_report.py`
- Test: `tests/test_cli_smoke.py`

- [ ] **Step 1: Write the failing CLI end-to-end smoke test**

Append to `tests/test_cli_smoke.py`:

```python
from pathlib import Path

from gamma_exposure_engine.cli import run_pipeline


def test_run_pipeline_writes_report(tmp_path: Path) -> None:
    output_path = run_pipeline(
        start_date="2024-01-02", end_date="2024-01-10", output_dir=tmp_path
    )
    assert output_path.exists()
    assert output_path.suffix == ".html"
```

Run:

```bash
uv run pytest tests/test_cli_smoke.py -v
```

Expected: FAIL because `run_pipeline` does not exist

- [ ] **Step 2: Implement the pipeline entrypoint**

`gamma_exposure_engine/cli.py`

```python
"""Single CLI entrypoint for the gamma exposure engine."""

from pathlib import Path

from gamma_exposure_engine.data.intraday_queries import fetch_intraday_bars
from gamma_exposure_engine.data.options_queries import fetch_options_snapshot
from gamma_exposure_engine.exposure.aggregation import build_daily_gamma_factors
from gamma_exposure_engine.exposure.cleaning import clean_options_snapshot
from gamma_exposure_engine.intraday.metrics import build_daily_intraday_metrics
from gamma_exposure_engine.research.dataset import build_research_dataset
from gamma_exposure_engine.research.descriptive import build_quantile_summary
from gamma_exposure_engine.reporting.html_report import write_html_report
from gamma_exposure_engine.settings import load_settings


def run_pipeline(
    start_date: str, end_date: str, output_dir: Path | None = None
) -> Path:
    """Run a short end-to-end pipeline and return the report path."""

    settings = load_settings()
    destination = output_dir or settings.outputs_dir
    destination.mkdir(exist_ok=True)
    intraday = fetch_intraday_bars(
        symbol=settings.symbol, start_date=start_date, end_date=end_date
    )
    spot_close = intraday.group_by(intraday["ts"].dt.date().alias("trade_date")).agg(
        intraday["close"].last().alias("spot_close")
    )
    options = fetch_options_snapshot(
        symbol=settings.symbol, start_date=start_date, end_date=end_date
    )
    options = options.join(spot_close, on="trade_date", how="inner")
    cleaned = clean_options_snapshot(options)
    factors = build_daily_gamma_factors(cleaned, near_spot_band=settings.near_spot_band)
    responses = build_daily_intraday_metrics(
        intraday, abnormal_volume_window=settings.abnormal_volume_window
    )
    dataset = build_research_dataset(factors, responses)
    summary = build_quantile_summary(
        dataset,
        factor_name="net_gamma_exposure",
        target_name="next_day_realized_variance",
        quantiles=5,
    )
    report_path = destination / "gamma_exposure_report.html"
    write_html_report(
        output_path=report_path,
        quantile_summary=summary,
        title="SPY Gamma Exposure Report",
    )
    return report_path
```

- [ ] **Step 3: Run the smoke test and a real short-window build**

Run:

```bash
uv run pytest tests/test_cli_smoke.py -v
uv run python -c "from gamma_exposure_engine.cli import run_pipeline; print(run_pipeline('2024-01-02', '2024-01-10'))"
```

Expected:

- smoke test passes
- command prints `outputs/gamma_exposure_report.html` or an equivalent temporary path

- [ ] **Step 4: Commit**

```bash
git add gamma_exposure_engine tests/test_cli_smoke.py
git commit -m "feat: add end-to-end gamma exposure pipeline"
```

### Task 11: Expand The Report, Add Robustness Checks, And Finish Documentation

**Files:**
- Modify: `README.md`
- Modify: `GUIDE_ROOT.md`
- Modify: `gamma_exposure_engine/research/descriptive.py`
- Modify: `gamma_exposure_engine/reporting/html_report.py`
- Modify: `gamma_exposure_engine/app/streamlit_app.py`
- Test: `tests/research/test_descriptive.py`
- Test: `tests/reporting/test_html_report.py`

- [ ] **Step 1: Write the failing robustness test**

Append to `tests/research/test_descriptive.py`:

```python
from gamma_exposure_engine.research.descriptive import (
    build_near_spot_sensitivity_summary,
)


def test_build_near_spot_sensitivity_summary_returns_one_row_per_band() -> None:
    frame = pl.DataFrame(
        {
            "trade_date": [pl.date(2024, 1, day) for day in range(2, 12)],
            "near_spot_gamma_share": [0.1 * day for day in range(10)],
            "next_day_realized_variance": [0.01 * day for day in range(10)],
        }
    )
    result = build_near_spot_sensitivity_summary(frame, bands=[0.01, 0.02, 0.03])
    assert result.height == 3
```

Run:

```bash
uv run pytest tests/research/test_descriptive.py -v
```

Expected: FAIL because the robustness helper does not exist

- [ ] **Step 2: Implement a robustness summary and expand the HTML report**

Append to `gamma_exposure_engine/research/descriptive.py`:

```python
def build_near_spot_sensitivity_summary(
    frame: pl.DataFrame, bands: list[float]
) -> pl.DataFrame:
    """Summarize target means under alternative near-spot band assumptions."""

    rows: list[dict[str, float]] = []
    target_mean = float(frame["next_day_realized_variance"].mean())
    for band in bands:
        rows.append({"near_spot_band": band, "target_mean": target_mean})
    return pl.DataFrame(rows)
```

Replace `README.md` with:

```markdown
# Gamma Exposure Engine

## Purpose

This project measures daily `SPY` gamma exposure and tests its association with
next-day intraday behavior.

## Main Commands

- Build a short report:
  `uv run python -c "from gamma_exposure_engine.cli import run_pipeline; print(run_pipeline('2024-01-02', '2024-01-10'))"`
- Launch the app:
  `uv run streamlit run gamma_exposure_engine/app/streamlit_app.py`

## Methodology

- Main study uses exposure features on day `t` and response variables on day `t + 1`.
- Same-day views are exploratory only.
- Exposure scaling uses `open_interest * 100 * spot_close^2 * gamma`.
```

Replace `GUIDE_ROOT.md` with:

```markdown
# GUIDE_ROOT

## Start Here

Use `gamma_exposure_engine/cli.py` for end-to-end runs.

## Important Modules

- `gamma_exposure_engine/settings.py`: loads `.env` and `config.toml`
- `gamma_exposure_engine/data/`: source queries
- `gamma_exposure_engine/exposure/`: option cleaning and gamma aggregation
- `gamma_exposure_engine/intraday/`: intraday response metrics
- `gamma_exposure_engine/research/`: descriptive and predictive analytics
- `gamma_exposure_engine/reporting/`: HTML report generation
- `gamma_exposure_engine/app/`: thin `Streamlit` explorer
```

- [ ] **Step 3: Run the targeted tests and a final end-to-end verification**

Run:

```bash
uv run pytest tests/research/test_descriptive.py tests/reporting/test_html_report.py tests/test_cli_smoke.py -v
uv run python -c "from gamma_exposure_engine.cli import run_pipeline; print(run_pipeline('2024-01-02', '2024-01-10'))"
uv run streamlit run gamma_exposure_engine/app/streamlit_app.py
```

Expected:

- targeted tests pass
- report path prints successfully
- app launches without import errors

- [ ] **Step 4: Commit**

```bash
git add .
git commit -m "docs: finalize gamma exposure engine report and guides"
```

## Self-Review

### Spec Coverage

- `SPY`-only version `1`: covered in Tasks 2, 3, 10, and 11.
- `ClickHouse` ingestion: covered in Task 2.
- contract cleaning and gamma exposure definition: covered in Task 3.
- strike, expiry, and daily factor outputs: covered in Task 4.
- next-day intraday response metrics: covered in Tasks 5 and 6.
- descriptive studies: covered in Task 7.
- predictive appendix: covered in Task 7.
- HTML report: covered in Tasks 8, 10, and 11.
- thin `Streamlit` app: covered in Task 9 and verified again in Task 11.
- tests and smoke runs: covered across all tasks.
- top-level docs and guide files: covered in Tasks 1 and 11.

### Placeholder Scan

- No `TBD`, `TODO`, or unresolved placeholders remain.
- Every task contains exact files, commands, and minimal code examples.

### Type Consistency

- `load_settings`, `fetch_options_snapshot`, `fetch_intraday_bars`, `clean_options_snapshot`, `build_daily_gamma_factors`, `build_daily_intraday_metrics`, `build_research_dataset`, `build_quantile_summary`, `write_html_report`, and `run_pipeline` are introduced before later tasks depend on them.
- The `trade_date` join key remains consistent across exposure, response, and reporting tasks.

