# Offline Data Contract

This project is offline-first. Normal analysis reads only local Parquet files
under `data/raw/`.

## Canonical Required Raw Inputs

- `data/raw/SPY_intraday_bars.parquet`
- `data/raw/SPY_options_snapshot.parquet`
- `data/raw/manifest.json`

## Required Columns and Meaning

### `SPY_intraday_bars.parquet`

- `symbol`: string ticker (expected `SPY` in shipped demo data)
- `ts`: bar timestamp in New York market timezone
- `open`: minute open price in dollars
- `high`: minute high price in dollars
- `low`: minute low price in dollars
- `close`: minute close price in dollars
- `volume`: minute traded volume

### `SPY_options_snapshot.parquet`

- `symbol`: string ticker (expected `SPY` in shipped demo data)
- `trade_date`: options snapshot trade date
- `strike_price`: option strike price in dollars
- `expiry_date`: listed option expiry date
- `option_type`: option side (`c` for call, `p` for put)
- `last_price`: last traded option premium
- `bid`: option best bid
- `ask`: option best ask
- `bid_iv`: implied volatility at bid
- `ask_iv`: implied volatility at ask
- `open_interest`: contract open interest count
- `volume`: contract traded volume count
- `delta`: option delta Greek
- `gamma`: option gamma Greek
- `vega`: option vega Greek
- `theta`: option theta Greek
- `rho`: option rho Greek

The options schema contains no account owner, customer/dealer classification,
trade direction, or position sign. `open_interest` is an unsigned count and
the supplied vanilla-option `gamma` is expected to be non-negative. The
pipeline can therefore compute open-interest-weighted gamma mass, but it
cannot identify signed dealer gamma. Negative gamma values are rejected as
invalid input rather than interpreted as short positions.

## Expected Date Coverage

- Current shipped demo coverage: `2024-01-02` through `2024-01-31`
- If raw data is refreshed later, `data/raw/manifest.json` is the source of
  truth for actual coverage.

## Offline Runtime Guarantee

- Offline analysis command (`gex run-offline-analysis`) uses only
  `src/data/raw_store.py` loaders.
- Teaching notebook (`notebooks/gamma_exposure_pipeline_demo.ipynb`) also uses
  only `raw_store.py` loaders.
- Offline execution never calls ClickHouse and never requires `.env`.
- Missing raw files fail fast with actionable local-file guidance.
