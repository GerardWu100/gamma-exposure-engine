# GUIDE_exposure

`src/gamma_exposure_engine/exposure/` transforms option snapshot rows into daily gamma factors.

## Modules

- `cleaning.py`
  - validates structural row assumptions
  - keeps diagnostics for dropped and surviving rows
  - computes contract-level gamma exposure
- `aggregation.py`
  - aggregates cleaned rows to strike, expiry, and daily factor levels
  - exposes explainable daily gamma factors used by research modules

## Inputs and Outputs

- input: options snapshot rows with `spot_close` already attached
- output: cleaned options and one-row-per-day factor tables
