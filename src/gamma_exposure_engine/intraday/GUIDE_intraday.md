# GUIDE_intraday

`src/gamma_exposure_engine/intraday/` computes daily response metrics from minute bars.

## Module

- `metrics.py`
  - reduces minute bars to one row per trade date
  - computes realized variance and realized volatility
  - computes open-to-close and close-to-close absolute returns
  - computes abnormal volume score from past-only rolling baselines
  - attaches pinning-distance proxy from prior-day candidate strikes

## Connection to Pipeline

These response metrics are joined with exposure factors in
`research/dataset.py` to build day-`t` to day-`t+1` aligned observations.
