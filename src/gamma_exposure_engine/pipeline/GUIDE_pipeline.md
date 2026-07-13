# GUIDE_pipeline

`src/gamma_exposure_engine/pipeline/` orchestrates the full offline analysis run.

## Module

- `offline_pipeline.py`
  - loads local raw Parquet inputs
  - builds spot close and joins options with spot
  - runs cleaning and unsigned gamma-mass factor construction
  - computes intraday response metrics and pinning proxy
  - builds aligned research dataset
  - runs descriptive, inferential, regime, robustness, and predictive analyses
  - writes non-HTML artifacts to the selected output directory

## Output Contract

Pipeline output is intentionally simple and interview-friendly:

- one aligned Parquet dataset
- multiple CSV tables for analysis summaries
- one JSON run manifest
