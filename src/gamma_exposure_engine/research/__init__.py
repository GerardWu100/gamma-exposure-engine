"""Research dataset construction and downstream study helpers."""

from gamma_exposure_engine.research.dataset import build_research_dataset
from gamma_exposure_engine.research.descriptive import build_quantile_summary
from gamma_exposure_engine.research.predictive import (
    add_naive_lagged_target_baseline,
    add_naive_volatility_baseline,
    walk_forward_linear_baseline,
)

__all__ = [
    "add_naive_lagged_target_baseline",
    "add_naive_volatility_baseline",
    "build_quantile_summary",
    "build_research_dataset",
    "walk_forward_linear_baseline",
]
