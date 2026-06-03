"""Cross-factor and factor-target rank correlation summaries.

The portfolio report needs a small matrix view of which factors align with
which targets, and whether the factors are mostly redundant with each other.
This module computes Spearman rank correlations because the research question
is monotonic association, not strictly linear fit.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Final

import polars as pl
from scipy import stats

FACTOR_LABEL_COLUMN: Final[str] = "factor"

__all__ = [
    "build_factor_factor_correlations",
    "build_factor_target_correlations",
]


def build_factor_target_correlations(
    frame: pl.DataFrame,
    factor_names: Sequence[str],
    target_names: Sequence[str],
) -> pl.DataFrame:
    """Build a factor-target Spearman correlation matrix.

    Parameters
    ----------
    frame:
        Research dataset with one row per trade date.
    factor_names:
        Factor columns used as matrix row labels.
    target_names:
        Target columns used as matrix value columns.

    Returns
    -------
    pl.DataFrame
        One row per factor, with one numeric column per target.
    """

    return _build_spearman_matrix(
        frame=frame,
        row_names=factor_names,
        column_names=target_names,
    )


def build_factor_factor_correlations(
    frame: pl.DataFrame,
    factor_names: Sequence[str],
) -> pl.DataFrame:
    """Build a symmetric factor-factor Spearman correlation matrix.

    Parameters
    ----------
    frame:
        Research dataset with one row per trade date.
    factor_names:
        Factor columns used as both row labels and matrix columns.

    Returns
    -------
    pl.DataFrame
        Square matrix with one row and one column per factor.
    """

    return _build_spearman_matrix(
        frame=frame,
        row_names=factor_names,
        column_names=factor_names,
        identity_diagonal=True,
    )


def _build_spearman_matrix(
    frame: pl.DataFrame,
    row_names: Sequence[str],
    column_names: Sequence[str],
    identity_diagonal: bool = False,
) -> pl.DataFrame:
    """Build one labeled Spearman correlation matrix from pairwise cleaning."""

    if frame.height == 0:
        return _empty_correlation_matrix(column_names=list(column_names))

    summary_rows: list[dict[str, object]] = []
    for row_name in row_names:
        row: dict[str, object] = {FACTOR_LABEL_COLUMN: row_name}
        for column_name in column_names:
            if identity_diagonal and row_name == column_name:
                row[column_name] = 1.0
                continue

            row[column_name] = _compute_pairwise_spearman(
                frame=frame,
                left_column=row_name,
                right_column=column_name,
            )
        summary_rows.append(row)

    return pl.DataFrame(summary_rows)


def _compute_pairwise_spearman(
    frame: pl.DataFrame,
    left_column: str,
    right_column: str,
) -> float:
    """Compute Spearman rho for one pair after pairwise null/finite cleaning."""

    clean_frame = frame.drop_nulls([left_column, right_column]).filter(
        pl.col(left_column).is_finite(),
        pl.col(right_column).is_finite(),
    )
    if clean_frame.height < 2:
        return float("nan")

    if clean_frame.get_column(left_column).n_unique() <= 1:
        return float("nan")

    if clean_frame.get_column(right_column).n_unique() <= 1:
        return float("nan")

    result = stats.spearmanr(
        clean_frame.get_column(left_column).to_numpy(),
        clean_frame.get_column(right_column).to_numpy(),
    )
    return float(result.statistic)


def _empty_correlation_matrix(column_names: list[str]) -> pl.DataFrame:
    """Return an empty correlation matrix with the public schema."""

    schema: dict[str, pl.DataType] = {FACTOR_LABEL_COLUMN: pl.String}
    for column_name in column_names:
        schema[column_name] = pl.Float64
    return pl.DataFrame(schema=schema)
