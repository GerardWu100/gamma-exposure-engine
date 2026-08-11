"""Predictive research helpers for aligned exposure-response frames.

These helpers provide compact, interpretable baselines on the aligned
research dataset. They are intentionally walk-forward and trade-date sorted so
they remain safe for time-series evaluation.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from math import sqrt

import numpy as np
import polars as pl
from sklearn.linear_model import LinearRegression, Ridge, RidgeCV

TRADE_DATE_COLUMN: str = "trade_date"
ACTUAL_COLUMN: str = "actual"
PREDICTION_COLUMN: str = "prediction"
NAIVE_LAGGED_TARGET_COLUMN: str = "naive_lagged_target"
MODEL_NAME_COLUMN: str = "model_name"
OBSERVATION_COUNT_COLUMN: str = "observation_count"
MEAN_ABSOLUTE_ERROR_COLUMN: str = "mean_absolute_error"
ROOT_MEAN_SQUARED_ERROR_COLUMN: str = "root_mean_squared_error"
ABSOLUTE_ERROR_COLUMN: str = "absolute_error"
PI_LOWER_COLUMN: str = "pi_lower"
PI_UPPER_COLUMN: str = "pi_upper"
PREDICTION_SCHEMA: dict[str, pl.DataType] = {
    TRADE_DATE_COLUMN: pl.Date,
    ACTUAL_COLUMN: pl.Float64,
    PREDICTION_COLUMN: pl.Float64,
}
PREDICTIVE_COMPARISON_SCHEMA: dict[str, pl.DataType] = {
    MODEL_NAME_COLUMN: pl.String,
    OBSERVATION_COUNT_COLUMN: pl.Int64,
    MEAN_ABSOLUTE_ERROR_COLUMN: pl.Float64,
    ROOT_MEAN_SQUARED_ERROR_COLUMN: pl.Float64,
}

__all__ = [
    "add_naive_lagged_target_baseline",
    "add_naive_volatility_baseline",
    "build_expanding_window_diagnostics",
    "build_prediction_intervals",
    "build_predictive_baseline_comparison",
    "walk_forward_linear_baseline",
    "walk_forward_ridge_baseline",
]


ModelTrainer = Callable[[np.ndarray, np.ndarray], LinearRegression | Ridge | RidgeCV]


def walk_forward_linear_baseline(
    frame: pl.DataFrame,
    feature_names: Sequence[str],
    target_name: str,
    min_train_size: int,
) -> pl.DataFrame:
    """Emit walk-forward out-of-sample predictions from a linear model.

    Args:
        frame:
            Research frame with one row per trade date. The frame must include
            ``trade_date``, the feature columns listed in ``feature_names``,
            and the target column named by ``target_name``.
        feature_names:
            Feature columns used by the linear regression baseline.
        target_name:
            Target column to predict.
        min_train_size:
            Minimum number of sorted rows required before the first
            out-of-sample prediction is emitted.

    Returns:
        pl.DataFrame: One row per out-of-sample prediction with ``trade_date``,
        the realized target as ``actual``, and the predicted target as
        ``prediction``.
    """

    # Delegate the walk-forward mechanics to one helper so linear and ridge
    # baselines share the exact same out-of-sample indexing contract.
    return _walk_forward_predictions(
        frame=frame,
        feature_names=feature_names,
        target_name=target_name,
        min_train_size=min_train_size,
        model_trainer=_train_linear_regression,
    )


def add_naive_lagged_target_baseline(
    frame: pl.DataFrame,
    target_name: str,
) -> pl.DataFrame:
    """Attach a one-day lagged target baseline on trade-date-sorted data.

    Parameters
    ----------
    frame:
        Research dataset with one row per trade date.
    target_name:
        Target column whose previous-day value is used as the naive forecast.

    Returns
    -------
    pl.DataFrame
        Sorted input data with one additional ``naive_lagged_target`` column.
    """

    ordered_frame = frame.sort(TRADE_DATE_COLUMN)
    # The first row has no prior observation, so the lagged baseline is null.
    return ordered_frame.with_columns(
        pl.col(target_name).shift(1).alias(NAIVE_LAGGED_TARGET_COLUMN)
    )


def add_naive_volatility_baseline(
    frame: pl.DataFrame,
    target_name: str,
) -> pl.DataFrame:
    """Backward-compatible alias for ``add_naive_lagged_target_baseline``.

    The old function name remains available so existing notebooks and scripts
    keep working, but the implementation now uses the clearer name.
    """

    return add_naive_lagged_target_baseline(
        frame=frame,
        target_name=target_name,
    )


def build_predictive_baseline_comparison(
    frame: pl.DataFrame,
    feature_name: str,
    target_name: str,
    min_train_size: int,
    ridge_alpha_candidates: Sequence[float] = (1.0,),
) -> pl.DataFrame:
    """Compare the configured feature against a naive lagged-target baseline.

    Args:
        frame:
            Research dataset with one row per trade date.
        feature_name:
            Single configured feature column used in the walk-forward linear
            baseline.
        target_name:
            Continuous target column evaluated in the appendix.
        min_train_size:
            Minimum sorted training history required before the first
            walk-forward prediction.
        ridge_alpha_candidates:
            Candidate Ridge penalty strengths used by the regularized baseline.

    Returns:
        pl.DataFrame: One row per baseline with the shared out-of-sample
        observation count plus mean absolute error and root mean squared error.
        Both baselines are evaluated on the exact same prediction dates.
    """

    # All baselines are evaluated on this same cleaned sample so model errors
    # are directly comparable and cannot differ because of missing-value drift.
    clean_frame = _clean_predictive_frame(
        frame=frame,
        feature_name=feature_name,
        target_name=target_name,
    )
    linear_predictions = walk_forward_linear_baseline(
        frame=clean_frame,
        feature_names=[feature_name],
        target_name=target_name,
        min_train_size=min_train_size,
    )
    if linear_predictions.height == 0:
        return _empty_predictive_comparison_frame()

    linear_summary = _summarize_prediction_frame(
        prediction_frame=linear_predictions,
        model_name="feature_linear_baseline",
    )
    ridge_predictions = walk_forward_ridge_baseline(
        frame=clean_frame,
        feature_names=[feature_name],
        target_name=target_name,
        min_train_size=min_train_size,
        alpha_candidates=ridge_alpha_candidates,
    )
    ridge_summary = _summarize_prediction_frame(
        prediction_frame=ridge_predictions,
        model_name="feature_ridge_baseline",
    )
    # Join on linear prediction dates so every baseline is scored on exactly
    # the same out-of-sample window.
    naive_predictions = (
        add_naive_lagged_target_baseline(
            frame=clean_frame,
            target_name=target_name,
        )
        .join(
            linear_predictions.select(TRADE_DATE_COLUMN, ACTUAL_COLUMN),
            on=TRADE_DATE_COLUMN,
            how="inner",
        )
        .drop_nulls([NAIVE_LAGGED_TARGET_COLUMN])
    )
    naive_summary = _summarize_prediction_frame(
        prediction_frame=naive_predictions.rename(
            {NAIVE_LAGGED_TARGET_COLUMN: PREDICTION_COLUMN}
        ).select(TRADE_DATE_COLUMN, ACTUAL_COLUMN, PREDICTION_COLUMN),
        model_name="naive_lagged_target_baseline",
    )
    return pl.concat([linear_summary, ridge_summary, naive_summary], how="vertical")


def _empty_prediction_frame() -> pl.DataFrame:
    """Build the empty walk-forward result with the public output schema."""

    return pl.DataFrame(
        {
            TRADE_DATE_COLUMN: pl.Series([], dtype=pl.Date),
            ACTUAL_COLUMN: pl.Series([], dtype=pl.Float64),
            PREDICTION_COLUMN: pl.Series([], dtype=pl.Float64),
        }
    )


def _prediction_errors_frame(
    prediction_frame: pl.DataFrame,
    model_name: str,
) -> pl.DataFrame:
    """Return per-date absolute errors for one walk-forward baseline."""

    return prediction_frame.with_columns(
        (pl.col(ACTUAL_COLUMN) - pl.col(PREDICTION_COLUMN))
        .abs()
        .alias(ABSOLUTE_ERROR_COLUMN),
        pl.lit(model_name).alias(MODEL_NAME_COLUMN),
    ).select(
        TRADE_DATE_COLUMN,
        MODEL_NAME_COLUMN,
        ABSOLUTE_ERROR_COLUMN,
    )


def _summarize_prediction_frame(
    prediction_frame: pl.DataFrame,
    model_name: str,
) -> pl.DataFrame:
    """Reduce row-level predictions to compact error metrics."""

    error_frame = prediction_frame.with_columns(
        (pl.col(ACTUAL_COLUMN) - pl.col(PREDICTION_COLUMN)).abs().alias("_abs_error"),
        (pl.col(ACTUAL_COLUMN) - pl.col(PREDICTION_COLUMN))
        .pow(2)
        .alias("_squared_error"),
    )
    mean_absolute_error = error_frame.select(pl.col("_abs_error").mean()).item()
    mean_squared_error = error_frame.select(pl.col("_squared_error").mean()).item()
    return pl.DataFrame(
        [
            {
                MODEL_NAME_COLUMN: model_name,
                OBSERVATION_COUNT_COLUMN: prediction_frame.height,
                MEAN_ABSOLUTE_ERROR_COLUMN: float(mean_absolute_error),
                ROOT_MEAN_SQUARED_ERROR_COLUMN: float(sqrt(mean_squared_error)),
            }
        ],
        schema=PREDICTIVE_COMPARISON_SCHEMA,
    )


def _empty_predictive_comparison_frame() -> pl.DataFrame:
    """Build the empty predictive-comparison result with the public schema."""

    return pl.DataFrame(
        {
            MODEL_NAME_COLUMN: pl.Series([], dtype=pl.String),
            OBSERVATION_COUNT_COLUMN: pl.Series([], dtype=pl.Int64),
            MEAN_ABSOLUTE_ERROR_COLUMN: pl.Series([], dtype=pl.Float64),
            ROOT_MEAN_SQUARED_ERROR_COLUMN: pl.Series([], dtype=pl.Float64),
        }
    )


def walk_forward_ridge_baseline(
    frame: pl.DataFrame,
    feature_names: Sequence[str],
    target_name: str,
    min_train_size: int,
    alpha_candidates: Sequence[float],
) -> pl.DataFrame:
    """Emit walk-forward predictions from a Ridge regression baseline.

    Parameters
    ----------
    frame:
        Research dataset with one row per trade date.
    feature_names:
        Feature columns used by the Ridge model.
    target_name:
        Target column to predict.
    min_train_size:
        Minimum sorted history required before the first prediction.
    alpha_candidates:
        Ridge penalty strengths passed to ``RidgeCV``.

    Returns
    -------
    pl.DataFrame
        Out-of-sample prediction rows with the same schema as the linear
        baseline.
    """

    # Reuse the shared walk-forward loop so ridge and linear predictions are
    # generated under identical indexing and data-slicing rules.
    return _walk_forward_predictions(
        frame=frame,
        feature_names=feature_names,
        target_name=target_name,
        min_train_size=min_train_size,
        model_trainer=lambda train_features, train_targets: _train_ridge_model(
            train_features=train_features,
            train_targets=train_targets,
            alpha_candidates=alpha_candidates,
        ),
    )


def build_expanding_window_diagnostics(
    frame: pl.DataFrame,
    feature_name: str,
    target_name: str,
    min_train_size: int,
    alpha_candidates: Sequence[float],
) -> pl.DataFrame:
    """Build per-step absolute prediction errors for the model baselines.

    Parameters
    ----------
    frame:
        Research dataset with one row per trade date.
    feature_name:
        Single feature column used by both baselines.
    target_name:
        Target column evaluated out-of-sample.
    min_train_size:
        Minimum training history required before scoring starts.
    alpha_candidates:
        Ridge penalty strengths passed to the regularized baseline.

    Returns
    -------
    pl.DataFrame
        One row per ``(trade_date, model_name)`` pair with absolute error.
    """

    # Reuse the same cleaning contract as the comparison table so diagnostics
    # and summary errors are based on the same valid input rows.
    clean_frame = _clean_predictive_frame(
        frame=frame,
        feature_name=feature_name,
        target_name=target_name,
    )
    linear_predictions = walk_forward_linear_baseline(
        frame=clean_frame,
        feature_names=[feature_name],
        target_name=target_name,
        min_train_size=min_train_size,
    )
    ridge_predictions = walk_forward_ridge_baseline(
        frame=clean_frame,
        feature_names=[feature_name],
        target_name=target_name,
        min_train_size=min_train_size,
        alpha_candidates=alpha_candidates,
    )

    diagnostics_frames = [
        _prediction_errors_frame(
            prediction_frame=prediction_frame,
            model_name=model_name,
        )
        for model_name, prediction_frame in [
            ("feature_linear_baseline", linear_predictions),
            ("feature_ridge_baseline", ridge_predictions),
        ]
        if prediction_frame.height > 0
    ]

    if not diagnostics_frames:
        return pl.DataFrame(
            {
                TRADE_DATE_COLUMN: pl.Series([], dtype=pl.Date),
                MODEL_NAME_COLUMN: pl.Series([], dtype=pl.String),
                ABSOLUTE_ERROR_COLUMN: pl.Series([], dtype=pl.Float64),
            }
        )

    return pl.concat(diagnostics_frames, how="vertical")


def build_prediction_intervals(
    prediction_frame: pl.DataFrame,
    confidence_level: float,
    bootstrap_iterations: int,
    random_seed: int = 42,
) -> pl.DataFrame:
    """Attach residual-bootstrap prediction intervals to point forecasts.

    Parameters
    ----------
    prediction_frame:
        Walk-forward predictions with ``actual`` and ``prediction`` columns.
    confidence_level:
        Desired interval coverage between ``0`` and ``1``.
    bootstrap_iterations:
        Number of residual resamples drawn per prediction.
    random_seed:
        Seed for reproducible residual resampling.

    Returns
    -------
    pl.DataFrame
        Input predictions plus ``pi_lower`` and ``pi_upper``.
    """

    if bootstrap_iterations < 1:
        msg = "bootstrap_iterations must be at least 1"
        raise ValueError(msg)

    if confidence_level <= 0.0 or confidence_level >= 1.0:
        msg = "confidence_level must be strictly between 0 and 1"
        raise ValueError(msg)

    if prediction_frame.height == 0:
        return prediction_frame.with_columns(
            pl.Series(PI_LOWER_COLUMN, [], dtype=pl.Float64),
            pl.Series(PI_UPPER_COLUMN, [], dtype=pl.Float64),
        )

    residuals = (
        prediction_frame.get_column(ACTUAL_COLUMN).to_numpy()
        - prediction_frame.get_column(PREDICTION_COLUMN).to_numpy()
    )
    point_predictions = prediction_frame.get_column(PREDICTION_COLUMN).to_numpy()
    random_number_generator = np.random.default_rng(random_seed)
    alpha = 1.0 - confidence_level
    lower_percentile = 100.0 * (alpha / 2.0)
    upper_percentile = 100.0 * (1.0 - (alpha / 2.0))

    pi_lower_values: list[float] = []
    pi_upper_values: list[float] = []
    for point_prediction in point_predictions:
        bootstrapped_predictions = point_prediction + random_number_generator.choice(
            residuals,
            size=bootstrap_iterations,
            replace=True,
        )
        pi_lower_values.append(
            float(np.percentile(bootstrapped_predictions, lower_percentile))
        )
        pi_upper_values.append(
            float(np.percentile(bootstrapped_predictions, upper_percentile))
        )

    return prediction_frame.with_columns(
        pl.Series(PI_LOWER_COLUMN, pi_lower_values, dtype=pl.Float64),
        pl.Series(PI_UPPER_COLUMN, pi_upper_values, dtype=pl.Float64),
    )


def _walk_forward_predictions(
    frame: pl.DataFrame,
    feature_names: Sequence[str],
    target_name: str,
    min_train_size: int,
    model_trainer: ModelTrainer,
) -> pl.DataFrame:
    """Run one generic walk-forward loop and return prediction rows.

    Parameters
    ----------
    frame:
        Research dataset with one row per trade date.
    feature_names:
        Feature columns used by the fitted model.
    target_name:
        Target column predicted out-of-sample.
    min_train_size:
        Minimum history required before the first forecast is emitted.
    model_trainer:
        Callable that fits and returns a scikit-learn-compatible model.

    Returns
    -------
    pl.DataFrame
        Walk-forward predictions with the shared public schema.
    """

    ordered_frame = frame.sort(TRADE_DATE_COLUMN)
    prediction_rows: list[dict[str, object]] = []

    for prediction_index in range(min_train_size, ordered_frame.height):
        # Train on rows strictly before the prediction row to keep the
        # evaluation out-of-sample and free from lookahead bias.
        train_frame = ordered_frame.slice(0, prediction_index)
        prediction_frame = ordered_frame.slice(prediction_index, 1)

        # Convert frame slices once per step so each model trainer receives
        # plain NumPy arrays with the same shape contract.
        train_features = train_frame.select(feature_names).to_numpy()
        train_targets = train_frame.select(target_name).to_numpy().ravel()
        model = model_trainer(train_features, train_targets)

        prediction_features = prediction_frame.select(feature_names).to_numpy()
        prediction_value = float(model.predict(prediction_features)[0])
        prediction_rows.append(
            {
                TRADE_DATE_COLUMN: prediction_frame.item(0, TRADE_DATE_COLUMN),
                ACTUAL_COLUMN: prediction_frame.item(0, target_name),
                PREDICTION_COLUMN: prediction_value,
            }
        )

    if not prediction_rows:
        return _empty_prediction_frame()

    return pl.DataFrame(prediction_rows, schema=PREDICTION_SCHEMA)


def _train_linear_regression(
    train_features: np.ndarray,
    train_targets: np.ndarray,
) -> LinearRegression:
    """Fit and return an ordinary least-squares linear model."""

    linear_model = LinearRegression()
    linear_model.fit(train_features, train_targets)
    return linear_model


def _train_ridge_model(
    train_features: np.ndarray,
    train_targets: np.ndarray,
    alpha_candidates: Sequence[float],
) -> Ridge | RidgeCV:
    """Fit and return a Ridge model, using CV only when feasible.

    The expanding walk-forward loop has very small training windows at the
    beginning. Cross-validation is only attempted when at least two folds are
    available; otherwise the model falls back to a fixed alpha.
    """

    # Limit folds so each split still has enough rows and avoid invalid CV
    # requests on tiny early windows.
    effective_cv = min(5, train_features.shape[0] // 2)
    if effective_cv >= 2:
        ridge_model: Ridge | RidgeCV = RidgeCV(
            alphas=list(alpha_candidates), cv=effective_cv
        )
    else:
        ridge_model = Ridge(alpha=float(alpha_candidates[0]))

    ridge_model.fit(train_features, train_targets)
    return ridge_model


def _clean_predictive_frame(
    frame: pl.DataFrame,
    feature_name: str,
    target_name: str,
) -> pl.DataFrame:
    """Keep only finite feature-target rows used by predictive baselines."""

    # Remove null and non-finite values once so every predictive helper shares
    # the same valid input contract.
    return frame.drop_nulls([feature_name, target_name]).filter(
        pl.col(feature_name).is_finite(),
        pl.col(target_name).is_finite(),
    )
