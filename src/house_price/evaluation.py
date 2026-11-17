import numpy as np

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


def calculate_regression_metrics(y_true, y_pred):
    """Calculate regression metrics after basic input checks."""

    actual = np.asarray(y_true, dtype=float)
    predicted = np.asarray(y_pred, dtype=float)

    if actual.ndim != 1 or predicted.ndim != 1:
        raise ValueError("Targets and predictions must be one-dimensional.")

    if actual.size == 0 or actual.shape != predicted.shape:
        raise ValueError(
            "Targets and predictions must be non-empty and equal in length."
        )

    if not np.isfinite(actual).all() or not np.isfinite(predicted).all():
        raise ValueError("Targets and predictions must be finite.")

    if actual.size < 2:
        raise ValueError("At least two observations are required.")

    return {
        "mae": float(mean_absolute_error(actual, predicted)),
        "rmse": float(np.sqrt(mean_squared_error(actual, predicted))),
        "r2": float(r2_score(actual, predicted)),
    }


def evaluate_quality_gate(metrics, thresholds):
    """Decide whether a candidate meets the configured quality thresholds."""

    failures = []

    if metrics["mae"] > thresholds["max_mae"]:
        failures.append(
            f"MAE {metrics['mae']:.2f} exceeds "
            f"maximum {thresholds['max_mae']:.2f}."
        )

    if metrics["rmse"] > thresholds["max_rmse"]:
        failures.append(
            f"RMSE {metrics['rmse']:.2f} exceeds "
            f"maximum {thresholds['max_rmse']:.2f}."
        )

    if metrics["r2"] < thresholds["min_r2"]:
        failures.append(
            f"R² {metrics['r2']:.4f} is below "
            f"minimum {thresholds['min_r2']:.4f}."
        )

    return {
        "passed": not failures,
        "failures": failures,
    }