import pytest

from house_price.evaluation import (
    calculate_regression_metrics,
    evaluate_quality_gate,
)


THRESHOLDS = {
    "max_mae": 30000,
    "max_rmse": 45000,
    "min_r2": 0.75,
}


def test_regression_metrics_are_calculated():
    metrics = calculate_regression_metrics(
        [100, 200, 300],
        [110, 180, 300],
    )

    assert metrics["mae"] == pytest.approx(10.0)
    assert metrics["rmse"] == pytest.approx((500 / 3) ** 0.5)
    assert "r2" in metrics


def test_quality_gate_accepts_satisfactory_model():
    metrics = {
        "mae": 18000,
        "rmse": 30000,
        "r2": 0.88,
    }

    result = evaluate_quality_gate(metrics, THRESHOLDS)

    assert result["passed"] is True
    assert result["failures"] == []


def test_quality_gate_rejects_poor_model():
    metrics = {
        "mae": 31000,
        "rmse": 46000,
        "r2": 0.70,
    }

    result = evaluate_quality_gate(metrics, THRESHOLDS)

    assert result["passed"] is False
    assert len(result["failures"]) == 3