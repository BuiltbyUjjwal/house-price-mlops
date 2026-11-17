import json
from pathlib import Path

import joblib

from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

from house_price.config import load_config
from house_price.data import TARGET_COLUMN, load_raw_data
from house_price.evaluation import (
    calculate_regression_metrics,
    evaluate_quality_gate,
)
from house_price.preprocessing import build_preprocessor


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = PROJECT_ROOT / "models"
REPORT_DIR = PROJECT_ROOT / "reports"


def main():
    """Train, evaluate, and publish a model only if it passes the gate."""

    config = load_config()

    print("Loading dataset...")
    df = load_raw_data()

    X = df.drop(columns=[TARGET_COLUMN, "Id"], errors="ignore")
    y = df[TARGET_COLUMN]

    X_train, X_valid, y_train, y_valid = train_test_split(
        X,
        y,
        test_size=config["validation"]["test_size"],
        random_state=config["random_state"],
    )

    preprocessor = build_preprocessor(X_train)

    model = RandomForestRegressor(
        n_estimators=config["model"]["n_estimators"],
        random_state=config["random_state"],
        n_jobs=config["model"].get("n_jobs", -1),
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )

    print("Training candidate model...")
    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_valid)

    metrics = calculate_regression_metrics(
        y_valid,
        predictions,
    )

    gate_result = evaluate_quality_gate(
        metrics,
        config["quality_gate"],
    )

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    model_path = MODEL_DIR / "house_price_model.joblib"
    metrics_path = REPORT_DIR / "metrics.json"

    report = {
        "metrics": metrics,
        "quality_gate": gate_result,
        "configuration": config,
        "validation_samples": int(len(y_valid)),
        "training_samples": int(len(y_train)),
    }

    # Save the evaluation report even when the candidate is rejected.
    with metrics_path.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=4)

    print("\n========== VALIDATION RESULTS ==========")
    print(f"MAE:  {metrics['mae']:,.2f}")
    print(f"RMSE: {metrics['rmse']:,.2f}")
    print(f"R²:   {metrics['r2']:.4f}")

    if not gate_result["passed"]:
        print("\nQUALITY GATE: FAILED")

        for failure in gate_result["failures"]:
            print(f"- {failure}")

        print(f"Evaluation report saved to: {metrics_path}")
        print("The model artifact was not overwritten.")

        raise SystemExit(1)

    # Only save the new model if it passes every quality check.
    joblib.dump(pipeline, model_path)

    print("\nQUALITY GATE: PASSED")
    print(f"Approved model artifact: {model_path}")
    print(f"Evaluation report: {metrics_path}")


if __name__ == "__main__":
    main()