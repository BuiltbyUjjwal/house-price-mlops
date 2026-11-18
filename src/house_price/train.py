import json
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn

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
from house_price.tracking import configure_mlflow


PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_DIR = PROJECT_ROOT / "models"
REPORT_DIR = PROJECT_ROOT / "reports"


def main():
    """Train, evaluate and track a house price model."""

    # 1. Load configuration and initialize MLflow
    config = load_config()

    experiment = configure_mlflow()

    # 2. Load dataset
    print("\nLoading dataset...")

    df = load_raw_data()

    X = df.drop(
        columns=[TARGET_COLUMN, "Id"],
        errors="ignore",
    )

    y = df[TARGET_COLUMN]

    # 3. Split data
    X_train, X_valid, y_train, y_valid = train_test_split(
        X,
        y,
        test_size=config["validation"]["test_size"],
        random_state=config["random_state"],
    )

    # 4. Build preprocessing and model pipeline
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

    # 5. Create an MLflow run
    with mlflow.start_run(
        experiment_id=experiment.experiment_id,
        run_name="random-forest-baseline",
    ) as run:

        print("\nTraining candidate model...")

        # Log parameters
        params = {
            "model_type": "RandomForestRegressor",
            "n_estimators": config["model"]["n_estimators"],
            "random_state": config["random_state"],
            "test_size": config["validation"]["test_size"],
            "n_jobs": config["model"].get("n_jobs", -1),
        }

        mlflow.log_params(params)

        # Train
        pipeline.fit(X_train, y_train)

        # Predict
        predictions = pipeline.predict(X_valid)

        # Evaluate
        metrics = calculate_regression_metrics(
            y_valid,
            predictions,
        )

        # Quality gate
        gate_result = evaluate_quality_gate(
            metrics,
            config["quality_gate"],
        )

        # Create directories
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        REPORT_DIR.mkdir(parents=True, exist_ok=True)

        # Create report
        report = {
            "metrics": metrics,
            "quality_gate": gate_result,
            "configuration": config,
            "training_samples": len(X_train),
            "validation_samples": len(X_valid),
        }

        metrics_path = REPORT_DIR / "metrics.json"

        with metrics_path.open("w", encoding="utf-8") as file:
            json.dump(report, file, indent=4)

        # Log metrics
        mlflow.log_metrics(metrics)

        # Log run information
        mlflow.set_tags({
            "model_type": "RandomForestRegressor",
            "dataset": "Ames Housing",
            "quality_gate": (
                "passed" if gate_result["passed"] else "failed"
            ),
        })

        # Log evaluation report as an artifact
        mlflow.log_artifact(
            str(metrics_path),
            artifact_path="reports",
        )

        print("\n========== VALIDATION RESULTS ==========")

        for metric, value in metrics.items():
            print(f"{metric.upper()}: {value:,.4f}")

        # Save model only if quality gate passes
        if gate_result["passed"]:

            model_path = MODEL_DIR / "house_price_model.joblib"

            joblib.dump(pipeline, model_path)

            # Save model into MLflow
            model_info = mlflow.sklearn.log_model(
                sk_model=pipeline,
                name="house_price_pipeline",
                skops_trusted_types=[
                    "numpy.dtype",
                    "sklearn.tree._tree.Tree",
                ],
            )

            print("\nQUALITY GATE: PASSED")
            print(f"Local model: {model_path}")
            print(f"MLflow model: {model_info.model_uri}")

        else:
            print("\nQUALITY GATE: FAILED")

            for failure in gate_result["failures"]:
                print(f"- {failure}")

        run_id = run.info.run_id

    print("\n========== MLFLOW TRACKING ==========")
    print(f"Experiment ID: {experiment.experiment_id}")
    print(f"Run ID: {run_id}")

    if not gate_result["passed"]:
        raise SystemExit(
            "Training completed, but the candidate failed the quality gate."
        )


if __name__ == "__main__":
    main()