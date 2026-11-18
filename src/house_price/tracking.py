from pathlib import Path

import mlflow


PROJECT_ROOT = Path(__file__).resolve().parents[2]

EXPERIMENT_NAME = "house-price-prediction"


def configure_mlflow():
    """Configure local MLflow tracking using SQLite."""

    database_path = PROJECT_ROOT / "mlflow.db"

    tracking_uri = f"sqlite:///{database_path.as_posix()}"

    mlflow.set_tracking_uri(tracking_uri)

    experiment = mlflow.set_experiment(EXPERIMENT_NAME)

    print(f"MLflow tracking URI: {tracking_uri}")
    print(f"Experiment: {experiment.name}")

    return experiment