import json
from pathlib import Path

import joblib
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from house_price.data import TARGET_COLUMN, load_raw_data


# Project directories
PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = PROJECT_ROOT / "models"
REPORT_DIR = PROJECT_ROOT / "reports"


def build_preprocessor(X):
    """Create preprocessing steps for numerical and categorical features."""

    numerical_features = X.select_dtypes(
        include=["number"]
    ).columns.tolist()

    categorical_features = X.select_dtypes(
        include=["object", "category", "bool"]
    ).columns.tolist()

    # Numerical features
    numerical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )

    # Categorical features
    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="constant",
                    fill_value="Missing",
                ),
            ),
            (
                "encoder",
                OneHotEncoder(handle_unknown="ignore"),
            ),
        ]
    )

    # Combine both preprocessing pipelines
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numerical_pipeline, numerical_features),
            ("cat", categorical_pipeline, categorical_features),
        ],
        remainder="drop",
    )

    return preprocessor


def main():
    """Train, evaluate, and save the house price prediction pipeline."""

    print("Loading dataset...")
    df = load_raw_data()

    # Separate features and target
    X = df.drop(columns=[TARGET_COLUMN, "Id"], errors="ignore")
    y = df[TARGET_COLUMN]

    print(f"Dataset shape: {df.shape}")
    print(f"Feature count: {X.shape[1]}")

    # Split before fitting any preprocessing steps
    X_train, X_valid, y_train, y_valid = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
    )

    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_valid)}")

    # Build preprocessing
    preprocessor = build_preprocessor(X_train)

    # Define the model
    model = RandomForestRegressor(
        n_estimators=200,
        random_state=42,
        n_jobs=-1,
    )

    # Combine preprocessing and model into one pipeline
    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )

    print("\nTraining model...")
    pipeline.fit(X_train, y_train)

    print("Generating predictions...")
    predictions = pipeline.predict(X_valid)

    # Calculate evaluation metrics
    metrics = {
        "mae": float(mean_absolute_error(y_valid, predictions)),
        "rmse": float(
            np.sqrt(mean_squared_error(y_valid, predictions))
        ),
        "r2": float(r2_score(y_valid, predictions)),
    }

    print("\n========== VALIDATION RESULTS ==========")
    print(f"MAE:  {metrics['mae']:,.2f}")
    print(f"RMSE: {metrics['rmse']:,.2f}")
    print(f"R²:   {metrics['r2']:.4f}")

    # Save model and metrics
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    model_path = MODEL_DIR / "house_price_model.joblib"
    metrics_path = REPORT_DIR / "metrics.json"

    joblib.dump(pipeline, model_path)

    with metrics_path.open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=4)

    print("\n========== ARTIFACTS SAVED ==========")
    print(f"Model:   {model_path}")
    print(f"Metrics: {metrics_path}")


if __name__ == "__main__":
    main()