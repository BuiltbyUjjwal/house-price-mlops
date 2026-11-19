from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = PROJECT_ROOT / "models" / "house_price_model.joblib"

_model = None

app = FastAPI(
    title="House Price Prediction API",
    description="Predict house prices using a trained ML pipeline.",
    version="1.0.0",
)


class PredictionRequest(BaseModel):
    features: dict[str, Any] = Field(
        ...,
        description="All input features required by the trained model."
    )


def get_model():
    global _model

    if _model is None:
        if not MODEL_PATH.is_file():
            raise HTTPException(
                status_code=503,
                detail="Model artifact not found. Train or retrieve the model first."
            )

        try:
            _model = joblib.load(MODEL_PATH)
        except Exception as exc:
            raise HTTPException(
                status_code=503,
                detail=f"Could not load model: {exc}"
            ) from exc

    return _model


@app.get("/")
def root():
    return {
        "service": "House Price Prediction API",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health")
def health():
    available = MODEL_PATH.is_file()

    return {
        "status": "ok" if available else "model_missing",
        "model_available": available,
    }


@app.post("/predict")
def predict(request: PredictionRequest):
    model = get_model()

    if not hasattr(model, "feature_names_in_"):
        raise HTTPException(
            status_code=500,
            detail="Model does not expose its expected feature names."
        )

    expected = list(model.feature_names_in_)
    supplied = set(request.features)

    missing = sorted(set(expected) - supplied)
    unexpected = sorted(supplied - set(expected))

    if missing or unexpected:
        raise HTTPException(
            status_code=422,
            detail={
                "missing_features": missing,
                "unexpected_features": unexpected,
            },
        )

    try:
        input_data = pd.DataFrame(
            [request.features],
            columns=expected,
        )

        prediction = float(model.predict(input_data)[0])

    except (ValueError, TypeError) as exc:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid feature values: {exc}",
        ) from exc

    return {
        "predicted_sale_price": round(prediction, 2)
    }