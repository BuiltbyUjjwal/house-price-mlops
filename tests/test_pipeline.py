import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline

from house_price.train import build_preprocessor


def test_pipeline_handles_missing_and_unseen_values():
    """Preprocessing should handle missing and unseen feature values."""

    X_train = pd.DataFrame({
        "LotArea": [8000, 10000, np.nan, 12000, 9000],
        "Neighborhood": ["A", "B", "A", "B", "A"],
    })

    y_train = [150000, 200000, 180000, 250000, 170000]

    X_new = pd.DataFrame({
        "LotArea": [11000, np.nan],
        "Neighborhood": ["C", None],
    })

    pipeline = Pipeline([
        ("preprocessor", build_preprocessor(X_train)),
        (
            "model",
            RandomForestRegressor(
                n_estimators=10,
                random_state=42,
            ),
        ),
    ])

    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_new)

    assert len(predictions) == 2
    assert np.isfinite(predictions).all()