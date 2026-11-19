import numpy as np
from fastapi.testclient import TestClient

from house_price import api


client = TestClient(api.app)


class DummyModel:
    feature_names_in_ = np.array(["OverallQual", "GrLivArea"])

    def predict(self, data):
        assert list(data.columns) == [
            "OverallQual",
            "GrLivArea",
        ]
        return np.array([250000.0])


def test_prediction_success(monkeypatch):
    monkeypatch.setattr(
        api, "get_model", lambda: DummyModel()
    )

    response = client.post(
        "/predict",
        json={
            "features": {
                "OverallQual": 7,
                "GrLivArea": 1710,
            }
        },
    )

    assert response.status_code == 200
    assert response.json()["predicted_sale_price"] == 250000.0


def test_prediction_rejects_missing_features(monkeypatch):
    monkeypatch.setattr(
        api, "get_model", lambda: DummyModel()
    )

    response = client.post(
        "/predict",
        json={
            "features": {
                "OverallQual": 7,
            }
        },
    )

    assert response.status_code == 422
    assert "missing_features" in response.json()["detail"]


def test_prediction_rejects_unexpected_features(monkeypatch):
    monkeypatch.setattr(
        api, "get_model", lambda: DummyModel()
    )

    response = client.post(
        "/predict",
        json={
            "features": {
                "OverallQual": 7,
                "GrLivArea": 1710,
                "WrongFeature": 100,
            }
        },
    )

    assert response.status_code == 422
    assert "unexpected_features" in response.json()["detail"]

def test_health_success(monkeypatch):
    monkeypatch.setattr(
        api, "get_model", lambda: DummyModel()
    )

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_health_model_missing(monkeypatch, tmp_path):
    monkeypatch.setattr(api, "_model", None)

    monkeypatch.setattr(
        api,
        "MODEL_PATH",
        tmp_path / "missing_model.joblib",
    )

    response = client.get("/health")

    assert response.status_code == 503