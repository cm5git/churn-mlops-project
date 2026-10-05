"""
Tests for the churn prediction API (app.py).

The trained pipeline is never loaded here: get_model() is patched with a
fake, so these tests run anywhere, including a CI machine that has no
mlflow.db and no trained model. The pipeline's own behaviour is covered in
test_features.py.
"""

from unittest.mock import MagicMock, patch

import joblib
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sklearn.linear_model import LogisticRegression

import app as app_module
from app import app
from src.features import build_pipeline
from tests.test_features import make_training_data

# "No internet service" is expected to be treated as "No" (an unknown category),
# so scikit-learn's warning about it is expected in these tests.
pytestmark = pytest.mark.filterwarnings("ignore:Found unknown categories")

client = TestClient(app)

VALID_CUSTOMER = {
    "gender": "Female", "SeniorCitizen": 0, "Partner": "Yes", "Dependents": "No",
    "tenure": 12, "PhoneService": "Yes", "MultipleLines": "No",
    "InternetService": "DSL", "OnlineSecurity": "No", "OnlineBackup": "No",
    "DeviceProtection": "No", "TechSupport": "No", "StreamingTV": "No",
    "StreamingMovies": "No", "Contract": "Month-to-month",
    "PaperlessBilling": "Yes", "PaymentMethod": "Electronic check",
    "MonthlyCharges": 70.0,
}


def fake_model(prob_churn: float) -> MagicMock:
    model = MagicMock()
    model.predict_proba.return_value = [[1 - prob_churn, prob_churn]]
    return model


# ---------- Endpoint tests ----------

def test_root_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    assert "message" in response.json()


def test_health_works_without_a_trained_model():
    # No patching: passes on a machine with no mlflow.db at all, which is the
    # benefit of loading the model lazily instead of at import time.
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}


@patch("app.get_model")
def test_predict_returns_valid_response(mock_get_model):
    mock_get_model.return_value = fake_model(0.7)

    response = client.post("/predict", json=VALID_CUSTOMER)

    assert response.status_code == 200
    body = response.json()
    assert body["churn_probability"] == 0.7
    assert body["churn_prediction"] == 1


@pytest.mark.parametrize("prob_churn, expected_prediction", [
    (0.7, 1),
    (0.51, 1),
    (0.5, 0),   # the threshold is strictly greater than 0.5
    (0.2, 0),
])
@patch("app.get_model")
def test_prediction_threshold(mock_get_model, prob_churn, expected_prediction):
    mock_get_model.return_value = fake_model(prob_churn)

    response = client.post("/predict", json=VALID_CUSTOMER)

    assert response.json()["churn_prediction"] == expected_prediction


@patch("app.get_model")
def test_predict_passes_one_raw_row_to_the_pipeline(mock_get_model):
    """The API does no preprocessing: the pipeline receives the raw fields as-is."""
    model = fake_model(0.4)
    mock_get_model.return_value = model

    client.post("/predict", json=VALID_CUSTOMER)

    X = model.predict_proba.call_args[0][0]
    assert isinstance(X, pd.DataFrame)
    assert X.shape == (1, 18)
    assert X["Contract"].iloc[0] == "Month-to-month"
    assert X["tenure"].iloc[0] == 12


@patch("app.get_model")
def test_predict_returns_500_when_model_is_unavailable(mock_get_model):
    mock_get_model.side_effect = RuntimeError("no trained model found")
    error_client = TestClient(app, raise_server_exceptions=False)

    response = error_client.post("/predict", json=VALID_CUSTOMER)

    assert response.status_code == 500


# ---------- Validation tests ----------

def test_predict_rejects_missing_fields():
    response = client.post("/predict", json={"gender": "Female", "tenure": 12})
    assert response.status_code == 422


def test_predict_rejects_wrong_type():
    bad_customer = {**VALID_CUSTOMER, "tenure": "not_a_number"}
    response = client.post("/predict", json=bad_customer)
    assert response.status_code == 422


# ---------- Allowed-value validation ----------

@pytest.mark.parametrize("field, bad_value", [
    ("Contract", "Two years"),        # the typo that used to be silently accepted
    ("gender", "Other"),
    ("PaymentMethod", "Crypto"),
    ("InternetService", "Cable"),
    ("OnlineSecurity", "Maybe"),
    ("SeniorCitizen", 2),
    ("tenure", -1),
    ("MonthlyCharges", -5.0),
])
@patch("app.get_model")
def test_predict_rejects_invalid_values(mock_get_model, field, bad_value):
    response = client.post("/predict", json={**VALID_CUSTOMER, field: bad_value})

    assert response.status_code == 422
    # The error names the offending field, so the caller can see what to fix.
    assert response.json()["detail"][0]["loc"] == ["body", field]
    # Bad input must be stopped before it ever reaches the model.
    mock_get_model.assert_not_called()


@pytest.mark.parametrize("field, value", [
    ("Contract", "Month-to-month"),
    ("Contract", "One year"),
    ("Contract", "Two year"),
    ("MultipleLines", "No phone service"),
    ("OnlineSecurity", "No internet service"),
    ("InternetService", "No"),
    ("tenure", 0),
])
@patch("app.get_model")
def test_predict_accepts_every_allowed_value(mock_get_model, field, value):
    mock_get_model.return_value = fake_model(0.3)

    response = client.post("/predict", json={**VALID_CUSTOMER, field: value})

    assert response.status_code == 200


# ---------- Model loading ----------

def test_load_model_reads_the_file_named_by_model_path(tmp_path, monkeypatch):
    model_file = tmp_path / "tiny.joblib"
    joblib.dump(LogisticRegression().fit([[0], [1]], [0, 1]), model_file)
    monkeypatch.setenv("MODEL_PATH", str(model_file))

    model = app_module.load_model()

    assert model.predict_proba([[1]])[0][1] > 0.5


def test_load_model_gives_a_helpful_error_when_file_is_missing(tmp_path, monkeypatch):
    monkeypatch.setenv("MODEL_PATH", str(tmp_path / "missing.joblib"))

    with pytest.raises(FileNotFoundError, match="src.train"):
        app_module.load_model()


def test_get_model_loads_once_and_reuses_the_cached_copy(monkeypatch):
    monkeypatch.setattr(app_module, "_model", None)  # start with an empty cache

    with patch("app.load_model", return_value="fake-model") as mock_load:
        first = app_module.get_model()
        second = app_module.get_model()

    assert first == second == "fake-model"
    mock_load.assert_called_once()


# ---------- Integration: real API + real fitted pipeline, no mocks ----------

@pytest.fixture
def real_pipeline():
    X, y = make_training_data()
    return build_pipeline().fit(X, y)


@patch("app.get_model")
def test_api_works_end_to_end_with_a_real_pipeline(mock_get_model, real_pipeline):
    mock_get_model.return_value = real_pipeline

    response = client.post("/predict", json=VALID_CUSTOMER)

    assert response.status_code == 200
    assert 0 <= response.json()["churn_probability"] <= 1


@patch("app.get_model")
def test_api_prediction_responds_to_categorical_fields(mock_get_model, real_pipeline):
    """
    End-to-end version of the original serving bug: changing ONLY the contract
    must change the prediction. When categoricals were silently zeroed, it didn't.
    """
    mock_get_model.return_value = real_pipeline

    month_to_month = client.post("/predict", json={**VALID_CUSTOMER, "Contract": "Month-to-month"})
    two_year = client.post("/predict", json={**VALID_CUSTOMER, "Contract": "Two year"})

    assert month_to_month.json()["churn_probability"] != two_year.json()["churn_probability"]