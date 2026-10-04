"""
FastAPI service for serving churn predictions.

The model artifact is a complete scikit-learn Pipeline (encoding, scaling
and classifier), so this file does no preprocessing of its own: it hands the
raw customer fields straight to the pipeline.
"""

import warnings
from typing import Literal  
import mlflow
import mlflow.sklearn
import pandas as pd
from fastapi import FastAPI
from mlflow.tracking import MlflowClient
from pydantic import BaseModel,Field

# The yes/no columns rely on "unknown category -> all zeros" to treat
# "No internet service" as "No" (see src/features.py), so scikit-learn's
# "unknown categories" warning is expected and would only add log noise.
warnings.filterwarnings("ignore", message="Found unknown categories", category=UserWarning)

# Set the tracking location explicitly in code, so the app doesn't depend on a
# hidden environment variable (a Docker container or CI runner wouldn't inherit it).
mlflow.set_tracking_uri("sqlite:///mlflow.db")

app = FastAPI(title="Churn Prediction API")

# Cache for the loaded pipeline. None until first use, so importing this file
# does not require a trained model to exist.
_model = None


def load_latest_model():
    """Load the pipeline from the most recent MLflow run."""
    client = MlflowClient()
    experiment = client.get_experiment_by_name("Default")
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["start_time DESC"],
        max_results=1,
    )
    run_id = runs[0].info.run_id
    model = mlflow.sklearn.load_model(f"runs:/{run_id}/model")
    print(f"Loaded model from run {run_id}")
    return model


def get_model():
    """
    Lazy loading: load on first use, then reuse the cached copy. The app can
    start and answer /health without a trained model, and tests can patch
    this function before it is ever called.
    """
    global _model
    if _model is None:
        _model = load_latest_model()
    return _model


class CustomerData(BaseModel):
    """
    Raw customer fields, exactly as they appear before any preprocessing.

    Categorical fields only accept the values seen in the training data. The
    pipeline would silently treat an unknown value as the baseline category,
    so a typo like "Two years" would quietly become "Month-to-month" and give
    a wrong prediction. Rejecting it here returns a clear 422 instead.
    """
    gender: Literal["Female", "Male"]
    SeniorCitizen: Literal[0, 1]
    Partner: Literal["Yes", "No"]
    Dependents: Literal["Yes", "No"]
    tenure: int = Field(ge=0)
    PhoneService: Literal["Yes", "No"]
    MultipleLines: Literal["Yes", "No", "No phone service"]
    InternetService: Literal["DSL", "Fiber optic", "No"]
    OnlineSecurity: Literal["Yes", "No", "No internet service"]
    OnlineBackup: Literal["Yes", "No", "No internet service"]
    DeviceProtection: Literal["Yes", "No", "No internet service"]
    TechSupport: Literal["Yes", "No", "No internet service"]
    StreamingTV: Literal["Yes", "No", "No internet service"]
    StreamingMovies: Literal["Yes", "No", "No internet service"]
    Contract: Literal["Month-to-month", "One year", "Two year"]
    PaperlessBilling: Literal["Yes", "No"]
    PaymentMethod: Literal[
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    ]
    MonthlyCharges: float = Field(ge=0)
    
@app.get("/")
def root():
    return {"message": "Churn Prediction API is running"}


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/predict")
def predict(customer: CustomerData):
    model = get_model()

    # One row of raw fields. The pipeline does all encoding and scaling.
    X = pd.DataFrame([customer.model_dump()])
    churn_probability = model.predict_proba(X)[0][1]  # probability of class 1 (churn)
    prediction = int(churn_probability > 0.5)

    return {
        "churn_probability": round(float(churn_probability), 4),
        "churn_prediction": prediction,
    }