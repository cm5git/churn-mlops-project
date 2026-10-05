"""
FastAPI service for serving churn predictions.

The model is a complete scikit-learn Pipeline (encoding, scaling and
classifier) saved as one file, so this file does no preprocessing of its own:
it hands the raw customer fields straight to the pipeline. Serving needs only
scikit-learn, not MLflow.
"""

import os
import warnings
from pathlib import Path
from typing import Literal

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

# The yes/no columns rely on "unknown category -> all zeros" to treat
# "No internet service" as "No" (see src/features.py), so scikit-learn's
# "unknown categories" warning is expected and would only add log noise.
warnings.filterwarnings("ignore", message="Found unknown categories", category=UserWarning)

# Where the trained pipeline lives. src/train.py writes it here. Set the
# MODEL_PATH environment variable to load it from somewhere else (for example
# a mounted volume or a downloaded registry artifact).
DEFAULT_MODEL_PATH = Path(__file__).resolve().parent / "model" / "model.joblib"

app = FastAPI(title="Churn Prediction API")

# Cache for the loaded pipeline. None until first use, so importing this file
# does not require a trained model to exist.
_model = None


def load_model():
    """
    Load the saved pipeline from disk. Only load files you created yourself:
    joblib files are pickles, and unpickling untrusted data can run code.
    """
    path = Path(os.environ.get("MODEL_PATH", DEFAULT_MODEL_PATH))
    if not path.exists():
        raise FileNotFoundError(
            f"No trained model at {path}. Run `python3 -m src.train` to create it."
        )
    model = joblib.load(path)
    print(f"Loaded model from {path}")
    return model


def get_model():
    """
    Lazy loading: load on first use, then reuse the cached copy. The app can
    start and answer /health without a trained model, and tests can patch
    this function before it is ever called.
    """
    global _model
    if _model is None:
        _model = load_model()
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