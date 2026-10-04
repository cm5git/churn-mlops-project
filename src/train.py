"""
Training script for the churn prediction model.

Fits ONE scikit-learn Pipeline (preprocessing + classifier) and logs it to
MLflow as a single artifact. Run from the project root:

    python3 -m src.train

(-m runs it as part of the `src` package, which is what lets it import
`src.features`.)
"""

import warnings
from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.metrics import (accuracy_score, classification_report, f1_score,
                             precision_score, recall_score)
from sklearn.model_selection import train_test_split

from src.features import FEATURE_COLUMNS, build_pipeline

# The yes/no columns rely on "unknown category -> all zeros" to treat
# "No internet service" as "No" (see src/features.py), so scikit-learn's
# "unknown categories" warning is expected here.
warnings.filterwarnings("ignore", message="Found unknown categories", category=UserWarning)


def load_data(path: Path) -> pd.DataFrame:
    """Load the raw Telco churn CSV."""
    return pd.read_csv(path)


def prepare_data(data: pd.DataFrame):
    """Select the raw feature columns and encode the target as 1/0."""
    X = data[FEATURE_COLUMNS]
    y = data["Churn"].map({"Yes": 1, "No": 0})
    return X, y


def split_data(X, y, test_size: float = 0.2, random_state: int = 42):
    """Stratified split to preserve the churn ratio in both sets."""
    return train_test_split(X, y, test_size=test_size, stratify=y, random_state=random_state)


def evaluate_model(pipeline, X_test, y_test) -> dict:
    """Return key metrics, focusing on the churn (positive) class."""
    y_pred = pipeline.predict(X_test)
    print(classification_report(y_test, y_pred))
    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision_churn": precision_score(y_test, y_pred),
        "recall_churn": recall_score(y_test, y_pred),
        "f1_churn": f1_score(y_test, y_pred),
    }


def main():
    project_root = Path(__file__).resolve().parent.parent
    data_path = project_root / "data" / "raw" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"

    X, y = prepare_data(load_data(data_path))
    X_train, X_test, y_train, y_test = split_data(X, y)

    mlflow.set_tracking_uri("sqlite:///mlflow.db")

    with mlflow.start_run():
        # Fitting the whole pipeline on the training set means the scaler and
        # encoder also learn from training data only.
        pipeline = build_pipeline()
        pipeline.fit(X_train, y_train)

        metrics = evaluate_model(pipeline, X_test, y_test)

        mlflow.log_param("model_type", "LogisticRegression")
        mlflow.log_param("class_weight", "balanced")
        mlflow.log_param("max_iter", 1000)
        mlflow.log_param("test_size", 0.2)

        for name, value in metrics.items():
            mlflow.log_metric(name, value)

        # One artifact: preprocessing + model together. No separate
        # scaler.pkl or feature_columns.pkl needed any more.
        mlflow.sklearn.log_model(pipeline, "model")

        print("MLflow run logged:", mlflow.active_run().info.run_id)


if __name__ == "__main__":
    main()