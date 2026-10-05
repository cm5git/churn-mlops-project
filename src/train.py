"""
Training script for the churn prediction models.

Fits ONE scikit-learn Pipeline (preprocessing + classifier) and logs it to
MLflow. Run from the project root:

    python3 -m src.train                       # champion: logistic regression
    python3 -m src.train --model challenger    # challenger: gradient boosting

Each run also saves a plain copy of the fitted pipeline for serving. The
champion goes to model/model.joblib (what the API loads by default) and the
challenger to model/challenger.joblib, so training one never overwrites the
other.

(-m runs it as part of the `src` package, which is what lets it import
`src.features`.)
"""

import argparse
import inspect
import warnings
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.metrics import (accuracy_score, average_precision_score,
                             classification_report, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split

from src.features import CHALLENGER, CHAMPION, FEATURE_COLUMNS, MODEL_NAMES, build_pipeline

# The yes/no columns rely on "unknown category -> all zeros" to treat
# "No internet service" as "No" (see src/features.py), so scikit-learn's
# "unknown categories" warning is expected here.
warnings.filterwarnings("ignore", message="Found unknown categories", category=UserWarning)

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Where each model is saved for serving. The API loads the champion by default
# (see app.py); set MODEL_PATH to serve the challenger instead.
MODEL_PATHS = {
    CHAMPION: PROJECT_ROOT / "model" / "model.joblib",
    CHALLENGER: PROJECT_ROOT / "model" / "challenger.joblib",
}


# MLflow saves scikit-learn models in a "safe" format (skops) that refuses types
# it doesn't recognise unless you name them. Gradient boosting's trees are
# flagged because a maliciously edited file could crash the program. We created
# this file ourselves, so we trust that one type.
TRUSTED_TYPES = ["sklearn.ensemble._hist_gradient_boosting.predictor.TreePredictor"]


def log_pipeline(pipeline):
    """Log the fitted pipeline to MLflow as a model artifact."""
    options = {}
    # Older MLflow versions don't have this option (and don't need it).
    if "skops_trusted_types" in inspect.signature(mlflow.sklearn.log_model).parameters:
        options["skops_trusted_types"] = TRUSTED_TYPES
    mlflow.sklearn.log_model(pipeline, "model", **options)


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
    """
    Return key metrics for the churn (positive) class.

    precision/recall/F1 depend on the 0.5 cutoff. ROC-AUC and PR-AUC don't:
    they measure how well the model RANKS customers by risk, which is a fairer
    way to compare two different models.
    """
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]
    print(classification_report(y_test, y_pred))
    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision_churn": precision_score(y_test, y_pred),
        "recall_churn": recall_score(y_test, y_pred),
        "f1_churn": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_prob),
        "pr_auc": average_precision_score(y_test, y_prob),
    }


def cross_validate_model(pipeline, X_train, y_train, folds: int = 5) -> dict:
    """
    Cross-validate on the TRAINING data only. The mean shows typical
    performance and the standard deviation shows how much it moves between
    folds, which tells you how big a difference between two models has to be
    before it means anything. The whole pipeline is refitted inside each fold,
    so the scaler and encoder never see the held-out part.
    """
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)
    scores = cross_validate(
        pipeline, X_train, y_train, cv=cv,
        scoring={"roc_auc": "roc_auc", "pr_auc": "average_precision"},
    )
    return {
        "cv_roc_auc_mean": scores["test_roc_auc"].mean(),
        "cv_roc_auc_std": scores["test_roc_auc"].std(),
        "cv_pr_auc_mean": scores["test_pr_auc"].mean(),
        "cv_pr_auc_std": scores["test_pr_auc"].std(),
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Train a churn model.")
    parser.add_argument(
        "--model", choices=MODEL_NAMES, default=CHAMPION,
        help="champion (logistic regression, default) or challenger (gradient boosting)",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    data_path = PROJECT_ROOT / "data" / "raw" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"

    X, y = prepare_data(load_data(data_path))
    X_train, X_test, y_train, y_test = split_data(X, y)

    mlflow.set_tracking_uri("sqlite:///mlflow.db")

    with mlflow.start_run(run_name=args.model):
        pipeline = build_pipeline(args.model)
        classifier = pipeline.named_steps["model"]

        cv_metrics = cross_validate_model(pipeline, X_train, y_train)

        # Fitting the whole pipeline on the training set means the scaler and
        # encoder also learn from training data only.
        pipeline.fit(X_train, y_train)
        metrics = evaluate_model(pipeline, X_test, y_test)

        mlflow.set_tag("model_name", args.model)
        mlflow.log_param("model_name", args.model)
        mlflow.log_param("model_type", type(classifier).__name__)
        mlflow.log_param("test_size", 0.2)
        mlflow.log_params({f"clf_{k}": v for k, v in classifier.get_params().items()})

        for name, value in {**metrics, **cv_metrics}.items():
            mlflow.log_metric(name, float(value))

        # One artifact: preprocessing + model together.
        log_pipeline(pipeline)

        # Also save a plain copy for serving. The API loads this file, so the
        # serving image needs scikit-learn but not MLflow.
        model_path = MODEL_PATHS[args.model]
        model_path.parent.mkdir(exist_ok=True)
        joblib.dump(pipeline, model_path)

        print(f"\n[{args.model}] {type(classifier).__name__}")
        print(f"  cross-validated ROC-AUC: {cv_metrics['cv_roc_auc_mean']:.4f} +/- {cv_metrics['cv_roc_auc_std']:.4f}")
        print(f"  cross-validated PR-AUC:  {cv_metrics['cv_pr_auc_mean']:.4f} +/- {cv_metrics['cv_pr_auc_std']:.4f}")
        print(f"  test ROC-AUC {metrics['roc_auc']:.4f} | test PR-AUC {metrics['pr_auc']:.4f}")
        print("Model saved to", model_path)
        print("MLflow run logged:", mlflow.active_run().info.run_id)


if __name__ == "__main__":
    main()