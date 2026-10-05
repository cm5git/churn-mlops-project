"""
Tests for the two model choices (champion / challenger) and the training helpers.

Everything here fits small pipelines on synthetic data, so it needs no dataset
and no MLflow server.
"""

import pytest
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

from app import DEFAULT_MODEL_PATH
from src.features import CHALLENGER, CHAMPION, MODEL_NAMES, build_classifier, build_pipeline
from src.train import MODEL_PATHS, cross_validate_model, evaluate_model, parse_args
from tests.test_features import BASE_CUSTOMER, make_training_data

# "No internet service" is expected to be treated as "No" (an unknown category).
# Models fitted on random labels may also predict only one class, which makes
# precision undefined; neither matters for what these tests check.
pytestmark = [
    pytest.mark.filterwarnings("ignore:Found unknown categories"),
    pytest.mark.filterwarnings("ignore::sklearn.exceptions.UndefinedMetricWarning"),
]


@pytest.fixture(scope="module")
def data():
    return make_training_data(n=300)


# ---------- Both models behave the same way from the outside ----------

@pytest.mark.parametrize("name", MODEL_NAMES)
def test_pipeline_gives_valid_probabilities(name, data):
    X, y = data
    probabilities = build_pipeline(name).fit(X, y).predict_proba(X)

    assert probabilities.shape == (len(X), 2)
    assert ((probabilities >= 0) & (probabilities <= 1)).all()
    assert probabilities.sum(axis=1) == pytest.approx(1.0)


@pytest.mark.parametrize("name", MODEL_NAMES)
def test_single_row_matches_batch_prediction(name, data):
    """The serving bug again: one row must be encoded exactly like a batch."""
    X, y = data
    pipeline = build_pipeline(name).fit(X, y)
    batch = X.head(15)
    batch_probs = pipeline.predict_proba(batch)[:, 1]

    for i in range(len(batch)):
        assert pipeline.predict_proba(batch.iloc[[i]])[0][1] == pytest.approx(batch_probs[i])


@pytest.mark.parametrize("name", MODEL_NAMES)
def test_unseen_category_does_not_crash(name, data):
    import pandas as pd

    X, y = data
    pipeline = build_pipeline(name).fit(X, y)
    customer = pd.DataFrame([{**BASE_CUSTOMER, "PaymentMethod": "Crypto"}])

    assert 0 <= pipeline.predict_proba(customer)[0][1] <= 1


# ---------- The two models are what we say they are ----------

def test_champion_and_challenger_are_different_model_types():
    assert isinstance(build_classifier(CHAMPION), LogisticRegression)
    assert isinstance(build_classifier(CHALLENGER), HistGradientBoostingClassifier)


def test_both_models_are_told_missing_a_churner_costs_more():
    """Same class weighting for both, so the comparison is fair."""
    assert build_classifier(CHAMPION).class_weight == "balanced"
    assert build_classifier(CHALLENGER).class_weight == "balanced"


def test_unknown_model_name_is_rejected():
    with pytest.raises(ValueError, match="champion, challenger"):
        build_classifier("magic")


# ---------- Where each model is saved ----------

def test_champion_is_saved_where_the_api_loads_it_by_default():
    assert MODEL_PATHS[CHAMPION] == DEFAULT_MODEL_PATH


def test_challenger_does_not_overwrite_the_champion():
    assert MODEL_PATHS[CHALLENGER] != MODEL_PATHS[CHAMPION]


def test_training_defaults_to_the_champion():
    assert parse_args([]).model == CHAMPION
    assert parse_args(["--model", "challenger"]).model == CHALLENGER


def test_training_rejects_an_unknown_model_name():
    with pytest.raises(SystemExit):
        parse_args(["--model", "magic"])


# ---------- Evaluation helpers ----------

def test_evaluate_model_reports_threshold_free_metrics(data):
    X, y = data
    pipeline = build_pipeline(CHAMPION).fit(X, y)

    metrics = evaluate_model(pipeline, X, y)

    assert {"roc_auc", "pr_auc", "recall_churn", "precision_churn"} <= set(metrics)
    assert 0 <= metrics["roc_auc"] <= 1
    assert 0 <= metrics["pr_auc"] <= 1


def test_cross_validation_reports_mean_and_spread(data):
    X, y = data

    scores = cross_validate_model(build_pipeline(CHAMPION), X, y, folds=3)

    assert set(scores) == {"cv_roc_auc_mean", "cv_roc_auc_std", "cv_pr_auc_mean", "cv_pr_auc_std"}
    assert 0 <= scores["cv_roc_auc_mean"] <= 1
    assert scores["cv_roc_auc_std"] >= 0