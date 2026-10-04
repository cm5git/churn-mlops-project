"""
Tests for the shared feature pipeline (src/features.py).

These fit a small pipeline on synthetic data inside the test, so they need
no dataset, no trained model and no MLflow.
"""

import numpy as np
import pandas as pd
import pytest

from src.features import FEATURE_COLUMNS, build_pipeline

# "No internet service" is expected to be treated as "No" (an unknown
# category), so scikit-learn's warning about it is expected here.
pytestmark = pytest.mark.filterwarnings("ignore:Found unknown categories")

CATEGORIES = {
    "gender": ["Female", "Male"],
    "Partner": ["No", "Yes"],
    "Dependents": ["No", "Yes"],
    "PhoneService": ["No", "Yes"],
    "MultipleLines": ["No", "Yes", "No phone service"],
    "InternetService": ["DSL", "Fiber optic", "No"],
    "OnlineSecurity": ["No", "Yes", "No internet service"],
    "OnlineBackup": ["No", "Yes", "No internet service"],
    "DeviceProtection": ["No", "Yes", "No internet service"],
    "TechSupport": ["No", "Yes", "No internet service"],
    "StreamingTV": ["No", "Yes", "No internet service"],
    "StreamingMovies": ["No", "Yes", "No internet service"],
    "Contract": ["Month-to-month", "One year", "Two year"],
    "PaperlessBilling": ["No", "Yes"],
    "PaymentMethod": ["Electronic check", "Mailed check",
                      "Bank transfer (automatic)", "Credit card (automatic)"],
}

BASE_CUSTOMER = {
    "gender": "Female", "SeniorCitizen": 0, "Partner": "Yes", "Dependents": "No",
    "tenure": 12, "PhoneService": "Yes", "MultipleLines": "No",
    "InternetService": "DSL", "OnlineSecurity": "No", "OnlineBackup": "No",
    "DeviceProtection": "No", "TechSupport": "No", "StreamingTV": "No",
    "StreamingMovies": "No", "Contract": "Month-to-month",
    "PaperlessBilling": "Yes", "PaymentMethod": "Electronic check",
    "MonthlyCharges": 70.0,
}


def make_training_data(n=300, seed=0):
    rng = np.random.default_rng(seed)
    data = {col: rng.choice(options, n) for col, options in CATEGORIES.items()}
    data["SeniorCitizen"] = rng.integers(0, 2, n)
    data["tenure"] = rng.integers(0, 73, n)
    data["MonthlyCharges"] = rng.uniform(18, 118, n)
    return pd.DataFrame(data)[FEATURE_COLUMNS], pd.Series(rng.integers(0, 2, n))


@pytest.fixture(scope="module")
def training_data():
    return make_training_data()


@pytest.fixture(scope="module")
def pipeline(training_data):
    X, y = training_data
    return build_pipeline().fit(X, y)


def customer(**overrides) -> pd.DataFrame:
    """One raw customer row, as the API would build it."""
    return pd.DataFrame([{**BASE_CUSTOMER, **overrides}])


def encode(pipeline, df) -> pd.DataFrame:
    """Run only the preprocessing step and label the resulting columns."""
    pre = pipeline.named_steps["preprocess"]
    return pd.DataFrame(pre.transform(df), columns=pre.get_feature_names_out())


def test_single_row_matches_batch_prediction(pipeline, training_data):
    """
    Regression test for the serving bug: encoding ONE row must give the same
    result as encoding it as part of a batch. A hand-rolled get_dummies on a
    single row did not.
    """
    X, _ = training_data
    batch = X.head(20)
    batch_probs = pipeline.predict_proba(batch)[:, 1]

    for i in range(len(batch)):
        single_prob = pipeline.predict_proba(batch.iloc[[i]])[:, 1][0]
        assert single_prob == pytest.approx(batch_probs[i])


def test_two_year_contract_is_encoded(pipeline):
    encoded = encode(pipeline, customer(Contract="Two year"))
    assert encoded["cat__Contract_Two year"].iloc[0] == 1
    assert encoded["cat__Contract_One year"].iloc[0] == 0


def test_baseline_category_is_all_zeros(pipeline):
    """Month-to-month is the dropped baseline, so both contract columns are 0."""
    encoded = encode(pipeline, customer(Contract="Month-to-month"))
    assert encoded["cat__Contract_One year"].iloc[0] == 0
    assert encoded["cat__Contract_Two year"].iloc[0] == 0


def test_yes_is_encoded_as_one(pipeline):
    encoded = encode(pipeline, customer(OnlineSecurity="Yes"))
    assert encoded["yes_no__OnlineSecurity_Yes"].iloc[0] == 1


def test_no_internet_service_is_treated_as_no(pipeline):
    """The consolidation rule: 'No internet service' must encode exactly like 'No'."""
    with_extra = encode(pipeline, customer(OnlineSecurity="No internet service",
                                           MultipleLines="No phone service"))
    plain_no = encode(pipeline, customer(OnlineSecurity="No", MultipleLines="No"))

    pd.testing.assert_frame_equal(with_extra, plain_no)
    assert with_extra["yes_no__OnlineSecurity_Yes"].iloc[0] == 0


def test_unseen_category_does_not_crash(pipeline):
    """A value never seen in training is encoded as the baseline, not an error."""
    df = customer(PaymentMethod="Crypto")
    encoded = encode(pipeline, df)

    payment_cols = [c for c in encoded.columns if c.startswith("cat__PaymentMethod")]
    assert (encoded[payment_cols].iloc[0] == 0).all()
    assert 0 <= pipeline.predict_proba(df)[0][1] <= 1


def test_column_order_and_extra_columns_do_not_matter(pipeline):
    df = customer()
    shuffled = df[list(reversed(df.columns))].assign(customerID="X-123")

    assert pipeline.predict_proba(shuffled)[0][1] == pytest.approx(
        pipeline.predict_proba(df)[0][1]
    )


def test_output_is_a_valid_probability(pipeline):
    prob = pipeline.predict_proba(customer())[0][1]
    assert 0 <= prob <= 1


def test_feature_count(pipeline):
    """22 model features: matches the (rows, 22) shape seen during exploration."""
    assert encode(pipeline, customer()).shape[1] == 22