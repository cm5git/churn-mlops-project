"""
Shared feature pipeline, used by BOTH training and serving.

Everything between "raw customer fields" and "model prediction" lives in
one scikit-learn Pipeline: encoding, scaling and the classifier. It is
fitted once in training and saved as a single artifact, so serving just
calls it on raw data. There is no second, hand-written copy of the
preprocessing to drift out of sync.

Two classifiers can sit at the end of the pipeline, sharing identical
preprocessing so that they are compared fairly:

    champion   - logistic regression (simple, interpretable)
    challenger - gradient boosting (more flexible, can capture interactions)

The pipeline uses only built-in scikit-learn components (no custom
functions), so the saved artifact can be loaded anywhere scikit-learn is
installed, without needing this project's code on the import path.
"""

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

CHAMPION = "champion"
CHALLENGER = "challenger"
MODEL_NAMES = (CHAMPION, CHALLENGER)

NUMERIC_FEATURES = ["tenure", "MonthlyCharges"]
PASSTHROUGH_FEATURES = ["SeniorCitizen"]  # already 0/1, used as-is

# These columns hold "Yes" / "No", plus a third value ("No internet service"
# or "No phone service") that only repeats what InternetService / PhoneService
# already say. We want that third value treated the same as "No".
YES_NO_FEATURES = [
    "MultipleLines", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies",
]

# Every other categorical column: the encoder learns its categories from the data.
OTHER_CATEGORICAL_FEATURES = [
    "gender", "Partner", "Dependents", "PhoneService", "InternetService",
    "Contract", "PaperlessBilling", "PaymentMethod",
]

# The raw columns the model uses. customerID and TotalCharges are left out
# on purpose (identifier; redundant with tenure and MonthlyCharges).
FEATURE_COLUMNS = (NUMERIC_FEATURES + PASSTHROUGH_FEATURES
                   + YES_NO_FEATURES + OTHER_CATEGORICAL_FEATURES)


def build_preprocessor() -> ColumnTransformer:
    """Encoding and scaling, identical for every model."""
    return ColumnTransformer(
        transformers=[
            # Scaler statistics come from whatever data fit() sees. Fitting the
            # pipeline on the training set only means no test-set leakage.
            ("num", StandardScaler(), NUMERIC_FEATURES),

            # Declaring ONLY ["No", "Yes"] as the known categories means a third
            # value like "No internet service" is "unknown", and with
            # handle_unknown="ignore" an unknown value is encoded as all zeros,
            # which is exactly how the dropped baseline ("No") is encoded. So the
            # consolidation happens inside the encoder, with no custom code.
            ("yes_no",
             OneHotEncoder(categories=[["No", "Yes"]] * len(YES_NO_FEATURES),
                           drop="first", handle_unknown="ignore", sparse_output=False),
             YES_NO_FEATURES),

            # The encoder learns every category at fit time, so a single customer
            # at serving time is encoded exactly like a row in a batch.
            # drop="first" keeps the same baselines as get_dummies(drop_first=True).
            # A category never seen in training is also encoded as all zeros
            # (treated as the baseline) rather than raising an error.
            ("cat",
             OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False),
             OTHER_CATEGORICAL_FEATURES),

            ("pass", "passthrough", PASSTHROUGH_FEATURES),
        ],
        remainder="drop",  # ignore any other columns (e.g. customerID)
    )


def build_classifier(model_name: str, random_state: int = 42):
    """The final step of the pipeline. Both use class_weight="balanced" so the
    comparison is fair: each is told that missing a churner costs more."""
    if model_name == CHAMPION:
        return LogisticRegression(
            class_weight="balanced", random_state=random_state, max_iter=1000
        )
    if model_name == CHALLENGER:
        # Deliberately modest settings (shallow trees, a small learning rate,
        # regularisation) because the dataset is small and boosting overfits easily.
        return HistGradientBoostingClassifier(
            class_weight="balanced", learning_rate=0.05, max_iter=150,
            max_leaf_nodes=15, min_samples_leaf=40, l2_regularization=1.0,
            random_state=random_state,
        )
    raise ValueError(
        f"Unknown model '{model_name}'. Choose one of: {', '.join(MODEL_NAMES)}"
    )


def build_pipeline(model_name: str = CHAMPION, random_state: int = 42) -> Pipeline:
    """Build the full, unfitted pipeline: encode and scale -> classify."""
    return Pipeline(steps=[
        ("preprocess", build_preprocessor()),
        ("model", build_classifier(model_name, random_state)),
    ])