# preprocess.py
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

# custom feature engineering transformer
class FeatureEngineer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()

        # Example ratio features
        if "loan_amount" in X and "existing_klarna_debt" in X:
            X["loan_to_debt_ratio"] = X["loan_amount"] / (X["existing_klarna_debt"] + 1)

        if "num_confirmed_payments_3m_total" in X and "num_failed_payments_3m" in X:
            X["fail_ratio_3m"] = X["num_failed_payments_3m"] / (
                X["num_confirmed_payments_3m_total"] + 1
            )

        # Convert loan_issue_date to datetime features if available
        if "loan_issue_date" in X:
            X["loan_issue_date"] = pd.to_datetime(X["loan_issue_date"])
            X["issue_month"] = X["loan_issue_date"].dt.month
            X["issue_dayofweek"] = X["loan_issue_date"].dt.dayofweek

        return X


def build_preprocessor(numeric_features, categorical_features):
    """
    Build preprocessing pipeline.
    - numeric: impute missing + scale
    - categorical: frequency encoding (simple)
    """

    # numeric pipeline
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    # for now: leave categorical as-is (convert to string codes)
    # you can replace with target/freq encoding later
    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent"))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features)
        ],
        remainder="drop"  # drop anything not listed
    )

    # full pipeline: feature engineering + preprocessing
    full_pipeline = Pipeline(steps=[
        ("feat_eng", FeatureEngineer()),
        ("preprocess", preprocessor)
    ])

    return full_pipeline