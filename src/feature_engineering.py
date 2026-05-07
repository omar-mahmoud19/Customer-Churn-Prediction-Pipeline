"""
feature_engineering.py
=======================
Role: Data Engineer / Preprocessing Specialist
Purpose: Reusable feature engineering transformers for the churn pipeline.
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


# ── Column registry ────────────────────────────────────────────────────────────

CATEGORICAL_COLS = [
    "gender", "education", "marital_status",
    "contract", "payment_method", "paperless_billing",
]

NUMERICAL_COLS = [
    "age", "annual_income", "dependents", "tenure",
    "monthlycharges", "totalcharges", "num_services",
    "has_phone_service", "has_internet_service", "has_online_security",
    "has_online_backup", "has_device_protection", "has_tech_support",
    "has_streaming_tv", "has_streaming_movies",
    "customer_satisfaction", "num_complaints", "num_service_calls",
    "late_payments", "avg_monthly_gb", "days_since_last_interaction",
    "credit_score", "senior_citizen",
]

DROP_COLS = ["customer_id", "signup_date"]
TARGET_COL = "ch"


# ── Custom Transformers ────────────────────────────────────────────────────────

class DateFeatureExtractor(BaseEstimator, TransformerMixin):
    """Extract year, month, and tenure-in-days from signup_date."""

    def __init__(self, date_col: str = "signup_date"):
        self.date_col = date_col

    def fit(self, X, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        if self.date_col in X.columns:
            dates = pd.to_datetime(X[self.date_col], errors="coerce")
            X["signup_year"] = dates.dt.year.fillna(0).astype(int)
            X["signup_month"] = dates.dt.month.fillna(0).astype(int)
            X.drop(columns=[self.date_col], inplace=True)
        return X


class DropColumnsTransformer(BaseEstimator, TransformerMixin):
    """Drop columns that carry no predictive value."""

    def __init__(self, cols_to_drop: list):
        self.cols_to_drop = cols_to_drop

    def fit(self, X, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        cols = [c for c in self.cols_to_drop if c in X.columns]
        return X.drop(columns=cols)


class ServiceBundleFeatures(BaseEstimator, TransformerMixin):
    """
    Engineer derived features from binary service flags:
    - total_services_subscribed
    - service_penetration_rate
    """

    SERVICE_FLAGS = [
        "has_phone_service", "has_internet_service", "has_online_security",
        "has_online_backup", "has_device_protection", "has_tech_support",
        "has_streaming_tv", "has_streaming_movies",
    ]

    def fit(self, X, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        available = [c for c in self.SERVICE_FLAGS if c in X.columns]
        if available:
            X["total_services_subscribed"] = X[available].sum(axis=1)
            X["service_penetration_rate"] = X["total_services_subscribed"] / len(available)
        return X


class FinancialRatioFeatures(BaseEstimator, TransformerMixin):
    """
    Engineer derived financial features:
    - charges_per_service
    - income_to_charge_ratio
    - late_payment_rate
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        if "monthlycharges" in X.columns and "total_services_subscribed" in X.columns:
            X["charges_per_service"] = (
                X["monthlycharges"] / (X["total_services_subscribed"] + 1)
            )
        if "annual_income" in X.columns and "monthlycharges" in X.columns:
            X["income_to_charge_ratio"] = (
                X["annual_income"] / (X["monthlycharges"] * 12 + 1)
            )
        if "late_payments" in X.columns and "tenure" in X.columns:
            X["late_payment_rate"] = X["late_payments"] / (X["tenure"] + 1)
        return X
