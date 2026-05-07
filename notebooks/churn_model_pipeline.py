"""
churn_model_pipeline.py
=======================
Role: Data Scientist + Data Engineer (joint deliverable)
Purpose: Single-script end-to-end pipeline: load -> preprocess -> train -> evaluate -> save.
         Runnable as a standalone script or imported as a module.
"""

import os
import sys
import pickle
import warnings
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score,
)

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.feature_engineering import (
    DateFeatureExtractor, DropColumnsTransformer,
    ServiceBundleFeatures, FinancialRatioFeatures,
    CATEGORICAL_COLS, NUMERICAL_COLS, DROP_COLS, TARGET_COL,
)

# ── Configuration ─────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG = {
    "raw_data": os.path.join(BASE_DIR, "data", "raw", "customer_churn.csv"),
    "model_path": os.path.join(BASE_DIR, "models", "model.pkl"),
    "meta_path": os.path.join(BASE_DIR, "models", "model_meta.pkl"),
    "sample_size": 200_000,
    "test_size": 0.2,
    "random_state": 42,
}

EXTRA_NUMERICAL = [
    "signup_year", "signup_month",
    "total_services_subscribed", "service_penetration_rate",
    "charges_per_service", "income_to_charge_ratio", "late_payment_rate",
]

MODELS = {
    "Logistic Regression": LogisticRegression(
        max_iter=1000, class_weight="balanced", random_state=42
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=100, class_weight="balanced", n_jobs=-1, random_state=42
    ),
    "Gradient Boosting": GradientBoostingClassifier(
        n_estimators=100, learning_rate=0.1, max_depth=4, random_state=42
    ),
}


# ── Pipeline builder ──────────────────────────────────────────────────────────
def build_pipeline(estimator) -> Pipeline:
    num_cols = [c for c in NUMERICAL_COLS if c != TARGET_COL] + EXTRA_NUMERICAL
    cat_cols = CATEGORICAL_COLS

    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    cat_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    preprocessor = ColumnTransformer([
        ("num", num_pipeline, num_cols),
        ("cat", cat_pipeline, cat_cols),
    ], remainder="drop")

    return Pipeline([
        ("date_features", DateFeatureExtractor()),
        ("service_features", ServiceBundleFeatures()),
        ("financial_features", FinancialRatioFeatures()),
        ("drop_cols", DropColumnsTransformer(DROP_COLS)),
        ("preprocessor", preprocessor),
        ("model", estimator),
    ])


# ── Entrypoint ────────────────────────────────────────────────────────────────
def run_pipeline():
    print("=" * 60)
    print("  Customer Churn Prediction Pipeline")
    print("=" * 60)

    # Load
    df = pd.read_csv(CONFIG["raw_data"])

    if len(df) > CONFIG["sample_size"]:
        churn_n = min(int(CONFIG["sample_size"] * 0.1), df[TARGET_COL].sum())

        sample = pd.concat([
            df[df[TARGET_COL] == 1].sample(churn_n, random_state=42),
            df[df[TARGET_COL] == 0].sample(CONFIG["sample_size"] - churn_n, random_state=42),
        ]).sample(frac=1, random_state=42).reset_index(drop=True)
    else:
        sample = df

    X = sample.drop(columns=[TARGET_COL])
    y = sample[TARGET_COL]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=CONFIG["test_size"],
        random_state=CONFIG["random_state"],
        stratify=y,
    )

    # Train all models and collect metrics
    best_pipeline, best_name, best_auc = None, None, -1
    all_metrics = []

    for name, estimator in MODELS.items():
        print(f"\n[->] Training {name}...")

        pipe = build_pipeline(estimator)
        pipe.fit(X_train, y_train)

        y_pred = pipe.predict(X_test)
        y_proba = pipe.predict_proba(X_test)[:, 1]

        metrics = {
            "model": name,
            "accuracy": round(accuracy_score(y_test, y_pred), 4),
            "precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
            "recall": round(recall_score(y_test, y_pred, zero_division=0), 4),
            "f1": round(f1_score(y_test, y_pred, zero_division=0), 4),
            "roc_auc": round(roc_auc_score(y_test, y_proba), 4),
        }

        all_metrics.append(metrics)

        print(f"    ROC-AUC={metrics['roc_auc']}  F1={metrics['f1']}")

        if metrics["roc_auc"] > best_auc:
            best_auc = metrics["roc_auc"]
            best_name = name
            best_pipeline = pipe

    print(f"\n[OK] Best model: {best_name}  (AUC={best_auc})")

    # Persist
    with open(CONFIG["model_path"], "wb") as f:
        pickle.dump(best_pipeline, f)

    with open(CONFIG["meta_path"], "wb") as f:
        pickle.dump({
            "best_model_name": best_name,
            "all_results": all_metrics,
            "num_cols": [c for c in NUMERICAL_COLS if c != TARGET_COL] + EXTRA_NUMERICAL,
            "cat_cols": CATEGORICAL_COLS,
            "target_col": TARGET_COL,
            "features": list(X.columns),
        }, f)

    print(f"[OK] Model saved -> {CONFIG['model_path']}")
    print(f"[OK] Meta  saved -> {CONFIG['meta_path']}")

    return best_pipeline


if __name__ == "__main__":
    run_pipeline()