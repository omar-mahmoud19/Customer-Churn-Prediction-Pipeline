"""
train_model.py
==============
Role: Data Scientist
Purpose: Train Logistic Regression, Random Forest, and Gradient Boosting models.
         Select the best model by ROC-AUC and persist as model.pkl + model_meta.pkl.
"""

import os
import pickle
import warnings
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, classification_report,
)

warnings.filterwarnings("ignore")

# ── Local imports ──────────────────────────────────────────────────────────────
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.feature_engineering import (
    DateFeatureExtractor, DropColumnsTransformer,
    ServiceBundleFeatures, FinancialRatioFeatures,
    CATEGORICAL_COLS, NUMERICAL_COLS, DROP_COLS, TARGET_COL,
)

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DATA_PATH   = os.path.join(BASE_DIR, "data", "raw", "customer_churn.csv")
PROC_DATA_PATH  = os.path.join(BASE_DIR, "data", "processed", "clean_data.csv")
MODEL_PATH      = os.path.join(BASE_DIR, "models", "model.pkl")
META_PATH       = os.path.join(BASE_DIR, "models", "model_meta.pkl")
PRED_PATH       = os.path.join(BASE_DIR, "data", "predicted_churn_customers.csv")


def load_data(path: str, sample_size: int = 200_000) -> pd.DataFrame:
    """Load raw data; sample for manageable training time."""
    print(f"[INFO] Loading data from {path} ...")
    df = pd.read_csv(path)
    print(f"[INFO] Full dataset: {df.shape}")
    if len(df) > sample_size:
        # Stratified sample to preserve churn ratio
        churn = df[df[TARGET_COL] == 1].sample(
            min(int(sample_size * 0.1), df[TARGET_COL].sum()), random_state=42
        )
        non_churn = df[df[TARGET_COL] == 0].sample(
            sample_size - len(churn), random_state=42
        )
        df = pd.concat([churn, non_churn]).sample(frac=1, random_state=42).reset_index(drop=True)
        print(f"[INFO] Sampled dataset: {df.shape}")
    return df


def build_preprocessor(num_cols: list, cat_cols: list) -> ColumnTransformer:
    """Build a ColumnTransformer for numerical and categorical features."""
    numerical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
    ])
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer(
        transformers=[
            ("num", numerical_pipeline, num_cols),
            ("cat", categorical_pipeline, cat_cols),
        ],
        remainder="drop",
    )


def build_full_pipeline(model, num_cols: list, cat_cols: list) -> Pipeline:
    """Wrap feature engineering + preprocessing + model into one Pipeline."""
    preprocessor = build_preprocessor(num_cols, cat_cols)
    return Pipeline([
        ("date_features",      DateFeatureExtractor()),
        ("service_features",   ServiceBundleFeatures()),
        ("financial_features", FinancialRatioFeatures()),
        ("drop_cols",          DropColumnsTransformer(DROP_COLS)),
        ("preprocessor",       preprocessor),
        ("model",              model),
    ])


def evaluate_model(name: str, pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    """Run predictions and return a metrics dict."""
    y_pred  = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]
    metrics = {
        "model":     name,
        "accuracy":  round(accuracy_score(y_test, y_pred),  4),
        "precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
        "recall":    round(recall_score(y_test, y_pred,    zero_division=0), 4),
        "f1":        round(f1_score(y_test, y_pred,        zero_division=0), 4),
        "roc_auc":   round(roc_auc_score(y_test, y_proba), 4),
    }
    print(f"\n{'='*50}")
    print(f"  {name}")
    print(f"{'='*50}")
    for k, v in metrics.items():
        if k != "model":
            print(f"  {k:<12}: {v}")
    print(classification_report(y_test, y_pred, target_names=["No Churn", "Churn"]))
    return metrics


def train():
    # ── Load & split ──────────────────────────────────────────────────────────
    df = load_data(RAW_DATA_PATH)
    X  = df.drop(columns=[TARGET_COL])
    y  = df[TARGET_COL]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # ── Derive extended column lists after feature engineering ────────────────
    extra_num = [
        "signup_year", "signup_month",
        "total_services_subscribed", "service_penetration_rate",
        "charges_per_service", "income_to_charge_ratio", "late_payment_rate",
    ]
    num_cols = [c for c in NUMERICAL_COLS if c != TARGET_COL] + extra_num
    cat_cols = CATEGORICAL_COLS

    # ── Define candidate models ───────────────────────────────────────────────
    candidates = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=42
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=100, class_weight="balanced",
            n_jobs=-1, random_state=42
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=100, learning_rate=0.1,
            max_depth=4, random_state=42
        ),
    }

    results  = []
    pipelines = {}

    for name, model in candidates.items():
        print(f"\n[INFO] Training {name} ...")
        pipe = build_full_pipeline(model, num_cols, cat_cols)
        pipe.fit(X_train, y_train)
        pipelines[name] = pipe
        metrics = evaluate_model(name, pipe, X_test, y_test)
        results.append(metrics)

    # ── Select best model ─────────────────────────────────────────────────────
    results_df  = pd.DataFrame(results).sort_values("roc_auc", ascending=False)
    best_name   = results_df.iloc[0]["model"]
    best_pipeline = pipelines[best_name]

    print(f"\n[INFO] Best model selected: {best_name}  (ROC-AUC={results_df.iloc[0]['roc_auc']})")

    # ── Save model ────────────────────────────────────────────────────────────
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(best_pipeline, f)
    print(f"[INFO] Model saved → {MODEL_PATH}")

    meta = {
        "best_model_name": best_name,
        "all_results":     results_df.to_dict(orient="records"),
        "num_cols":        num_cols,
        "cat_cols":        cat_cols,
        "target_col":      TARGET_COL,
        "features":        list(X_train.columns),
    }
    with open(META_PATH, "wb") as f:
        pickle.dump(meta, f)
    print(f"[INFO] Metadata saved → {META_PATH}")

    # ── Save processed clean data ─────────────────────────────────────────────
    df.to_csv(PROC_DATA_PATH, index=False)
    print(f"[INFO] Processed data saved → {PROC_DATA_PATH}")

    # ── Generate predictions on full dataset ──────────────────────────────────
    print("\n[INFO] Generating predictions on full dataset ...")
    full_df = pd.read_csv(RAW_DATA_PATH)
    X_full  = full_df.drop(columns=[TARGET_COL])
    full_df["churn_probability"] = best_pipeline.predict_proba(X_full)[:, 1]
    full_df["predicted_churn"]   = best_pipeline.predict(X_full)
    predicted = full_df[full_df["predicted_churn"] == 1].copy()
    predicted.to_csv(PRED_PATH, index=False)
    print(f"[INFO] Predicted churn customers saved → {PRED_PATH}")
    print(f"[INFO] Total predicted churners: {len(predicted):,}")

    return best_pipeline, meta, results_df


if __name__ == "__main__":
    train()
