"""
evaluate_model.py
=================
Role: Data Scientist
Purpose: Load saved model.pkl and run full evaluation suite:
         Confusion matrix, ROC curve, precision-recall curve,
         feature importance, and model comparison table.
"""

import os
import pickle
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    confusion_matrix, ConfusionMatrixDisplay,
    roc_curve, auc, precision_recall_curve,
    classification_report,
)
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.feature_engineering import TARGET_COL

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH  = os.path.join(BASE_DIR, "models", "model.pkl")
META_PATH   = os.path.join(BASE_DIR, "models", "model_meta.pkl")
DATA_PATH   = os.path.join(BASE_DIR, "data", "raw", "customer_churn.csv")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")


def load_artifacts():
    with open(MODEL_PATH, "rb") as f:
        pipeline = pickle.load(f)
    with open(META_PATH, "rb") as f:
        meta = pickle.load(f)
    return pipeline, meta


def get_test_set(sample: int = 50_000):
    df = pd.read_csv(DATA_PATH, nrows=sample)
    X  = df.drop(columns=[TARGET_COL])
    y  = df[TARGET_COL]
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    return X_test, y_test


def plot_confusion_matrix(pipeline, X_test, y_test, save_dir: str):
    y_pred = pipeline.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No Churn", "Churn"])
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title("Confusion Matrix", fontsize=14, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(save_dir, "confusion_matrix.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[INFO] Saved -> {path}")


def plot_roc_curve(pipeline, X_test, y_test, save_dir: str):
    y_proba = pipeline.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(fpr, tpr, color="#2563EB", lw=2, label=f"ROC (AUC = {roc_auc:.4f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1.2, alpha=0.6, label="Random Classifier")
    ax.fill_between(fpr, tpr, alpha=0.08, color="#2563EB")
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC Curve", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right")
    plt.tight_layout()
    path = os.path.join(save_dir, "roc_curve.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[INFO] Saved -> {path}")


def plot_precision_recall(pipeline, X_test, y_test, save_dir: str):
    y_proba = pipeline.predict_proba(X_test)[:, 1]
    precision, recall, _ = precision_recall_curve(y_test, y_proba)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(recall, precision, color="#16A34A", lw=2)
    ax.fill_between(recall, precision, alpha=0.08, color="#16A34A")
    ax.set_xlabel("Recall", fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)
    ax.set_title("Precision-Recall Curve", fontsize=14, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(save_dir, "precision_recall_curve.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[INFO] Saved -> {path}")


def plot_feature_importance(pipeline, meta: dict, save_dir: str, top_n: int = 20):
    """Extract and plot feature importances from tree-based models."""
    model = pipeline.named_steps.get("model")
    preprocessor = pipeline.named_steps.get("preprocessor")

    if not hasattr(model, "feature_importances_"):
        print("[INFO] Model does not expose feature_importances_ — skipping.")
        return

    # Get feature names from ColumnTransformer
    try:
        num_names = meta["num_cols"]
        cat_encoder = preprocessor.named_transformers_["cat"]["encoder"]
        cat_names   = list(cat_encoder.get_feature_names_out(meta["cat_cols"]))
        feature_names = num_names + cat_names
    except Exception:
        feature_names = [f"feature_{i}" for i in range(len(model.feature_importances_))]

    importances = model.feature_importances_
    n = min(len(importances), len(feature_names))
    fi_df = pd.DataFrame({
        "feature":    feature_names[:n],
        "importance": importances[:n],
    }).sort_values("importance", ascending=False).head(top_n)

    fig, ax = plt.subplots(figsize=(9, 6))
    colors = ["#2563EB" if i < 5 else "#93C5FD" for i in range(len(fi_df))]
    ax.barh(fi_df["feature"][::-1], fi_df["importance"][::-1], color=colors[::-1])
    ax.set_xlabel("Importance Score", fontsize=12)
    ax.set_title(f"Top {top_n} Feature Importances", fontsize=14, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(save_dir, "feature_importance.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[INFO] Saved -> {path}")
    return fi_df


def plot_model_comparison(meta: dict, save_dir: str):
    results_df = pd.DataFrame(meta["all_results"])
    metrics = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    x = np.arange(len(metrics))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 6))
    for i, (_, row) in enumerate(results_df.iterrows()):
        vals = [row[m] for m in metrics]
        ax.bar(x + i * width, vals, width, label=row["model"])

    ax.set_xticks(x + width)
    ax.set_xticklabels([m.replace("_", " ").title() for m in metrics])
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Score")
    ax.set_title("Model Comparison", fontsize=14, fontweight="bold")
    ax.legend()
    plt.tight_layout()
    path = os.path.join(save_dir, "model_comparison.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[INFO] Saved -> {path}")


def evaluate():
    os.makedirs(REPORTS_DIR, exist_ok=True)
    pipeline, meta = load_artifacts()
    X_test, y_test = get_test_set()

    print(f"\n[INFO] Best model: {meta['best_model_name']}")
    y_pred = pipeline.predict(X_test)
    print("\n" + classification_report(y_test, y_pred, target_names=["No Churn", "Churn"]))

    plot_confusion_matrix(pipeline, X_test, y_test, REPORTS_DIR)
    plot_roc_curve(pipeline, X_test, y_test, REPORTS_DIR)
    plot_precision_recall(pipeline, X_test, y_test, REPORTS_DIR)
    fi_df = plot_feature_importance(pipeline, meta, REPORTS_DIR)
    plot_model_comparison(meta, REPORTS_DIR)

    print("\n[INFO] All evaluation artifacts saved to reports/")
    return fi_df


if __name__ == "__main__":
    evaluate()
