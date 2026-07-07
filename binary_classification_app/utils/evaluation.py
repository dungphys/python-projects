"""
Model evaluation: metrics, confusion matrix, ROC curve, precision-recall
curve, and a leaderboard builder for comparing multiple trained models.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve, precision_recall_curve,
    average_precision_score,
)


def evaluate_model(fitted_pipeline, X_test, y_test) -> dict:
    y_pred = fitted_pipeline.predict(X_test)

    if hasattr(fitted_pipeline, "predict_proba"):
        y_proba = fitted_pipeline.predict_proba(X_test)[:, 1]
    elif hasattr(fitted_pipeline, "decision_function"):
        scores = fitted_pipeline.decision_function(X_test)
        y_proba = (scores - scores.min()) / (scores.max() - scores.min() + 1e-9)
    else:
        y_proba = y_pred.astype(float)

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_proba) if len(np.unique(y_test)) > 1 else np.nan,
        "avg_precision": average_precision_score(y_test, y_proba) if len(np.unique(y_test)) > 1 else np.nan,
    }
    cm = confusion_matrix(y_test, y_pred)
    return {"metrics": metrics, "y_pred": y_pred, "y_proba": y_proba, "confusion_matrix": cm}


def leaderboard(results: dict[str, dict]) -> pd.DataFrame:
    rows = []
    for name, res in results.items():
        m = res["metrics"]
        rows.append({
            "Model": name,
            "Accuracy": m["accuracy"],
            "Precision": m["precision"],
            "Recall": m["recall"],
            "F1": m["f1"],
            "ROC AUC": m["roc_auc"],
            "Avg Precision": m["avg_precision"],
        })
    df = pd.DataFrame(rows).sort_values("ROC AUC", ascending=False).reset_index(drop=True)
    return df


def confusion_matrix_fig(cm: np.ndarray, class_labels=("0", "1")) -> go.Figure:
    fig = go.Figure(data=go.Heatmap(
        z=cm, x=[f"Pred {c}" for c in class_labels], y=[f"Actual {c}" for c in class_labels],
        text=cm, texttemplate="%{text}", colorscale="Blues",
    ))
    fig.update_layout(title="Confusion Matrix")
    return fig


def roc_curve_fig(y_test, y_proba, model_name: str) -> go.Figure:
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    auc = roc_auc_score(y_test, y_proba)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"{model_name} (AUC={auc:.3f})"))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Random",
                              line=dict(dash="dash", color="gray")))
    fig.update_layout(title="ROC Curve", xaxis_title="False Positive Rate",
                       yaxis_title="True Positive Rate")
    return fig


def multi_roc_curve_fig(results: dict[str, dict], y_test) -> go.Figure:
    fig = go.Figure()
    for name, res in results.items():
        fpr, tpr, _ = roc_curve(y_test, res["y_proba"])
        auc = res["metrics"]["roc_auc"]
        fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"{name} (AUC={auc:.3f})"))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Random",
                              line=dict(dash="dash", color="gray")))
    fig.update_layout(title="ROC Curves — Model Comparison", xaxis_title="False Positive Rate",
                       yaxis_title="True Positive Rate")
    return fig


def precision_recall_fig(y_test, y_proba, model_name: str) -> go.Figure:
    precision, recall, _ = precision_recall_curve(y_test, y_proba)
    ap = average_precision_score(y_test, y_proba)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=recall, y=precision, mode="lines",
                              name=f"{model_name} (AP={ap:.3f})"))
    fig.update_layout(title="Precision-Recall Curve", xaxis_title="Recall", yaxis_title="Precision")
    return fig


def feature_importance_fig(importance: pd.Series, top_n: int = 20) -> go.Figure:
    top = importance.head(top_n).sort_values(ascending=True)
    fig = go.Figure(go.Bar(x=top.values, y=top.index, orientation="h"))
    fig.update_layout(title=f"Top {min(top_n, len(importance))} Feature Importances",
                       xaxis_title="Importance", yaxis_title="Feature")
    return fig
