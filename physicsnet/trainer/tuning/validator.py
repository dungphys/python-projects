"""
Comprehensive model validation:
- Stratified K-Fold cross-validation
- Nested cross-validation (unbiased performance estimate)
- Learning curves (bias-variance diagnosis)
- Confusion matrix & per-class metrics
- Calibration check
"""

import numpy as np
import pandas as pd
import mlflow
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tempfile, os

from sklearn.model_selection import (
    StratifiedKFold, cross_validate, learning_curve,
    cross_val_predict,
)
from sklearn.metrics import (
    confusion_matrix, ConfusionMatrixDisplay,
    classification_report, roc_auc_score,
    calibration_curve,
)


def cross_validate_model(
    pipeline,
    X, y,
    class_names: list,
    n_folds: int = 5,
    label: str = "model",
) -> dict:
    """
    Full stratified K-fold cross-validation.
    Returns mean ± std for accuracy, F1, precision, recall.
    Logs all metrics and plots to MLflow.
    """
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

    scoring = {
        "accuracy":  "accuracy",
        "f1":        "f1_weighted",
        "precision": "precision_weighted",
        "recall":    "recall_weighted",
    }

    print(f"\n[Validator] Running {n_folds}-fold CV on {label}...")
    results = cross_validate(
        pipeline, X, y,
        cv=cv,
        scoring=scoring,
        return_train_score=True,
        n_jobs=-1,
    )

    summary = {}
    for metric in ["accuracy", "f1", "precision", "recall"]:
        test_scores  = results[f"test_{metric}"]
        train_scores = results[f"train_{metric}"]
        summary[metric] = {
            "mean":  float(test_scores.mean()),
            "std":   float(test_scores.std()),
            "min":   float(test_scores.min()),
            "max":   float(test_scores.max()),
            "train_mean": float(train_scores.mean()),
        }
        mlflow.log_metric(f"cv_{metric}_mean", test_scores.mean())
        mlflow.log_metric(f"cv_{metric}_std",  test_scores.std())
        mlflow.log_metric(f"cv_{metric}_train",train_scores.mean())
        print(f"  {metric:10s}: {test_scores.mean():.4f} ± {test_scores.std():.4f}"
              f"  (train: {train_scores.mean():.4f})")

    # Out-of-fold predictions for confusion matrix
    y_oof = cross_val_predict(pipeline, X, y, cv=cv, n_jobs=-1)
    _log_confusion_matrix(y, y_oof, class_names, label)
    _log_per_class_f1(y, y_oof, class_names, label)

    return summary


def nested_cross_validate(
    pipeline,
    param_distributions: dict,
    X, y,
    class_names: list,
    outer_folds: int = 5,
    inner_folds: int = 3,
    n_iter:      int = 20,
) -> dict:
    """
    Nested cross-validation — provides an unbiased estimate of
    generalization performance when hyperparameter tuning is included.

    Outer loop:  test set  (measures generalization)
    Inner loop:  val set   (selects hyperparameters)
    """
    from sklearn.model_selection import RandomizedSearchCV

    outer_cv = StratifiedKFold(n_splits=outer_folds, shuffle=True, random_state=42)
    inner_cv = StratifiedKFold(n_splits=inner_folds, shuffle=True, random_state=0)

    clf = RandomizedSearchCV(
        pipeline, param_distributions,
        n_iter=n_iter, cv=inner_cv,
        scoring="f1_weighted", n_jobs=-1, random_state=42,
    )

    print(f"\n[Validator] Nested CV ({outer_folds}×{inner_folds})...")
    results = cross_validate(
        clf, X, y, cv=outer_cv,
        scoring={"accuracy": "accuracy", "f1": "f1_weighted"},
        return_train_score=True, n_jobs=-1, verbose=0,
    )

    nested_acc = results["test_accuracy"]
    nested_f1  = results["test_f1"]

    mlflow.log_metric("nested_cv_acc_mean", nested_acc.mean())
    mlflow.log_metric("nested_cv_acc_std",  nested_acc.std())
    mlflow.log_metric("nested_cv_f1_mean",  nested_f1.mean())
    mlflow.log_metric("nested_cv_f1_std",   nested_f1.std())

    print(f"  Nested CV Accuracy: {nested_acc.mean():.4f} ± {nested_acc.std():.4f}")
    print(f"  Nested CV F1:       {nested_f1.mean():.4f} ± {nested_f1.std():.4f}")

    return {"accuracy": nested_acc, "f1": nested_f1}


def plot_learning_curve(
    pipeline, X, y,
    label: str = "model",
    n_folds: int = 5,
) -> None:
    """
    Plots train vs validation score as a function of training size.
    Reveals underfitting / overfitting / data-hunger.
    """
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    train_sizes = np.linspace(0.1, 1.0, 10)

    print(f"\n[Validator] Computing learning curve for {label}...")
    sizes, train_scores, val_scores = learning_curve(
        pipeline, X, y,
        cv=cv,
        train_sizes=train_sizes,
        scoring="f1_weighted",
        n_jobs=-1,
    )

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(sizes, train_scores.mean(axis=1), "o-", label="Train F1", color="steelblue")
    ax.fill_between(sizes,
                    train_scores.mean(1) - train_scores.std(1),
                    train_scores.mean(1) + train_scores.std(1),
                    alpha=0.15, color="steelblue")
    ax.plot(sizes, val_scores.mean(axis=1), "s--", label="Val F1", color="tomato")
    ax.fill_between(sizes,
                    val_scores.mean(1) - val_scores.std(1),
                    val_scores.mean(1) + val_scores.std(1),
                    alpha=0.15, color="tomato")
    ax.set_xlabel("Training samples")
    ax.set_ylabel("F1 (weighted)")
    ax.set_title(f"Learning Curve — {label}")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.5)

    _save_and_log_fig(fig, f"learning_curve_{label}.png")


def _log_confusion_matrix(y_true, y_pred, class_names, label):
    cm  = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(7, 6))
    disp = ConfusionMatrixDisplay(cm, display_labels=class_names)
    disp.plot(ax=ax, colorbar=True, cmap="Blues")
    ax.set_title(f"Confusion Matrix — {label}")
    _save_and_log_fig(fig, f"confusion_matrix_{label}.png")


def _log_per_class_f1(y_true, y_pred, class_names, label):
    report = classification_report(
        y_true, y_pred, target_names=class_names, output_dict=True
    )
    f1s = {cls: report[cls]["f1-score"] for cls in class_names}
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(f1s.keys(), f1s.values(), color="steelblue")
    ax.set_ylim(0, 1)
    ax.set_ylabel("F1 Score")
    ax.set_title(f"Per-class F1 — {label}")
    for i, (k, v) in enumerate(f1s.items()):
        ax.text(i, v + 0.01, f"{v:.3f}", ha="center", fontsize=10)
    _save_and_log_fig(fig, f"per_class_f1_{label}.png")


def _save_and_log_fig(fig, filename):
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        fig.savefig(f.name, dpi=120, bbox_inches="tight")
        mlflow.log_artifact(f.name, artifact_path="plots")
    os.unlink(f.name)
    plt.close(fig)