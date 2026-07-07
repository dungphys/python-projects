"""
RandomizedSearchCV — random sampling over continuous/large discrete distributions.
Best for: large search spaces, quick exploration, time-constrained tuning.
Cost: exactly n_iter * n_folds fits (you control it).
"""

import mlflow
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import classification_report
import numpy as np    
import pandas as pd


def run_random_search(
    pipeline,
    param_distributions: dict,
    X_train, y_train,
    X_test,  y_test,
    class_names: list,
    n_iter:  int = 50,
    n_folds: int = 5,
    scoring: str = "f1_weighted",
    run_name: str = "random-search",
) -> tuple:
    """
    Runs RandomizedSearchCV and logs results to the active MLflow run.

    Returns:
        best_estimator, best_params, cv_results_df
    """
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

    searcher = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=param_distributions,
        n_iter=n_iter,
        cv=cv,
        scoring=scoring,
        n_jobs=-1,
        verbose=1,
        refit=True,
        return_train_score=True,
        random_state=42,
    )

    print(f"\n[RandomSearch] Fitting {n_iter} iterations × {n_folds} folds...")
    searcher.fit(X_train, y_train)

    best   = searcher.best_estimator_
    params = searcher.best_params_
    score  = searcher.best_score_

    y_pred = best.predict(X_test)
    report = classification_report(
        y_test, y_pred, target_names=class_names, output_dict=True
    )

    # Log overfitting gap: train score vs CV score
    best_idx    = searcher.best_index_
    train_score = searcher.cv_results_["mean_train_score"][best_idx]
    cv_score    = searcher.cv_results_["mean_test_score"][best_idx]
    overfit_gap = float(train_score - cv_score)

    mlflow.log_param("tuner",             "RandomizedSearchCV")
    mlflow.log_param("n_iter",            n_iter)
    mlflow.log_param("n_folds",           n_folds)
    mlflow.log_param("scoring",           scoring)
    mlflow.log_params({f"best_{k}": v for k, v in params.items()})

    mlflow.log_metric("cv_best_score",    score)
    mlflow.log_metric("train_score",      train_score)
    mlflow.log_metric("overfit_gap",      overfit_gap)
    mlflow.log_metric("test_accuracy",    report["accuracy"])
    mlflow.log_metric("test_f1_weighted", report["weighted avg"]["f1-score"])

    for cls in class_names:
        mlflow.log_metric(f"f1_{cls}", report[cls]["f1-score"])

    print(f"\n[RandomSearch] Best CV {scoring}: {score:.4f}")
    print(f"[RandomSearch] Train score:        {train_score:.4f}")
    print(f"[RandomSearch] Overfitting gap:    {overfit_gap:.4f}")
    print(f"[RandomSearch] Test accuracy:      {report['accuracy']:.4f}")

    cv_df = pd.DataFrame(searcher.cv_results_).sort_values(
        "mean_test_score", ascending=False
    )

    return best, params, cv_df