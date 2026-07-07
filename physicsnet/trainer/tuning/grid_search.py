"""
GridSearchCV - exhaustive search over a discrete parameter grid.
Best for: small grids, when needing guaranteed coverage.
Cost: O(n_params * n_folds) fits.
"""

import mlflow
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import classification_report
import numpy as np
import pandas as pd
    

def run_grid_search(
    pipeline,
    param_grid: dict,
    X_train, y_train,
    X_test,  y_test,
    class_names: list,
    n_folds: int = 5,
    scoring: str = "f1_weighted",
    run_name: str = "grid-search",
) -> tuple:
    """
    Runs GridSearchCV and logs all results to the active MLflow run.

    Returns:
        best_estimator, best_params, cv_results_df
    """
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

    searcher = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=cv,
        scoring=scoring,
        n_jobs=-1,
        verbose=1,
        refit=True,      # refit best model on full training set
        return_train_score=True,
    )

    print(f"\n[GridSearch] Fitting {searcher.get_n_splits()} folds × "
          f"{len(searcher.param_grid)} param combos...")
    searcher.fit(X_train, y_train)

    best   = searcher.best_estimator_
    params = searcher.best_params_
    score  = searcher.best_score_

    # Evaluate on held-out test set
    y_pred = best.predict(X_test)
    report = classification_report(
        y_test, y_pred, target_names=class_names, output_dict=True
    )

    # Log to MLflow
    mlflow.log_param("tuner",            "GridSearchCV")
    mlflow.log_param("n_folds",          n_folds)
    mlflow.log_param("scoring",          scoring)
    mlflow.log_param("n_candidates",     len(searcher.cv_results_["params"]))
    mlflow.log_params({f"best_{k}": v for k, v in params.items()})

    mlflow.log_metric("cv_best_score",   score)
    mlflow.log_metric("test_accuracy",   report["accuracy"])
    mlflow.log_metric("test_f1_weighted",report["weighted avg"]["f1-score"])
    mlflow.log_metric("test_precision",  report["weighted avg"]["precision"])
    mlflow.log_metric("test_recall",     report["weighted avg"]["recall"])

    for cls in class_names:
        mlflow.log_metric(f"f1_{cls}", report[cls]["f1-score"])

    print(f"\n[GridSearch] Best CV {scoring}: {score:.4f}")
    print(f"[GridSearch] Test accuracy:      {report['accuracy']:.4f}")
    print(f"[GridSearch] Best params:        {params}")

    cv_df = pd.DataFrame(searcher.cv_results_).sort_values(
        "mean_test_score", ascending=False
    )

    return best, params, cv_df