"""
Model registry, hyperparameter search spaces, and training helpers.

Every model except Decision Tree is tuned with RandomizedSearchCV under
stratified k-fold cross-validation. Decision Tree is fit directly with a
single sensible, non-tuned configuration, per spec.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy.stats import randint, uniform, loguniform

from sklearn.pipeline import Pipeline
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

# Optional gradient boosting libraries -- degrade gracefully if not installed.
_OPTIONAL_IMPORT_ERRORS: dict[str, str] = {}

try:
    from xgboost import XGBClassifier
    _HAS_XGB = True
except Exception as e:  # pragma: no cover
    _HAS_XGB = False
    _OPTIONAL_IMPORT_ERRORS["XGBoost"] = str(e)

try:
    from lightgbm import LGBMClassifier
    _HAS_LGBM = True
except Exception as e:  # pragma: no cover
    _HAS_LGBM = False
    _OPTIONAL_IMPORT_ERRORS["LightGBM"] = str(e)

try:
    from catboost import CatBoostClassifier
    _HAS_CATBOOST = True
except Exception as e:  # pragma: no cover
    _HAS_CATBOOST = False
    _OPTIONAL_IMPORT_ERRORS["CatBoost"] = str(e)


@dataclass
class ModelSpec:
    name: str
    estimator: Any
    param_distributions: dict | None  # None => no tuning (Decision Tree)
    tunable: bool = True
    available: bool = True
    note: str = ""


def get_model_registry(random_state: int = 42) -> dict[str, ModelSpec]:
    registry: dict[str, ModelSpec] = {}

    # --- Decision Tree: NOT tuned, fit with the default configuration ---
    registry["Decision Tree"] = ModelSpec(
        name="Decision Tree",
        estimator=DecisionTreeClassifier(random_state=random_state),
        param_distributions=None,
        tunable=False,
    )

    # --- Logistic Regression ---
    registry["Logistic Regression"] = ModelSpec(
        name="Logistic Regression",
        estimator=LogisticRegression(max_iter=2000, random_state=random_state),
        param_distributions={
            "model__C": loguniform(1e-3, 1e2),
            "model__penalty": ["l1", "l2"],
            "model__solver": ["liblinear", "saga"],
            "model__class_weight": [None, "balanced"],
        },
    )

    # --- Random Forest ---
    registry["Random Forest"] = ModelSpec(
        name="Random Forest",
        estimator=RandomForestClassifier(random_state=random_state, n_jobs=-1),
        param_distributions={
            "model__n_estimators": randint(100, 1000),
            "model__max_depth": [None, 4, 6, 8, 10, 14, 20],
            "model__min_samples_split": randint(2, 20),
            "model__min_samples_leaf": randint(1, 10),
            "model__max_features": ["sqrt", "log2", None],
            "model__class_weight": [None, "balanced"],
        },
    )

    # --- Gradient Boosting (sklearn) ---
    registry["Gradient Boosting"] = ModelSpec(
        name="Gradient Boosting",
        estimator=GradientBoostingClassifier(random_state=random_state),
        param_distributions={
            "model__n_estimators": randint(100, 1000),
            "model__learning_rate": loguniform(1e-3, 5e-1),
            "model__max_depth": randint(2, 6),
            "model__subsample": uniform(0.6, 0.4),
            "model__min_samples_leaf": randint(1, 10),
        },
    )

    # --- XGBoost ---
    if _HAS_XGB:
        registry["XGBoost"] = ModelSpec(
            name="XGBoost",
            estimator=XGBClassifier(
                random_state=random_state,
                eval_metric="logloss", n_jobs=-1,
            ),
            param_distributions={
                "model__n_estimators": randint(100, 1000),
                "model__max_depth": randint(2, 10),
                "model__learning_rate": loguniform(1e-3, 5e-1),
                "model__subsample": uniform(0.6, 0.4),
                "model__colsample_bytree": uniform(0.6, 0.4),
                "model__gamma": uniform(0, 5),
                "model__min_child_weight": randint(1, 10),
            },
        )
    else:
        registry["XGBoost"] = ModelSpec(
            name="XGBoost", estimator=None, param_distributions=None,
            available=False, note="xgboost is not installed",
        )

    # --- LightGBM ---
    if _HAS_LGBM:
        registry["LightGBM"] = ModelSpec(
            name="LightGBM",
            estimator=LGBMClassifier(random_state=random_state, n_jobs=-1, verbose=-1),
            param_distributions={
                "model__n_estimators": randint(100, 600),
                "model__num_leaves": randint(15, 150),
                "model__max_depth": [-1, 3, 5, 7, 9, 12],
                "model__learning_rate": loguniform(1e-3, 5e-1),
                "model__subsample": uniform(0.6, 0.4),
                "model__colsample_bytree": uniform(0.6, 0.4),
                "model__min_child_samples": randint(5, 50),
            },
        )
    else:
        registry["LightGBM"] = ModelSpec(
            name="LightGBM", estimator=None, param_distributions=None,
            available=False, note="lightgbm is not installed",
        )

    # --- CatBoost ---
    if _HAS_CATBOOST:
        registry["CatBoost"] = ModelSpec(
            name="CatBoost",
            estimator=CatBoostClassifier(random_state=random_state, verbose=0),
            param_distributions={
                "model__iterations": randint(100, 800),
                "model__depth": randint(3, 10),
                "model__learning_rate": loguniform(1e-3, 5e-1),
                "model__l2_leaf_reg": uniform(1, 9),
            },
        )
    else:
        registry["CatBoost"] = ModelSpec(
            name="CatBoost", estimator=None, param_distributions=None,
            available=False, note="catboost is not installed",
        )

    return registry


def build_full_pipeline(preprocessor, estimator) -> Pipeline:
    return Pipeline(steps=[("preprocess", preprocessor), ("model", estimator)])


def train_single_model(
    spec: ModelSpec,
    preprocessor,
    X_train, y_train,
    cv_folds: int = 5,
    n_iter: int = 20,
    scoring: str = "roc_auc",
    random_state: int = 42,
):
    """Fit one model. Returns (fitted_pipeline_or_search, best_params, cv_score)."""
    pipeline = build_full_pipeline(preprocessor, spec.estimator)

    if not spec.tunable or spec.param_distributions is None:
        # Decision Tree: plain fit, no CV/tuning.
        pipeline.fit(X_train, y_train)
        return pipeline, {}, None

    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    search = RandomizedSearchCV(
        pipeline,
        param_distributions=spec.param_distributions,
        n_iter=n_iter,
        scoring=scoring,
        cv=cv,
        random_state=random_state,
        n_jobs=-1,
        refit=True,
        error_score="raise",
    )
    search.fit(X_train, y_train)
    return search.best_estimator_, search.best_params_, search.best_score_


def get_feature_importance(fitted_pipeline: Pipeline, feature_names: list[str]):
    """Extract a normalized feature-importance Series from a fitted pipeline,
    supporting tree-based models (feature_importances_) and linear models
    (abs(coef_))."""
    import pandas as pd

    model = fitted_pipeline.named_steps["model"]

    if hasattr(model, "feature_importances_"):
        values = np.asarray(model.feature_importances_, dtype=float)
    elif hasattr(model, "coef_"):
        values = np.abs(np.asarray(model.coef_, dtype=float)).ravel()
    else:
        return None

    n = min(len(values), len(feature_names))
    series = pd.Series(values[:n], index=feature_names[:n], name="importance")
    series = series.sort_values(ascending=False)
    return series


def optional_import_errors() -> dict[str, str]:
    return dict(_OPTIONAL_IMPORT_ERRORS)
