"""
Optuna — Bayesian hyperparameter optimization via Tree-structured Parzen Estimators (TPE).
Best for: expensive models, large spaces, wants to learn from past trials.
Advantage over Grid/Random: focuses trials on promising regions automatically.
"""

import optuna
import mlflow
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import classification_report
from feature_engineering import PhysicsFeatureEngineer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

optuna.logging.set_verbosity(optuna.logging.WARNING)


def _make_rf_objective(X_train, y_train, cv):
    def objective(trial):
        params = {
            "n_estimators":     trial.suggest_int("n_estimators", 50, 500, step=50),
            "max_depth":        trial.suggest_int("max_depth", 4, 20),
            "min_samples_split":trial.suggest_int("min_samples_split", 2, 12),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 6),
            "max_features":     trial.suggest_categorical("max_features",
                                                           ["sqrt", "log2", 0.3, 0.5]),
            "class_weight":     trial.suggest_categorical("class_weight",
                                                           ["balanced", None]),
        }
        pipe = Pipeline([
            ("features", PhysicsFeatureEngineer()),
            ("scaler",   StandardScaler()),
            ("clf",      RandomForestClassifier(**params, n_jobs=-1, random_state=42)),
        ])
        scores = cross_val_score(pipe, X_train, y_train, cv=cv,
                                 scoring="f1_weighted", n_jobs=-1)
        return scores.mean()
    return objective


def _make_gb_objective(X_train, y_train, cv):
    def objective(trial):
        params = {
            "n_estimators":  trial.suggest_int("n_estimators", 50, 400, step=50),
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.3, log=True),
            "max_depth":     trial.suggest_int("max_depth", 2, 8),
            "subsample":     trial.suggest_float("subsample", 0.6, 1.0),
            "max_features":  trial.suggest_categorical("max_features",
                                                        ["sqrt", "log2", None]),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
        }
        pipe = Pipeline([
            ("features", PhysicsFeatureEngineer()),
            ("scaler",   StandardScaler()),
            ("clf",      GradientBoostingClassifier(**params, random_state=42)),
        ])
        scores = cross_val_score(pipe, X_train, y_train, cv=cv,
                                 scoring="f1_weighted", n_jobs=-1)
        return scores.mean()
    return objective


def _make_mlp_objective(X_train, y_train, cv):
    def objective(trial):
        n_layers = trial.suggest_int("n_layers", 1, 4)
        layers   = tuple(
            trial.suggest_int(f"n_units_l{i}", 32, 256, step=32)
            for i in range(n_layers)
        )
        params = {
            "hidden_layer_sizes": layers,
            "activation":         trial.suggest_categorical("activation", ["relu", "tanh"]),
            "alpha":              trial.suggest_float("alpha", 1e-6, 1e-1, log=True),
            "learning_rate_init": trial.suggest_float("lr_init", 1e-4, 1e-2, log=True),
            "batch_size":         trial.suggest_categorical("batch_size", [32, 64, 128, "auto"]),
        }
        pipe = Pipeline([
            ("features", PhysicsFeatureEngineer()),
            ("scaler",   StandardScaler()),
            ("clf",      MLPClassifier(**params, max_iter=300,
                                        early_stopping=True,
                                        validation_fraction=0.1,
                                        random_state=42)),
        ])
        scores = cross_val_score(pipe, X_train, y_train, cv=cv,
                                 scoring="f1_weighted", n_jobs=-1)
        return scores.mean()
    return objective


OBJECTIVES = {
    "random_forest":       _make_rf_objective,
    "gradient_boosting":   _make_gb_objective,
    "neural_network":      _make_mlp_objective,
}


def run_optuna_search(
    model_name:  str,
    X_train, y_train,
    X_test,  y_test,
    class_names: list,
    n_trials:    int = 60,
    n_folds:     int = 5,
    timeout:     int = None,   # seconds; None = run all trials
) -> tuple:
    """
    Runs Optuna Bayesian optimization and logs to the active MLflow run.

    Returns:
        best_pipeline, best_params, study
    """
    cv        = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    objective = OBJECTIVES[model_name](X_train, y_train, cv)

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=10),
    )

    print(f"\n[Optuna/{model_name}] Running {n_trials} trials (TPE sampler)...")
    study.optimize(objective, n_trials=n_trials, timeout=timeout, show_progress_bar=True)

    best_params = study.best_params
    best_score  = study.best_value

    # Rebuild and refit best pipeline on full training data
    from model import build_random_forest, build_gradient_boosting, build_neural_network
    builders = {
        "random_forest":     build_random_forest,
        "gradient_boosting": build_gradient_boosting,
        "neural_network":    build_neural_network,
    }
    # Strip Optuna-added keys (n_layers, n_units_lN) before passing to sklearn
    clean = {k: v for k, v in best_params.items()
             if not k.startswith("n_layers") and not k.startswith("n_units")}

    best_pipeline = builders[model_name](**clean)
    best_pipeline.fit(X_train, y_train)

    y_pred = best_pipeline.predict(X_test)
    report = classification_report(
        y_test, y_pred, target_names=class_names, output_dict=True
    )

    # Log to MLflow
    mlflow.log_param("tuner",             f"Optuna-TPE/{model_name}")
    mlflow.log_param("n_trials",          n_trials)
    mlflow.log_param("n_folds",           n_folds)
    mlflow.log_params({f"best_{k}": v for k, v in best_params.items()})

    mlflow.log_metric("cv_best_score",    best_score)
    mlflow.log_metric("test_accuracy",    report["accuracy"])
    mlflow.log_metric("test_f1_weighted", report["weighted avg"]["f1-score"])
    mlflow.log_metric("test_precision",   report["weighted avg"]["precision"])
    mlflow.log_metric("test_recall",      report["weighted avg"]["recall"])
    mlflow.log_metric("n_completed_trials", len(study.trials))

    for cls in class_names:
        mlflow.log_metric(f"f1_{cls}", report[cls]["f1-score"])

    # Log trial history as artifact
    import pandas as pd, tempfile, os
    trials_df = study.trials_dataframe()
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w") as f:
        trials_df.to_csv(f, index=False)
        mlflow.log_artifact(f.name, artifact_path="optuna_trials")
    os.unlink(f.name)

    print(f"\n[Optuna] Best CV F1:    {best_score:.4f}")
    print(f"[Optuna] Test accuracy: {report['accuracy']:.4f}")
    print(f"[Optuna] Best params:   {best_params}")

    return best_pipeline, best_params, study