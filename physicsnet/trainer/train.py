"""
Main training orchestrator.
Runs all three tuning strategies in separate MLflow runs,
then picks the best model overall.
"""

import os
import joblib
import mlflow
import mlflow.sklearn
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split

from data_generator import generate_particle_data, PARTICLE_CLASSES
from model import (
    build_random_forest, build_gradient_boosting, build_stacking,
    RF_PARAM_GRID, RF_PARAM_DIST, GB_PARAM_DIST, MLP_PARAM_DIST,
)
from tuning.grid_search   import run_grid_search
from tuning.random_search import run_random_search
from tuning.optuna_search import run_optuna_search
from tuning.validator     import (
    cross_validate_model, nested_cross_validate, plot_learning_curve
)

# ── Config ───────────────────────────────────────────────────────────────────
MLFLOW_URI   = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
EXPERIMENT   = os.getenv("EXPERIMENT_NAME", "particle-classification")
MODEL_PATH   = os.getenv("MODEL_PATH", "/app/models/physicsnet.pkl")
FEATURE_COLS = ["energy", "momentum_x", "momentum_y", "momentum_z",
                "charge", "hit_0", "hit_1", "hit_2", "hit_3"]


def train():
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(EXPERIMENT)

    # ── 1. Data ──────────────────────────────────────────────────────────────
    print("Generating particle dataset...")
    df      = generate_particle_data(n_samples=20_000)
    X       = df[FEATURE_COLS].values
    y       = df["label"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    print(f"Train: {len(X_train):,}  |  Test: {len(X_test):,}")

    results = {}

    # ── 2. Grid Search on Random Forest ─────────────────────────────────────
    with mlflow.start_run(run_name="grid-search-rf"):
        mlflow.log_param("dataset_size", len(df))
        mlflow.log_param("n_features",   len(FEATURE_COLS))
        pipe = build_random_forest()
        best, params, cv_df = run_grid_search(
            pipe, RF_PARAM_GRID,
            X_train, y_train, X_test, y_test,
            class_names=PARTICLE_CLASSES,
            n_folds=5,
        )
        # Full CV validation on best model
        cv_summary = cross_validate_model(best, X_train, y_train,
                                          PARTICLE_CLASSES, label="grid-rf")
        plot_learning_curve(best, X_train, y_train, label="grid-rf")
        results["grid_rf"] = {
            "model": best, "f1": cv_summary["f1"]["mean"],
            "params": params,
        }
        cv_df.to_csv("/tmp/grid_search_results.csv", index=False)
        mlflow.log_artifact("/tmp/grid_search_results.csv",
                            artifact_path="search_results")

    # ── 3. Random Search on Gradient Boosting ───────────────────────────────
    with mlflow.start_run(run_name="random-search-gb"):
        mlflow.log_param("dataset_size", len(df))
        pipe = build_gradient_boosting()
        best, params, cv_df = run_random_search(
            pipe, GB_PARAM_DIST,
            X_train, y_train, X_test, y_test,
            class_names=PARTICLE_CLASSES,
            n_iter=40, n_folds=5,
        )
        cv_summary = cross_validate_model(best, X_train, y_train,
                                          PARTICLE_CLASSES, label="random-gb")
        plot_learning_curve(best, X_train, y_train, label="random-gb")
        results["random_gb"] = {
            "model": best, "f1": cv_summary["f1"]["mean"],
            "params": params,
        }

    # ── 4. Optuna Bayesian Search on MLP ────────────────────────────────────
    with mlflow.start_run(run_name="optuna-mlp"):
        mlflow.log_param("dataset_size", len(df))
        best, params, study = run_optuna_search(
            model_name="neural_network",
            X_train=X_train, y_train=y_train,
            X_test=X_test,   y_test=y_test,
            class_names=PARTICLE_CLASSES,
            n_trials=50, n_folds=5,
        )
        cv_summary = cross_validate_model(best, X_train, y_train,
                                          PARTICLE_CLASSES, label="optuna-mlp")
        plot_learning_curve(best, X_train, y_train, label="optuna-mlp")
        results["optuna_mlp"] = {
            "model": best, "f1": cv_summary["f1"]["mean"],
            "params": params,
        }

    # ── 5. Nested CV on best candidate ──────────────────────────────────────
    best_key   = max(results, key=lambda k: results[k]["f1"])
    best_model = results[best_key]["model"]
    print(f"\n[Orchestrator] Best tuning strategy: {best_key} "
          f"(CV F1={results[best_key]['f1']:.4f})")

    with mlflow.start_run(run_name=f"nested-cv-{best_key}"):
        nested = nested_cross_validate(
            build_random_forest(), RF_PARAM_DIST,
            X_train, y_train,
            PARTICLE_CLASSES,
            outer_folds=5, inner_folds=3, n_iter=20,
        )
        mlflow.log_param("final_strategy", best_key)

    # ── 6. Final model: retrain best on all train data ───────────────────────
    print(f"\n[Orchestrator] Retraining final model on full training set...")
    best_model.fit(X_train, y_train)

    Path(MODEL_PATH).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({
        "model":      best_model,
        "features":   FEATURE_COLS,
        "classes":    PARTICLE_CLASSES,
        "strategy":   best_key,
        "best_params":results[best_key]["params"],
    }, MODEL_PATH)
    print(f"[Orchestrator] Final model saved → {MODEL_PATH}")

    return results


if __name__ == "__main__":
    train()