import pytest
import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "tuning"))

from data_generator import generate_particle_data, PARTICLE_CLASSES
from model import build_random_forest, build_gradient_boosting, RF_PARAM_DIST
from feature_engineering import PhysicsFeatureEngineer
from tuning.validator import cross_validate_model

FEATURE_COLS = ["energy", "momentum_x", "momentum_y", "momentum_z",
                "charge", "hit_0", "hit_1", "hit_2", "hit_3"]


@pytest.fixture(scope="module")
def dataset():
    df = generate_particle_data(n_samples=2000)
    X  = df[FEATURE_COLS].values
    y  = df["label"].values
    return X, y


# ── Feature engineering tests ─────────────────────────────────────────────

def test_physics_features_shape(dataset):
    X, _ = dataset
    eng  = PhysicsFeatureEngineer()
    Xt   = eng.fit_transform(X)
    assert Xt.shape[1] == X.shape[1] + 6, "Expected 6 new features"


def test_physics_features_finite(dataset):
    X, _ = dataset
    Xt   = PhysicsFeatureEngineer().fit_transform(X)
    assert np.all(np.isfinite(Xt)), "NaN/Inf in engineered features"


def test_pseudorapidity_bounded(dataset):
    X, _ = dataset
    Xt   = PhysicsFeatureEngineer().fit_transform(X)
    eta  = Xt[:, 11]   # pseudorapidity column
    assert np.all(np.abs(eta) < 20), "Pseudorapidity out of physical range"


# ── Model pipeline tests ───────────────────────────────────────────────────

def test_rf_pipeline_fits(dataset):
    X, y  = dataset
    model = build_random_forest()
    model.fit(X, y)
    preds = model.predict(X)
    assert preds.shape == (len(y),)


def test_gb_pipeline_fits(dataset):
    X, y  = dataset
    model = build_gradient_boosting()
    model.fit(X, y)
    preds = model.predict(X)
    assert preds.shape == (len(y),)


def test_all_classes_in_output(dataset):
    X, y  = dataset
    model = build_random_forest()
    model.fit(X, y)
    unique = set(model.predict(X))
    assert len(unique) == len(PARTICLE_CLASSES)


# ── Random search param space tests ───────────────────────────────────────

def test_param_dist_keys_valid():
    pipe = build_random_forest()
    pipe_params = set(pipe.get_params().keys())
    for key in RF_PARAM_DIST:
        assert key in pipe_params, f"Unknown param: {key}"


# ── Validation tests ───────────────────────────────────────────────────────

def test_cv_accuracy_above_baseline(dataset):
    """Model should significantly outperform random chance (25% for 4 classes)."""
    import mlflow
    X, y  = dataset
    model = build_random_forest()
    model.fit(X, y)
    mlflow.set_tracking_uri("file:///tmp/mlflow_test")
    mlflow.set_experiment("test")
    with mlflow.start_run():
        summary = cross_validate_model(model, X, y, PARTICLE_CLASSES,
                                       n_folds=3, label="test")
    assert summary["accuracy"]["mean"] > 0.6, "Model not better than baseline"
    assert summary["f1"]["mean"] > 0.6


def test_cv_std_reasonable(dataset):
    """CV std should be small — indicates stable model."""
    import mlflow
    X, y  = dataset
    model = build_random_forest()
    model.fit(X, y)
    mlflow.set_tracking_uri("file:///tmp/mlflow_test")
    mlflow.set_experiment("test")
    with mlflow.start_run():
        summary = cross_validate_model(model, X, y, PARTICLE_CLASSES,
                                       n_folds=3, label="test")
    assert summary["f1"]["std"] < 0.1, "High CV std: possible instability"