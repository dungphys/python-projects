from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from feature_engineering import PhysicsFeatureEngineer

def build_pipeline(classifier):
    """Wraps any classifier in the standard preprocessing pipeline."""
    return Pipeline([
        ("features", PhysicsFeatureEngineer()),
        ("scaler",   StandardScaler()),
        ("clf",      classifier),
    ])
    
def build_random_forest(**kwargs):
    defaults = dict(n_estimators=200, max_depth=15, class_weight="balanced",
                    n_jobs=-1, random_state=42)
    return build_pipeline(RandomForestClassifier(**{**defaults, **kwargs}))

    
def build_xgboost(**kwargs):
    defaults = dict(random_state=42)
    return build_pipeline(XGBClassifier(**{**defaults, **kwargs}))

def build_neural_network(**kwargs):
    defaults = dict(hidden_layer_sizes=(128, 64, 32), activation="relu",
                    solver="adam", max_iter=300, early_stopping=True,
                    validation_fraction=0.1, random_state=42)
    return build_pipeline(MLPClassifier(**{**defaults, **kwargs}))

# ── Hyperparameter search spaces ─────────────────────────────────────────────

RF_PARAM_GRID = {
    "clf__n_estimators":     [100, 200, 300],
    "clf__max_depth":        [8, 12, 15, None],
    "clf__min_samples_split":[2, 4, 8],
    "clf__min_samples_leaf": [1, 2, 4],
    "clf__max_features":     ["sqrt", "log2", 0.5],
}

RF_PARAM_DIST = {
    "clf__n_estimators":     [50, 100, 200, 300, 500],
    "clf__max_depth":        [5, 8, 10, 12, 15, 20, None],
    "clf__min_samples_split":[2, 4, 6, 8, 10],
    "clf__min_samples_leaf": [1, 2, 3, 4],
    "clf__max_features":     ["sqrt", "log2", 0.3, 0.5, 0.7],
    "clf__class_weight":     ["balanced", None],
}

XGB_PARAM_DIST = {
    "clf__n_estimators":        [100, 200, 300],
    "clf__learning_rate":       [0.01, 0.05, 0.1, 0.2],
    "clf__max_depth":           [3, 4, 5, 6, 8],
    "clf__subsample":           [0.7, 0.8, 0.9, 1.0],
    "clf__colsample_bytree":    [0.7, 0.9, 1.0],
}

MLP_PARAM_DIST = {
    "clf__hidden_layer_sizes": [
        (64,), (128,), (256,),
        (128, 64), (256, 128), (128, 64, 32), (256, 128, 64),
    ],
    "clf__activation":    ["relu", "tanh"],
    "clf__alpha":         [1e-5, 1e-4, 1e-3, 1e-2],
    "clf__learning_rate": ["constant", "adaptive"],
}