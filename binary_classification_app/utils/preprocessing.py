"""
Builds a scikit-learn ColumnTransformer that handles arbitrary numeric and
categorical columns discovered at runtime -- no hardcoded column names.
"""
from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder


def build_preprocessor(numeric_features: list[str], categorical_features: list[str]) -> ColumnTransformer:
    numeric_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    transformers = []
    if numeric_features:
        transformers.append(("num", numeric_pipeline, numeric_features))
    if categorical_features:
        transformers.append(("cat", categorical_pipeline, categorical_features))

    preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")
    return preprocessor


def get_output_feature_names(preprocessor: ColumnTransformer) -> list[str]:
    """Retrieve the expanded feature names after fitting (post one-hot)."""
    try:
        return list(preprocessor.get_feature_names_out())
    except Exception:
        # Fallback for older sklearn / edge cases
        names = []
        for name, trans, cols in preprocessor.transformers_:
            if name == "remainder":
                continue
            if hasattr(trans, "get_feature_names_out"):
                try:
                    names.extend(trans.get_feature_names_out(cols))
                    continue
                except Exception:
                    pass
            names.extend(cols)
        return names
