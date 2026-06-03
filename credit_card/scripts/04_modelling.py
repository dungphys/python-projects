import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

import xgboost as xgb

# ------------------------------------------------------------------
# helper functions
# ------------------------------------------------------------------
def _resolve_input_path() -> Path:
    base_dir = Path(__file__).resolve().parent
    print(base_dir)
    candidates = [
        base_dir / "fe_data.csv",
        base_dir.parent / "fe_data.csv",
        base_dir / "data" / "fe_data.csv",
        base_dir.parent / "data" / "fe_data.csv",
        Path("fe_data.csv"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "Could not find cleaned_data.csv in the current project structure. "
        "Place the file in the data directory."
    )

def load_data() -> pd.DataFrame:
    p = _resolve_input_path()
    return pd.read_csv(p)


def build_preprocessor(X: pd.DataFrame,
                    ohe_cols: list[str], 
                    ord_cols: list[str]
) -> ColumnTransformer:
    #
    ohe_pipeline = Pipeline([
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    ord_pipeline = Pipeline([
        ("ordinal", OrdinalEncoder(handle_unknown="ignore")),
    ])

    preprocessor = ColumnTransformer([
        ("ohe", ohe_pipeline, ohe_cols),
        ("ord", ord_pipeline, ord_cols),
    ])

    return preprocessor


def baseline_model(X_train, y_train, X_test, y_test):
    clf = DummyClassifier(strategy="most_frequent")
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
    probs = clf.predict_proba(X_test)[:, 1] if hasattr(clf, "predict_proba") else None
    return preds, probs, clf


def train_xgb_pipeline(X_train, y_train, preprocessor):
    xgb_clf = xgb.XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=42)
    pipe = Pipeline([("preprocessor", preprocessor), ("clf", xgb_clf)])
    pipe.fit(X_train, y_train)
    return pipe


def evaluate(y_true, y_pred, y_prob=None):
    res = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }
    if y_prob is not None:
        try:
            res["roc_auc"] = roc_auc_score(y_true, y_prob)
        except Exception:
            res["roc_auc"] = None
    return res


def feature_importance_from_pipeline(pipe: Pipeline, preprocessor: ColumnTransformer):
    # extract feature names after preprocessing
    try:
        ohe = preprocessor.named_transformers_["cat"].named_steps["onehot"]
        num_cols = preprocessor.transformers_[0][2]
        cat_cols = preprocessor.transformers_[1][2]
        cat_names = ohe.get_feature_names_out(cat_cols)
        feature_names = list(num_cols) + list(cat_names)
    except Exception:
        feature_names = None

    model = pipe.named_steps["clf"]
    importances = getattr(model, "feature_importances_", None)
    if importances is None or feature_names is None:
        return None
    fi = pd.Series(importances, index=feature_names).sort_values(ascending=False)
    return fi


def main():
    df = load_data("data/fe_data.csv")
    # assume target column named 'target' or 'churn'
    if "attrition_flag" in df.columns:
        target_col = "attrition_flag"
    elif "target" in df.columns:
        target_col = "target"
    else:
        # pick last column as target
        target_col = df.columns[-1]

    X = df.drop(columns=[target_col])
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    preprocessor = build_preprocessor(X_train)

    # baseline
    baseline_preds, baseline_probs, baseline_clf = baseline_model(X_train, y_train, X_test, y_test)
    baseline_metrics = evaluate(y_test, baseline_preds, baseline_probs)
    print("baseline:", baseline_metrics)

    # xgboost with pipeline
    pipe = train_xgb_pipeline(X_train, y_train, preprocessor)
    preds = pipe.predict(X_test)
    probs = pipe.predict_proba(X_test)[:, 1]
    metrics = evaluate(y_test, preds, probs)
    print("xgboost:", metrics)

    # cross-validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(pipe, X, y, cv=cv, scoring="roc_auc")
    print("cv roc_auc:", cv_scores.mean(), cv_scores)

    # feature importance
    fi = feature_importance_from_pipeline(pipe, preprocessor)
    if fi is not None:
        print("top features:\n", fi.head(20).to_dict())
    else:
        print("feature importance unavailable")


if __name__ == "__main__":
    main()