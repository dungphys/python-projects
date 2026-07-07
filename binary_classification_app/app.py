"""
ML Binary Classification App
=============================
Upload any tabular CSV with a binary target (0/1, Yes/No, True/False, etc.)
Column names are discovered at runtime -- nothing is hardcoded.

Run with:  streamlit run app.py
"""
import io
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split

from utils.data_utils import (
    load_csv, get_column_types, detect_binary_candidates, encode_target, basic_overview,
)
from utils.eda_utils import (
    target_balance_fig, missing_values_fig, numeric_distribution_fig,
    categorical_distribution_fig, correlation_heatmap_fig, boxplot_fig,
)
from utils.preprocessing import build_preprocessor, get_output_feature_names
from utils.model_utils import get_model_registry, train_single_model, get_feature_importance, optional_import_errors
from utils.evaluation import (
    evaluate_model, leaderboard, confusion_matrix_fig, roc_curve_fig,
    multi_roc_curve_fig, precision_recall_fig, feature_importance_fig,
)

st.set_page_config(page_title="ML Binary Classification App", layout="wide")

# ----------------------------------------------------------------------------
# Session state initialization
# ----------------------------------------------------------------------------
for key, default in [
    ("df", None), ("target_col", None), ("target_mapping", None),
    ("feature_cols", None), ("numeric_cols", None), ("categorical_cols", None),
    ("X_train", None), ("X_test", None), ("y_train", None), ("y_test", None),
    ("preprocessor", None), ("trained_models", {}), ("eval_results", {}),
    ("feature_names_out", None), ("test_size", 0.2), ("random_state", 42),
]:
    if key not in st.session_state:
        st.session_state[key] = default

st.title("🔎 ML Binary Classification App")
st.caption("Upload any CSV, pick the target, and run Data Overview → EDA → Modelling → Prediction.")

# ----------------------------------------------------------------------------
# Sidebar: data upload + target/feature selection
# ----------------------------------------------------------------------------
with st.sidebar:
    st.header("1. Upload Data")
    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

    if uploaded_file is not None:
        try:
            df = load_csv(uploaded_file)
            st.session_state.df = df
            st.success(f"Loaded {df.shape[0]:,} rows × {df.shape[1]} columns")
        except Exception as e:
            st.error(f"Could not read CSV: {e}")

    df = st.session_state.df

    if df is not None:
        st.header("2. Select Target")
        candidates = detect_binary_candidates(df)
        all_cols = list(df.columns)
        default_idx = all_cols.index(candidates[0]) if candidates else 0
        target_col = st.selectbox(
            "Binary target column",
            options=all_cols,
            index=default_idx,
            help="Columns with exactly 2 unique values are suggested first.",
        )
        st.session_state.target_col = target_col

        try:
            y_encoded, mapping = encode_target(df[target_col])
            st.session_state.target_mapping = mapping
            st.caption(f"Encoding: `{mapping}`")
            valid_target = True
        except Exception as e:
            st.error(f"Target column issue: {e}")
            valid_target = False

        if valid_target:
            st.header("3. Select Features")
            feature_candidates = [c for c in df.columns if c != target_col]
            id_like = [c for c in feature_candidates if df[c].nunique() == len(df)]
            if id_like:
                st.caption(f"⚠️ Likely ID columns detected (unique per row): {', '.join(id_like)}")
            default_features = [c for c in feature_candidates if c not in id_like]
            feature_cols = st.multiselect(
                "Feature columns to use", options=feature_candidates,
                default=default_features,
            )
            st.session_state.feature_cols = feature_cols

            st.header("4. Train / Test Split")
            test_size = st.slider("Test set size", 0.1, 0.4, 0.2, 0.05)
            random_state = st.number_input("Random seed", value=42, step=1)
            st.session_state.test_size = test_size
            st.session_state.random_state = int(random_state)

if df is None:
    st.info("👈 Upload a CSV file in the sidebar to get started.")
    st.stop()

if st.session_state.target_col is None or not st.session_state.feature_cols:
    st.warning("Select a target column and at least one feature column in the sidebar.")
    st.stop()

target_col = st.session_state.target_col
feature_cols = st.session_state.feature_cols

# ----------------------------------------------------------------------------
# Tabs
# ----------------------------------------------------------------------------
tab_overview, tab_eda, tab_model, tab_predict = st.tabs(
    ["📋 Data Overview", "📊 EDA", "🤖 Modelling", "🔮 Prediction"]
)

# ============================== DATA OVERVIEW ===============================
with tab_overview:
    st.subheader("Dataset Snapshot")
    st.dataframe(df.head(20), use_container_width=True)

    overview = basic_overview(df)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", f"{overview['n_rows']:,}")
    c2.metric("Columns", overview["n_cols"])
    c3.metric("Duplicate Rows", overview["duplicate_rows"])
    c4.metric("Memory (MB)", overview["memory_mb"])

    st.markdown("#### Column Data Types")
    dtype_df = pd.DataFrame({"dtype": overview["dtypes"]})
    dtype_df["role"] = np.where(
        dtype_df.index == target_col, "target",
        np.where(dtype_df.index.isin(feature_cols), "feature", "unused"),
    )
    st.dataframe(dtype_df, use_container_width=True)

    st.markdown("#### Missing Values")
    miss = overview["missing_by_col"]
    st.dataframe(miss[miss["missing_count"] > 0], use_container_width=True)

    st.markdown("#### Descriptive Statistics (numeric columns)")
    st.dataframe(df[feature_cols + [target_col]].describe(include=[np.number]).T,
                 use_container_width=True)

    cat_cols_all = df[feature_cols].select_dtypes(exclude=[np.number]).columns.tolist()
    if cat_cols_all:
        st.markdown("#### Descriptive Statistics (categorical columns)")
        st.dataframe(df[cat_cols_all].describe().T, use_container_width=True)

# ================================== EDA ======================================
with tab_eda:
    y_encoded, _ = encode_target(df[target_col])
    valid_mask = y_encoded.notna()
    df_eda = df.loc[valid_mask]
    y_eda = y_encoded.loc[valid_mask].astype(int)

    numeric_cols, categorical_cols = get_column_types(df_eda[feature_cols])
    st.session_state.numeric_cols = numeric_cols
    st.session_state.categorical_cols = categorical_cols

    st.subheader("Target Distribution")
    st.plotly_chart(target_balance_fig(y_eda, target_col), use_container_width=True)

    st.subheader("Missing Values")
    st.plotly_chart(missing_values_fig(df_eda[feature_cols]), use_container_width=True)

    if numeric_cols:
        st.subheader("Numeric Feature Distributions")
        num_col = st.selectbox("Choose a numeric feature", numeric_cols, key="eda_num_col")
        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(numeric_distribution_fig(df_eda, num_col, y_eda, target_col),
                             use_container_width=True)
        with c2:
            st.plotly_chart(boxplot_fig(df_eda, num_col, y_eda, target_col),
                             use_container_width=True)

        st.subheader("Correlation Heatmap")
        st.plotly_chart(correlation_heatmap_fig(df_eda, numeric_cols), use_container_width=True)

    if categorical_cols:
        st.subheader("Categorical Feature Distributions")
        cat_col = st.selectbox("Choose a categorical feature", categorical_cols, key="eda_cat_col")
        st.plotly_chart(categorical_distribution_fig(df_eda, cat_col, y_eda, target_col),
                         use_container_width=True)

# ================================ MODELLING ==================================
with tab_model:
    st.subheader("Train Models")

    errs = optional_import_errors()
    if errs:
        st.caption(f"⚠️ Not installed, skipping: {', '.join(errs.keys())} "
                    f"(run `pip install xgboost lightgbm catboost` to enable).")

    registry = get_model_registry(random_state=st.session_state.random_state)
    available_models = [name for name, spec in registry.items() if spec.available]

    col_a, col_b, col_c = st.columns(3)
    with col_a:
        models_to_run = st.multiselect(
            "Models to train", options=available_models, default=available_models,
        )
    with col_b:
        cv_folds = st.slider("CV folds", 3, 10, 5)
    with col_c:
        n_iter = st.slider("Random search iterations", 5, 50, 15)

    scoring = st.selectbox("CV scoring metric", ["roc_auc", "f1", "accuracy", "recall", "precision"], index=0)

    if st.button("🚀 Train selected models", type="primary"):
        y_encoded, _ = encode_target(df[target_col])
        valid_mask = y_encoded.notna()
        X = df.loc[valid_mask, feature_cols].copy()
        y = y_encoded.loc[valid_mask].astype(int)

        numeric_cols, categorical_cols = get_column_types(X)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=st.session_state.test_size,
            random_state=st.session_state.random_state, stratify=y,
        )
        st.session_state.X_train, st.session_state.X_test = X_train, X_test
        st.session_state.y_train, st.session_state.y_test = y_train, y_test

        preprocessor = build_preprocessor(numeric_cols, categorical_cols)
        st.session_state.preprocessor = preprocessor

        trained_models, eval_results = {}, {}
        progress = st.progress(0.0, text="Starting...")
        for i, name in enumerate(models_to_run, start=1):
            spec = registry[name]
            progress.progress((i - 1) / len(models_to_run), text=f"Training {name}...")
            try:
                fitted, best_params, cv_score = train_single_model(
                    spec, preprocessor, X_train, y_train,
                    cv_folds=cv_folds, n_iter=n_iter, scoring=scoring,
                    random_state=st.session_state.random_state,
                )
                trained_models[name] = {
                    "pipeline": fitted, "best_params": best_params, "cv_score": cv_score,
                }
                eval_results[name] = evaluate_model(fitted, X_test, y_test)
            except Exception as e:
                st.error(f"{name} failed: {e}")
            progress.progress(i / len(models_to_run), text=f"Finished {name}")
        progress.empty()

        st.session_state.trained_models = trained_models
        st.session_state.eval_results = eval_results

        try:
            fitted_preprocessor = trained_models[list(trained_models.keys())[0]]["pipeline"].named_steps["preprocess"]
            st.session_state.feature_names_out = get_output_feature_names(fitted_preprocessor)
        except Exception:
            st.session_state.feature_names_out = None

        st.success(f"Trained {len(trained_models)} model(s).")

    # ---- Results ----
    if st.session_state.eval_results:
        st.markdown("---")
        st.subheader("Model Comparison / Leaderboard")
        board = leaderboard(st.session_state.eval_results)
        st.dataframe(
            board.style.format({
                "Accuracy": "{:.3f}", "Precision": "{:.3f}", "Recall": "{:.3f}",
                "F1": "{:.3f}", "ROC AUC": "{:.3f}", "Avg Precision": "{:.3f}",
            }).background_gradient(cmap="Greens", subset=["ROC AUC"]),
            use_container_width=True,
        )

        st.plotly_chart(
            multi_roc_curve_fig(st.session_state.eval_results, st.session_state.y_test),
            use_container_width=True,
        )

        st.markdown("#### Per-Model Detail")
        selected_model = st.selectbox("Select a model", list(st.session_state.eval_results.keys()))
        res = st.session_state.eval_results[selected_model]
        info = st.session_state.trained_models[selected_model]

        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Accuracy", f"{res['metrics']['accuracy']:.3f}")
        m2.metric("Precision", f"{res['metrics']['precision']:.3f}")
        m3.metric("Recall", f"{res['metrics']['recall']:.3f}")
        m4.metric("F1", f"{res['metrics']['f1']:.3f}")
        m5.metric("ROC AUC", f"{res['metrics']['roc_auc']:.3f}")

        if info["best_params"]:
            with st.expander("Best hyperparameters (from CV search)"):
                st.json(info["best_params"])
            if info["cv_score"] is not None:
                st.caption(f"Best CV {scoring} score: {info['cv_score']:.4f}")
        else:
            st.caption("Decision Tree is fit directly (no hyperparameter tuning/CV, per spec).")

        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(confusion_matrix_fig(res["confusion_matrix"]), use_container_width=True)
        with c2:
            st.plotly_chart(
                precision_recall_fig(st.session_state.y_test, res["y_proba"], selected_model),
                use_container_width=True,
            )

        st.markdown("#### Feature Importance")
        feat_names = st.session_state.feature_names_out
        if feat_names is not None:
            importance = get_feature_importance(info["pipeline"], feat_names)
            if importance is not None:
                top_n = st.slider("Top N features", 5, min(40, len(importance)), min(20, len(importance)))
                st.plotly_chart(feature_importance_fig(importance, top_n), use_container_width=True)
                with st.expander("Full importance table"):
                    st.dataframe(importance.to_frame(), use_container_width=True)
            else:
                st.info(f"{selected_model} does not expose feature importances or coefficients.")
    else:
        st.info("Configure models above and click **Train selected models**.")

# ================================ PREDICTION =================================
with tab_predict:
    st.subheader("Generate Predictions")

    if not st.session_state.trained_models:
        st.info("Train at least one model in the **Modelling** tab first.")
    else:
        model_name = st.selectbox(
            "Model to use for prediction", list(st.session_state.trained_models.keys()),
            key="predict_model_select",
        )
        pipeline = st.session_state.trained_models[model_name]["pipeline"]

        pred_mode = st.radio("Input method", ["Upload CSV", "Manual entry"], horizontal=True)

        if pred_mode == "Upload CSV":
            new_file = st.file_uploader("Upload new data (same feature columns)", type=["csv"], key="predict_upload")
            if new_file is not None:
                try:
                    new_df = load_csv(new_file)
                    missing_cols = [c for c in feature_cols if c not in new_df.columns]
                    if missing_cols:
                        st.error(f"Uploaded file is missing required columns: {missing_cols}")
                    else:
                        X_new = new_df[feature_cols]
                        preds = pipeline.predict(X_new)
                        probs = pipeline.predict_proba(X_new)[:, 1] if hasattr(pipeline, "predict_proba") else None

                        out = new_df.copy()
                        out["prediction"] = preds
                        if probs is not None:
                            out["probability_class_1"] = probs

                        st.dataframe(out.head(50), use_container_width=True)

                        csv_bytes = out.to_csv(index=False).encode("utf-8")
                        st.download_button(
                            "⬇️ Download predictions as CSV", data=csv_bytes,
                            file_name="predictions.csv", mime="text/csv",
                        )
                except Exception as e:
                    st.error(f"Prediction failed: {e}")

        else:  # Manual entry
            st.caption("Enter values for a single record:")
            input_vals = {}
            n_cols_per_row = 3
            cols = st.columns(n_cols_per_row)
            for i, col_name in enumerate(feature_cols):
                target_col_widget = cols[i % n_cols_per_row]
                series = df[col_name]
                if pd.api.types.is_numeric_dtype(series):
                    default_val = float(series.median()) if series.notna().any() else 0.0
                    input_vals[col_name] = target_col_widget.number_input(col_name, value=default_val)
                else:
                    options = series.dropna().unique().tolist()
                    input_vals[col_name] = target_col_widget.selectbox(col_name, options=options)

            if st.button("Predict", type="primary"):
                X_new = pd.DataFrame([input_vals])
                pred = pipeline.predict(X_new)[0]
                prob = pipeline.predict_proba(X_new)[0, 1] if hasattr(pipeline, "predict_proba") else None

                mapping = st.session_state.target_mapping or {}
                inv_mapping = {v: k for k, v in mapping.items()} if mapping else {}
                label = inv_mapping.get(pred, pred)

                st.success(f"Predicted class: **{label}** (encoded = {pred})")
                if prob is not None:
                    st.metric("Predicted probability of class 1", f"{prob:.3f}")
