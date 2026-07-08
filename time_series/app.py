"""
Time-Series Analysis & Forecasting Studio
==========================================
A Streamlit app that takes any CSV / TSV / XLS(X) file containing one or more
time series, lets you explore it, and forecasts it with SARIMAX, Prophet,
and/or XGBoost side by side.
"""
import io
import numpy as np
import pandas as pd
import streamlit as st

from modules import data_loader as dl
from modules import eda
from modules import forecasting as fc
from modules import metrics as mt
from modules import anomaly as an
from modules import multiseasonal as ms

st.set_page_config(page_title="Time-Series Forecasting Studio", layout="wide", page_icon="📈")

# --------------------------------------------------------------------------
# Session state helpers
# --------------------------------------------------------------------------
if "raw_df" not in st.session_state:
    st.session_state.raw_df = None
if "indexed_df" not in st.session_state:
    st.session_state.indexed_df = None
if "results" not in st.session_state:
    st.session_state.results = {}
if "anomaly_results" not in st.session_state:
    st.session_state.anomaly_results = {}

st.title("📈 Time-Series Analysis & Forecasting Studio")
st.caption("Upload any CSV, TSV, or Excel file with one or more time series, explore it, forecast with SARIMAX/Prophet/XGBoost, and detect multivariate anomalies.")

tab_upload, tab_eda, tab_forecast, tab_compare, tab_anomaly = st.tabs(
    ["1 · Upload & Setup", "2 · Explore", "3 · Forecast", "4 · Compare & Export", "5 · Anomaly Detection"]
)

# ==========================================================================
# TAB 1 — Upload & setup
# ==========================================================================
with tab_upload:
    st.subheader("Upload your data")
    uploaded = st.file_uploader("CSV, TSV, or Excel (.xlsx/.xls)", type=["csv", "tsv", "xlsx", "xls"])

    if uploaded is not None:
        try:
            raw = dl.load_file(uploaded)
            st.session_state.raw_df = raw
        except Exception as e:
            st.error(f"Could not read file: {e}")

    if st.session_state.raw_df is not None:
        raw = st.session_state.raw_df
        st.success(f"Loaded {raw.shape[0]:,} rows × {raw.shape[1]} columns.")
        st.dataframe(raw.head(20), use_container_width=True)

        st.markdown("---")
        st.subheader("Configure the time index")

        cols = list(raw.columns)
        guess = dl.guess_datetime_column(raw)
        date_col = st.selectbox(
            "Which column is the date/timestamp?",
            options=cols,
            index=cols.index(guess) if guess in cols else 0,
        )

        indexed, n_bad = dl.build_time_index(raw, date_col)
        if n_bad:
            st.warning(f"{n_bad} row(s) had an unparseable date and were dropped.")

        num_cols = dl.numeric_columns(indexed)
        if not num_cols:
            st.error("No numeric columns found to forecast. Please check your file.")
        else:
            target_col = st.selectbox("Primary series to analyze / forecast", options=num_cols)

            inferred_freq = dl.infer_frequency(indexed.index)
            freq_label = dl.FREQ_LABELS.get(inferred_freq, inferred_freq or "Unknown")
            st.info(f"Detected frequency: **{freq_label}** ({inferred_freq or 'n/a'}) · Range: {indexed.index.min().date()} → {indexed.index.max().date()}")

            freq_override = st.selectbox(
                "Frequency to use (override if the detection looks wrong)",
                options=list(dl.FREQ_LABELS.keys()),
                index=list(dl.FREQ_LABELS.keys()).index(inferred_freq) if inferred_freq in dl.FREQ_LABELS else 0,
                format_func=lambda f: f"{f} — {dl.FREQ_LABELS[f]}",
            )

            missing_method = st.selectbox(
                "Missing value handling",
                options=["None (drop)", "Forward fill", "Backward fill", "Linear interpolation", "Mean"],
            )

            if st.button("Apply configuration", type="primary"):
                series = indexed[target_col]
                # Reindex to a regular calendar at the chosen frequency, then fill.
                full_index = pd.date_range(series.index.min(), series.index.max(), freq=freq_override)
                series = series.reindex(full_index)
                series = eda.fill_missing(series, missing_method)

                clean_df = indexed.reindex(full_index)
                clean_df[target_col] = series

                st.session_state.indexed_df = clean_df
                st.session_state.target_col = target_col
                st.session_state.freq = freq_override
                st.session_state.num_cols = num_cols
                st.session_state.results = {}
                st.success("Configuration applied — head to the **Explore** or **Forecast** tab.")

# ==========================================================================
# TAB 2 — EDA
# ==========================================================================
with tab_eda:
    if st.session_state.indexed_df is None:
        st.info("Upload and configure a dataset in Tab 1 first.")
    else:
        df = st.session_state.indexed_df
        target_col = st.session_state.target_col
        series = df[target_col]

        st.subheader(f"Overview — `{target_col}`")
        compare_cols = st.multiselect(
            "Columns to plot", options=st.session_state.num_cols, default=[target_col]
        )
        if compare_cols:
            st.plotly_chart(eda.line_plot(df, compare_cols), use_container_width=True)

        c1, c2 = st.columns([1, 2])
        with c1:
            st.markdown("**Summary statistics**")
            st.dataframe(eda.summary_stats(series), use_container_width=True)
        with c2:
            st.markdown("**Stationarity tests (ADF & KPSS)**")
            tests = eda.stationarity_tests(series)
            for name, res in tests.items():
                if "error" in res:
                    st.write(f"{name}: could not compute ({res['error']})")
                else:
                    st.write(
                        f"**{name}**: statistic={res['statistic']:.3f}, "
                        f"p-value={res['p_value']:.4f} → {res['interpretation']}"
                    )

        st.markdown("---")
        st.subheader("Seasonal decomposition")
        default_period = {"D": 7, "B": 5, "W": 52, "MS": 12, "M": 12, "Q": 4, "H": 24}.get(st.session_state.freq, 12)
        period = st.number_input("Seasonal period (cycle length in data points)", min_value=2, value=default_period)
        decomp_model = st.radio("Decomposition model", ["additive", "multiplicative"], horizontal=True)
        if len(series.dropna()) > 2 * period:
            try:
                st.plotly_chart(eda.decomposition_plot(series, period, decomp_model), use_container_width=True)
            except Exception as e:
                st.warning(f"Decomposition failed: {e}")
        else:
            st.warning("Not enough data points for this seasonal period.")

        st.markdown("---")
        st.subheader("Autocorrelation")
        lags = st.slider("Lags", 5, 100, 40)
        st.plotly_chart(eda.acf_pacf_plot(series, lags), use_container_width=True)

        st.markdown("---")
        st.subheader("Multi-seasonal analysis (optional)")
        st.caption(
            "For series with more than one seasonal cycle (e.g. daily *and* weekly patterns "
            "in hourly data, or weekly *and* yearly patterns in daily data), decompose them "
            "all at once with MSTL instead of picking a single period above."
        )
        enable_mstl = st.checkbox("Enable multi-seasonal decomposition (MSTL)")
        if enable_mstl:
            periods_text = st.text_input(
                "Seasonal periods, in number of time steps (comma-separated)",
                value=f"{default_period},{default_period * 4}",
                help="E.g. '7,365' for weekly + yearly seasonality on daily data, or '24,168' for daily + weekly on hourly data.",
            )
            periods = ms.parse_periods(periods_text)
            min_len_needed = 2 * (max(periods) if periods else 0)
            if not periods:
                st.warning("Enter at least one period greater than 1.")
            elif len(series.dropna()) < min_len_needed:
                st.warning(
                    f"Need at least {int(min_len_needed)} data points for the largest period "
                    f"({max(periods):.0f}); this series has {len(series.dropna())}."
                )
            else:
                try:
                    result, used_periods = ms.mstl_decompose(series, periods)
                    st.plotly_chart(ms.mstl_plot(result, used_periods), use_container_width=True)
                    st.caption(
                        "Each seasonal panel isolates one cycle length. These period values are "
                        "also good starting points for the 'extra seasonal periods' settings in "
                        "the Forecast tab (SARIMAX and XGBoost)."
                    )
                except Exception as e:
                    st.error(f"MSTL decomposition failed: {e}")

# ==========================================================================
# TAB 3 — Forecast
# ==========================================================================
with tab_forecast:
    if st.session_state.indexed_df is None:
        st.info("Upload and configure a dataset in Tab 1 first.")
    else:
        df = st.session_state.indexed_df
        target_col = st.session_state.target_col
        freq = st.session_state.freq
        series = df[target_col].dropna()

        st.subheader("Train / test split & horizon")
        c1, c2 = st.columns(2)
        with c1:
            test_size = st.slider("Holdout (test) size — number of periods", 1, max(2, len(series) // 2), min(30, max(2, len(series) // 5)))
        with c2:
            horizon = st.number_input("Additional periods to forecast beyond the data (future)", min_value=0, value=0)

        train = series.iloc[: -test_size] if test_size < len(series) else series
        test = series.iloc[-test_size:] if test_size < len(series) else series.iloc[0:0]
        total_horizon = test_size + horizon

        st.caption(f"Training on {len(train)} points, validating on {len(test)} held-out points" + (f", plus forecasting {horizon} points into the future." if horizon else "."))

        model_choice = st.multiselect(
            "Models to run", ["SARIMAX", "Prophet", "XGBoost"], default=["SARIMAX", "Prophet", "XGBoost"]
        )

        with st.expander("SARIMAX settings", expanded="SARIMAX" in model_choice):
            suggestion = fc.auto_order_suggestion(train)
            st.caption(f"ADF-based suggestion: d={suggestion['suggested_d']}, D={suggestion['suggested_D']}")
            sc1, sc2, sc3 = st.columns(3)
            p = sc1.number_input("p", 0, 5, 1)
            d = sc2.number_input("d", 0, 2, suggestion["suggested_d"])
            q = sc3.number_input("q", 0, 5, 1)
            st.caption("Seasonal order (set seasonal period to 0 to disable)")
            sc4, sc5, sc6, sc7 = st.columns(4)
            P = sc4.number_input("P", 0, 3, 0)
            D_ = sc5.number_input("D", 0, 2, 0)
            Q = sc6.number_input("Q", 0, 3, 0)
            m = sc7.number_input("Seasonal period (m)", 0, 366, 0)

            st.markdown("**Multi-seasonal (optional)**")
            st.caption(
                "Add extra seasonal cycles beyond the one above as Fourier terms — "
                "e.g. set m=7 for weekly seasonality, then add 365 here for a yearly cycle too."
            )
            sarimax_extra_enabled = st.checkbox("Add extra seasonal periods to SARIMAX", key="sarimax_extra_enabled")
            sarimax_extra_periods_text = st.text_input(
                "Extra periods, in steps (comma-separated)", value="365", disabled=not sarimax_extra_enabled,
                key="sarimax_extra_periods",
            )
            sarimax_fourier_order = st.slider(
                "Fourier order per extra period", 1, 10, 3, disabled=not sarimax_extra_enabled, key="sarimax_fourier_order"
            )

        with st.expander("Prophet settings", expanded="Prophet" in model_choice):
            pc1, pc2 = st.columns(2)
            seasonality_mode = pc1.selectbox("Seasonality mode", ["additive", "multiplicative"])
            cps = pc2.slider("Changepoint prior scale (flexibility)", 0.001, 0.5, 0.05, step=0.001)

            st.markdown("**Multi-seasonal (optional)**")
            st.caption(
                "Prophet already models yearly/weekly/daily seasonality automatically. Add custom "
                "cycles here for anything else — e.g. a 30-day billing cycle or a custom pay period."
            )
            prophet_extra_enabled = st.checkbox("Add custom seasonalities to Prophet", key="prophet_extra_enabled")
            prophet_extra_periods_text = st.text_input(
                "Extra periods, in days (comma-separated)", value="30", disabled=not prophet_extra_enabled,
                key="prophet_extra_periods",
            )
            prophet_fourier_order = st.slider(
                "Fourier order per extra period", 1, 15, 5, disabled=not prophet_extra_enabled, key="prophet_fourier_order"
            )

        with st.expander("XGBoost settings", expanded="XGBoost" in model_choice):
            xc1, xc2, xc3 = st.columns(3)
            n_estimators = xc1.number_input("Trees", 50, 1000, 300, step=50)
            max_depth = xc2.number_input("Max depth", 2, 12, 4)
            lr = xc3.number_input("Learning rate", 0.001, 1.0, 0.05, step=0.01)
            lags_str = st.text_input("Lag features (comma-separated)", "1,2,3,7,14")
            lags = [int(x) for x in lags_str.split(",") if x.strip().isdigit()]

            st.markdown("**Multi-seasonal (optional)**")
            st.caption(
                "Calendar features (day-of-week, month, etc.) already give XGBoost some seasonal "
                "signal. Add explicit Fourier terms for cycles that don't line up with calendar "
                "units, e.g. a 13-day cycle."
            )
            xgb_extra_enabled = st.checkbox("Add extra seasonal periods to XGBoost", key="xgb_extra_enabled")
            xgb_extra_periods_text = st.text_input(
                "Extra periods, in steps (comma-separated)", value="365", disabled=not xgb_extra_enabled,
                key="xgb_extra_periods",
            )
            xgb_fourier_order = st.slider(
                "Fourier order per extra period", 1, 10, 3, disabled=not xgb_extra_enabled, key="xgb_fourier_order"
            )

        run = st.button("Run forecast", type="primary")

        if run:
            results = {}
            progress = st.progress(0.0, text="Starting...")
            n_models = max(len(model_choice), 1)

            if "SARIMAX" in model_choice:
                progress.progress(0.05, text="Fitting SARIMAX...")
                try:
                    seasonal_order = (P, D_, Q, m) if m and m > 1 else (0, 0, 0, 0)
                    extra_periods_sarimax = None
                    if sarimax_extra_enabled:
                        parsed = ms.parse_periods(sarimax_extra_periods_text)
                        extra_periods_sarimax = [(p, sarimax_fourier_order) for p in parsed]
                    results["SARIMAX"] = fc.run_sarimax(
                        train, total_horizon, order=(p, d, q), seasonal_order=seasonal_order, freq=freq,
                        extra_seasonal_periods=extra_periods_sarimax,
                    )
                except Exception as e:
                    st.error(f"SARIMAX failed: {e}")

            if "Prophet" in model_choice:
                progress.progress(0.4, text="Fitting Prophet...")
                try:
                    custom_seasonalities = None
                    if prophet_extra_enabled:
                        parsed = ms.parse_periods(prophet_extra_periods_text)
                        custom_seasonalities = [
                            {"name": f"custom_{int(round(pval))}", "period_days": pval, "fourier_order": prophet_fourier_order}
                            for pval in parsed
                        ]
                    results["Prophet"] = fc.run_prophet(
                        train, total_horizon, freq=freq, seasonality_mode=seasonality_mode, changepoint_prior_scale=cps,
                        custom_seasonalities=custom_seasonalities,
                    )
                except Exception as e:
                    st.error(f"Prophet failed: {e}")

            if "XGBoost" in model_choice:
                progress.progress(0.7, text="Fitting XGBoost...")
                try:
                    extra_periods_xgb = None
                    if xgb_extra_enabled:
                        parsed = ms.parse_periods(xgb_extra_periods_text)
                        extra_periods_xgb = [(p, xgb_fourier_order) for p in parsed]
                    results["XGBoost"] = fc.run_xgboost(
                        train, total_horizon, freq=freq, lags=lags or [1], n_estimators=n_estimators, max_depth=max_depth,
                        learning_rate=lr, extra_seasonal_periods=extra_periods_xgb,
                    )
                except Exception as e:
                    st.error(f"XGBoost failed: {e}")

            progress.progress(1.0, text="Done.")
            st.session_state.results = results
            st.session_state.train, st.session_state.test = train, test
            st.session_state.test_size, st.session_state.horizon = test_size, horizon

        if st.session_state.results:
            st.markdown("---")
            st.subheader("Forecast vs. actuals")
            results = st.session_state.results
            train_s = st.session_state.train
            test_s = st.session_state.test

            import plotly.graph_objects as go
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=train_s.index, y=train_s.values, mode="lines", name="Train (actual)", line=dict(color="#111827")))
            if len(test_s):
                fig.add_trace(go.Scatter(x=test_s.index, y=test_s.values, mode="lines", name="Test (actual)", line=dict(color="#111827", dash="dot")))

            colors = {"SARIMAX": "#2563eb", "Prophet": "#16a34a", "XGBoost": "#dc2626"}
            for name, res in results.items():
                fig.add_trace(go.Scatter(x=res["forecast"].index, y=res["forecast"].values, mode="lines", name=f"{name} forecast", line=dict(color=colors.get(name))))
                if res.get("lower") is not None and res.get("upper") is not None:
                    fig.add_trace(go.Scatter(x=res["forecast"].index, y=res["upper"].values, mode="lines", line=dict(width=0), showlegend=False))
                    fig.add_trace(go.Scatter(x=res["forecast"].index, y=res["lower"].values, mode="lines", line=dict(width=0), fill="tonexty",
                                              fillcolor="rgba(0,0,0,0.08)", showlegend=False))

            fig.update_layout(template="plotly_white", hovermode="x unified", height=550, title="Actual vs. Forecast")
            st.plotly_chart(fig, use_container_width=True)

            st.markdown("**Model summaries**")
            for name, res in results.items():
                st.write(f"- **{name}**: {res['summary']}")

# ==========================================================================
# TAB 4 — Compare & export
# ==========================================================================
with tab_compare:
    if not st.session_state.get("results"):
        st.info("Run a forecast in Tab 3 first.")
    else:
        results = st.session_state.results
        test_s = st.session_state.test

        st.subheader("Accuracy on the held-out test period")
        if len(test_s):
            table = mt.metrics_table(results, test_s)
            st.dataframe(table.style.format({"MAE": "{:.3f}", "RMSE": "{:.3f}", "MAPE": "{:.2f}%", "R2": "{:.3f}"}), use_container_width=True)
            best = table["RMSE"].idxmin()
            st.success(f"Lowest RMSE on the test set: **{best}**")
        else:
            st.info("No holdout period was set, so accuracy metrics aren't available — only pure future forecasts were produced.")

        if "XGBoost" in results and "feature_importance" in results["XGBoost"]:
            st.markdown("---")
            st.subheader("XGBoost feature importance")
            st.bar_chart(results["XGBoost"]["feature_importance"])

        st.markdown("---")
        st.subheader("Export forecasts")
        export_df = pd.DataFrame({name: res["forecast"] for name, res in results.items()})
        export_df.index.name = "date"
        st.dataframe(export_df, use_container_width=True)
        csv_bytes = export_df.to_csv().encode("utf-8")
        st.download_button("Download forecasts as CSV", data=csv_bytes, file_name="forecasts.csv", mime="text/csv")


def _hex_to_rgba(hexcolor, alpha):
    hexcolor = hexcolor.lstrip("#")
    r, g, b = tuple(int(hexcolor[i:i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{alpha})"


# ==========================================================================
# TAB 5 — Multivariate anomaly detection
# ==========================================================================
with tab_anomaly:
    if st.session_state.indexed_df is None:
        st.info("Upload and configure a dataset in Tab 1 first.")
    else:
        df = st.session_state.indexed_df
        num_cols = st.session_state.num_cols

        st.subheader("Choose the series to analyze jointly")
        st.caption(
            "Anomaly detection here is *multivariate*: a point is flagged if the combination of "
            "values across the selected columns looks unusual, even if no single column looks "
            "extreme on its own."
        )
        default_cols = num_cols if len(num_cols) <= 6 else num_cols[:6]
        anomaly_cols = st.multiselect("Columns to include", options=num_cols, default=default_cols)

        c1, c2 = st.columns(2)
        with c1:
            contamination = st.slider(
                "Expected anomaly rate", 0.01, 0.25, 0.05, step=0.01,
                help="Rough prior on what fraction of points are anomalous. Higher = more points flagged.",
            )
        with c2:
            methods = st.multiselect(
                "Detection methods",
                ["Isolation Forest", "PCA Reconstruction Error", "Robust Covariance (Elliptic Envelope)"],
                default=["Isolation Forest", "PCA Reconstruction Error", "Robust Covariance (Elliptic Envelope)"],
            )

        with st.expander("Advanced settings"):
            ac1, ac2 = st.columns(2)
            n_estimators_if = ac1.number_input("Isolation Forest trees", 50, 1000, 200, step=50)
            n_components_pca = ac2.number_input(
                "PCA components (0 = auto)", 0, max(1, len(anomaly_cols) - 1) if anomaly_cols else 1, 0
            )
            min_votes = st.slider(
                "Ensemble: minimum methods agreeing to flag a point", 1, max(1, len(methods)), min(2, max(1, len(methods)))
            )

        run_anomaly = st.button("Run anomaly detection", type="primary")

        if run_anomaly:
            if len(anomaly_cols) < 1:
                st.error("Select at least one column.")
            elif len(anomaly_cols) < 2:
                st.warning("Only one column selected — this reduces to univariate outlier detection on that column.")

            if anomaly_cols:
                results = {}
                sub = df[anomaly_cols].dropna()
                if len(sub) < 10:
                    st.error("Not enough complete (non-missing) rows across the selected columns to run detection.")
                else:
                    if "Isolation Forest" in methods:
                        try:
                            res, _ = an.isolation_forest_scores(
                                df, anomaly_cols, contamination=contamination, n_estimators=n_estimators_if
                            )
                            results["Isolation Forest"] = res
                        except Exception as e:
                            st.error(f"Isolation Forest failed: {e}")

                    if "PCA Reconstruction Error" in methods:
                        try:
                            n_comp = n_components_pca if n_components_pca > 0 else None
                            res, _ = an.pca_reconstruction_scores(
                                df, anomaly_cols, n_components=n_comp, contamination=contamination
                            )
                            results["PCA Reconstruction Error"] = res
                        except Exception as e:
                            st.error(f"PCA reconstruction failed: {e}")

                    if "Robust Covariance (Elliptic Envelope)" in methods:
                        try:
                            res, _ = an.elliptic_envelope_scores(df, anomaly_cols, contamination=contamination)
                            results["Robust Covariance (Elliptic Envelope)"] = res
                        except Exception as e:
                            st.error(
                                f"Robust covariance failed: {e} (this method needs more rows than "
                                "columns, and can fail if columns are near-perfectly correlated)"
                            )

                    st.session_state.anomaly_results = results
                    st.session_state.anomaly_cols = anomaly_cols
                    st.session_state.anomaly_min_votes = min_votes

        if st.session_state.anomaly_results:
            results = st.session_state.anomaly_results
            anomaly_cols = st.session_state.anomaly_cols
            min_votes = st.session_state.get("anomaly_min_votes", 2)

            st.markdown("---")
            st.subheader("Anomaly scores by method")
            st.plotly_chart(an.score_plot(results), use_container_width=True)

            ensemble = an.ensemble_vote(results, min_votes=min_votes)
            anomaly_index = ensemble[ensemble["is_anomaly"]].index

            st.markdown("---")
            st.subheader(f"Series with ensemble-flagged anomalies ({len(anomaly_index)} points)")
            st.caption(f"A point is marked when at least {min_votes} of {len(results)} method(s) flag it.")
            st.plotly_chart(
                an.multivariate_overlay_plot(df, anomaly_cols, anomaly_index), use_container_width=True
            )

            st.markdown("---")
            st.subheader("Detected anomalies")
            detail = df.loc[anomaly_index, anomaly_cols].copy()
            detail["votes"] = ensemble.loc[anomaly_index, "votes"]
            for name, res in results.items():
                detail[f"{name} score"] = res.reindex(anomaly_index)["score"]
            detail = detail.sort_values("votes", ascending=False)
            st.dataframe(detail, use_container_width=True)

            csv_bytes = detail.to_csv().encode("utf-8")
            st.download_button(
                "Download detected anomalies as CSV", data=csv_bytes, file_name="anomalies.csv", mime="text/csv"
            )