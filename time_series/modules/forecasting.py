"""
Forecasting engines: SARIMAX, Prophet, and XGBoost.

Each `run_*` function takes a training series (and optionally exogenous
regressors), a forecast horizon, and returns a common result shape:

    {
        "fitted": pd.Series,       # in-sample fitted values (train range)
        "forecast": pd.Series,     # out-of-sample forecast (test/future range)
        "lower": pd.Series | None, # lower confidence bound for forecast
        "upper": pd.Series | None, # upper confidence bound for forecast
        "model": object,           # fitted model object (for inspection)
        "summary": str,            # short text summary for display
    }

This shared shape lets the app overlay/compare models generically.
"""
from __future__ import annotations

import warnings
import numpy as np
import pandas as pd

from modules import multiseasonal as ms

warnings.filterwarnings("ignore")


# --------------------------------------------------------------------------
# SARIMAX
# --------------------------------------------------------------------------
def run_sarimax(
    train: pd.Series,
    horizon: int,
    order=(1, 1, 1),
    seasonal_order=(0, 0, 0, 0),
    exog_train: pd.DataFrame | None = None,
    exog_future: pd.DataFrame | None = None,
    freq: str | None = None,
    extra_seasonal_periods: list[tuple[float, int]] | None = None,
):
    """
    extra_seasonal_periods: optional list of (period_in_steps, fourier_order)
    tuples for seasonalities *beyond* the one already captured by
    `seasonal_order`. These are added as deterministic Fourier exogenous
    regressors -- the standard "dynamic harmonic regression" trick for
    handling more than one seasonal period in an ARIMA-family model, since
    SARIMAX's native seasonal order only supports a single period.
    """
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    future_index = _make_future_index(train.index, horizon, freq)

    if extra_seasonal_periods:
        anchor = train.index[0]
        step = ms.infer_step(train.index)
        fourier_train = ms.build_multi_fourier_exog(train.index, extra_seasonal_periods, anchor, step)
        fourier_future = ms.build_multi_fourier_exog(future_index, extra_seasonal_periods, anchor, step)
        exog_train = fourier_train if exog_train is None else pd.concat([exog_train, fourier_train], axis=1)
        exog_future = fourier_future if exog_future is None else pd.concat([exog_future, fourier_future], axis=1)

    model = SARIMAX(
        train,
        order=order,
        seasonal_order=seasonal_order,
        exog=exog_train,
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    fit = model.fit(disp=False)

    fitted = fit.fittedvalues
    pred = fit.get_forecast(steps=horizon, exog=exog_future)
    mean = pred.predicted_mean
    conf = pred.conf_int(alpha=0.05)

    mean.index = future_index
    conf.index = future_index

    summary = f"SARIMAX{order}x{seasonal_order}"
    if extra_seasonal_periods:
        periods_str = ",".join(str(int(round(p))) for p, _ in extra_seasonal_periods)
        summary += f" + Fourier(periods={periods_str})"
    summary += f" — AIC={fit.aic:.2f}, BIC={fit.bic:.2f}"

    return {
        "fitted": fitted,
        "forecast": mean,
        "lower": conf.iloc[:, 0],
        "upper": conf.iloc[:, 1],
        "model": fit,
        "summary": summary,
    }


def auto_order_suggestion(series: pd.Series, seasonal_period: int = 0) -> dict:
    """Lightweight ADF-based suggestion for the differencing order `d`."""
    from statsmodels.tsa.stattools import adfuller

    s = series.dropna()
    d = 0
    test_s = s.copy()
    for _ in range(3):
        try:
            p = adfuller(test_s, autolag="AIC")[1]
        except Exception:
            break
        if p < 0.05:
            break
        test_s = test_s.diff().dropna()
        d += 1
    D = 1 if seasonal_period and seasonal_period > 1 else 0
    return {"suggested_d": d, "suggested_D": D}


# --------------------------------------------------------------------------
# Prophet
# --------------------------------------------------------------------------
def run_prophet(
    train: pd.Series,
    horizon: int,
    freq: str = "D",
    yearly_seasonality="auto",
    weekly_seasonality="auto",
    daily_seasonality="auto",
    seasonality_mode="additive",
    changepoint_prior_scale=0.05,
    regressors_train: pd.DataFrame | None = None,
    regressors_future: pd.DataFrame | None = None,
    custom_seasonalities: list[dict] | None = None,
):
    """
    custom_seasonalities: optional list of dicts, each
    {"name": str, "period_days": float, "fourier_order": int}, registered via
    Prophet's native `add_seasonality` -- the built-in way to model more than
    one seasonal cycle (e.g. yearly + monthly, or weekly + a custom
    pay-period cycle) in the same model.
    """
    from prophet import Prophet

    df = pd.DataFrame({"ds": train.index, "y": train.values})
    m = Prophet(
        yearly_seasonality=yearly_seasonality,
        weekly_seasonality=weekly_seasonality,
        daily_seasonality=daily_seasonality,
        seasonality_mode=seasonality_mode,
        changepoint_prior_scale=changepoint_prior_scale,
    )

    if custom_seasonalities:
        for cs in custom_seasonalities:
            m.add_seasonality(
                name=cs["name"], period=cs["period_days"], fourier_order=cs.get("fourier_order", 5)
            )

    if regressors_train is not None:
        for col in regressors_train.columns:
            m.add_regressor(col)
            df[col] = regressors_train[col].values

    m.fit(df)

    future = m.make_future_dataframe(periods=horizon, freq=freq, include_history=True)
    if regressors_future is not None and regressors_train is not None:
        combined = pd.concat([regressors_train, regressors_future])
        combined = combined.reindex(future["ds"]).reset_index(drop=True)
        for col in regressors_train.columns:
            future[col] = combined[col].values

    forecast = m.predict(future)
    forecast = forecast.set_index("ds")

    fitted = forecast["yhat"].iloc[: len(train)]
    fitted.index = train.index
    future_part = forecast.iloc[len(train):]

    summary = f"Prophet ({seasonality_mode} seasonality, cps={changepoint_prior_scale})"
    if custom_seasonalities:
        names = ",".join(cs["name"] for cs in custom_seasonalities)
        summary += f" + custom seasonalities({names})"

    return {
        "fitted": fitted,
        "forecast": future_part["yhat"],
        "lower": future_part["yhat_lower"],
        "upper": future_part["yhat_upper"],
        "model": m,
        "summary": summary,
        "components": forecast,
    }


# --------------------------------------------------------------------------
# XGBoost (with lag + calendar feature engineering)
# --------------------------------------------------------------------------
def _make_features(index: pd.DatetimeIndex, series: pd.Series | None, lags: list[int], roll_windows: list[int]) -> pd.DataFrame:
    feat = pd.DataFrame(index=index)
    feat["dayofweek"] = index.dayofweek
    feat["month"] = index.month
    feat["quarter"] = index.quarter
    feat["dayofyear"] = index.dayofyear
    feat["weekofyear"] = index.isocalendar().week.astype(int)
    feat["year"] = index.year
    feat["is_month_start"] = index.is_month_start.astype(int)
    feat["is_month_end"] = index.is_month_end.astype(int)

    if series is not None:
        for lag in lags:
            feat[f"lag_{lag}"] = series.shift(lag)
        for w in roll_windows:
            feat[f"roll_mean_{w}"] = series.shift(1).rolling(w).mean()
            feat[f"roll_std_{w}"] = series.shift(1).rolling(w).std()
    return feat


def run_xgboost(
    train: pd.Series,
    horizon: int,
    freq: str,
    lags: list[int] = (1, 2, 3, 7, 14),
    roll_windows: list[int] = (7, 14),
    n_estimators: int = 300,
    max_depth: int = 4,
    learning_rate: float = 0.05,
    extra_seasonal_periods: list[tuple[float, int]] | None = None,
):
    """
    extra_seasonal_periods: optional list of (period_in_steps, fourier_order)
    tuples. Calendar features (day-of-week, month, etc.) already capture some
    seasonality implicitly, but explicit Fourier terms give the trees a
    smoother, more direct signal for periods that don't align neatly with
    calendar units (e.g. a 13-day cycle).
    """
    from xgboost import XGBRegressor

    lags = list(lags)
    roll_windows = list(roll_windows)
    full_index = train.index
    X_train = _make_features(full_index, train, lags, roll_windows)

    anchor = full_index[0]
    step = ms.infer_step(full_index)
    if extra_seasonal_periods:
        fourier_train = ms.build_multi_fourier_exog(full_index, extra_seasonal_periods, anchor, step)
        X_train = pd.concat([X_train, fourier_train], axis=1)

    y_train = train.copy()

    valid = X_train.dropna().index
    X_fit = X_train.loc[valid]
    y_fit = y_train.loc[valid]

    model = XGBRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="reg:squarederror",
        random_state=42,
    )
    model.fit(X_fit, y_fit)

    fitted = pd.Series(model.predict(X_fit), index=valid)

    # Recursive multi-step forecasting: predict one step, append it to the
    # history, recompute lag/rolling/Fourier features, predict the next step.
    history = train.copy()
    future_index = _make_future_index(full_index, horizon, freq)
    preds = []
    for ts in future_index:
        extended_index = history.index.append(pd.DatetimeIndex([ts]))
        extended_series = pd.concat([history, pd.Series([np.nan], index=[ts])])
        feat_row = _make_features(extended_index, extended_series, lags, roll_windows).iloc[[-1]]
        if extra_seasonal_periods:
            fourier_row = ms.build_multi_fourier_exog(pd.DatetimeIndex([ts]), extra_seasonal_periods, anchor, step)
            feat_row = pd.concat([feat_row.reset_index(drop=True), fourier_row.reset_index(drop=True)], axis=1)
            feat_row.index = [ts]
        feat_row = feat_row[X_fit.columns]
        yhat = model.predict(feat_row)[0]
        preds.append(yhat)
        history = pd.concat([history, pd.Series([yhat], index=[ts])])

    forecast = pd.Series(preds, index=future_index)

    importances = pd.Series(model.feature_importances_, index=X_fit.columns).sort_values(ascending=False)

    summary = f"XGBoost ({n_estimators} trees, depth={max_depth}, lr={learning_rate})"
    if extra_seasonal_periods:
        periods_str = ",".join(str(int(round(p))) for p, _ in extra_seasonal_periods)
        summary += f" + Fourier(periods={periods_str})"

    return {
        "fitted": fitted,
        "forecast": forecast,
        "lower": None,
        "upper": None,
        "model": model,
        "summary": summary,
        "feature_importance": importances,
    }


# --------------------------------------------------------------------------
# Shared helpers
# --------------------------------------------------------------------------
def _make_future_index(train_index: pd.DatetimeIndex, horizon: int, freq: str | None) -> pd.DatetimeIndex:
    if freq is None:
        freq = pd.infer_freq(train_index) or "D"
    start = train_index[-1]
    return pd.date_range(start=start, periods=horizon + 1, freq=freq)[1:]