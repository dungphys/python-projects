"""
Exploratory data analysis helpers: summary stats, missing-value handling,
seasonal decomposition, ACF/PACF, and stationarity testing.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller, kpss, acf, pacf


def summary_stats(series: pd.Series) -> pd.DataFrame:
    s = series.dropna()
    stats = {
        "count": len(s),
        "missing": series.isna().sum(),
        "mean": s.mean(),
        "std": s.std(),
        "min": s.min(),
        "25%": s.quantile(0.25),
        "median": s.median(),
        "75%": s.quantile(0.75),
        "max": s.max(),
    }
    return pd.DataFrame(stats, index=["value"]).T


def fill_missing(series: pd.Series, method: str) -> pd.Series:
    if method == "None (drop)":
        return series.dropna()
    if method == "Forward fill":
        return series.ffill()
    if method == "Backward fill":
        return series.bfill()
    if method == "Linear interpolation":
        return series.interpolate(method="linear")
    if method == "Mean":
        return series.fillna(series.mean())
    return series


def line_plot(df: pd.DataFrame, cols: list[str], title: str = "Time Series") -> go.Figure:
    fig = go.Figure()
    for c in cols:
        fig.add_trace(go.Scatter(x=df.index, y=df[c], mode="lines", name=c))
    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Value",
        template="plotly_white",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    return fig


def decomposition_plot(series: pd.Series, period: int, model: str = "additive") -> go.Figure:
    s = series.dropna()
    result = seasonal_decompose(s, period=period, model=model, extrapolate_trend="freq")
    fig = make_subplots(
        rows=4, cols=1, shared_xaxes=True,
        subplot_titles=("Observed", "Trend", "Seasonal", "Residual"),
        vertical_spacing=0.06,
    )
    fig.add_trace(go.Scatter(x=s.index, y=result.observed, mode="lines", name="Observed"), row=1, col=1)
    fig.add_trace(go.Scatter(x=s.index, y=result.trend, mode="lines", name="Trend"), row=2, col=1)
    fig.add_trace(go.Scatter(x=s.index, y=result.seasonal, mode="lines", name="Seasonal"), row=3, col=1)
    fig.add_trace(go.Scatter(x=s.index, y=result.resid, mode="markers", name="Residual", marker=dict(size=3)), row=4, col=1)
    fig.update_layout(template="plotly_white", showlegend=False, height=700, title="Seasonal Decomposition")
    return fig


def acf_pacf_plot(series: pd.Series, lags: int = 40) -> go.Figure:
    s = series.dropna()
    lags = min(lags, len(s) // 2 - 1) if len(s) > 4 else 1
    lags = max(lags, 1)
    acf_vals, acf_confint = acf(s, nlags=lags, alpha=0.05, fft=True)
    pacf_vals, pacf_confint = pacf(s, nlags=lags, alpha=0.05)

    fig = make_subplots(rows=1, cols=2, subplot_titles=("ACF", "PACF"))

    def _add(vals, confint, col):
        x = list(range(len(vals)))
        lower = confint[:, 0] - vals
        upper = confint[:, 1] - vals
        fig.add_trace(go.Bar(x=x, y=vals, marker_color="#2563eb", showlegend=False), row=1, col=col)
        fig.add_trace(go.Scatter(x=x, y=upper, mode="lines", line=dict(width=0), showlegend=False), row=1, col=col)
        fig.add_trace(go.Scatter(x=x, y=lower, mode="lines", line=dict(width=0), fill="tonexty",
                                  fillcolor="rgba(37,99,235,0.15)", showlegend=False), row=1, col=col)

    _add(acf_vals, acf_confint, 1)
    _add(pacf_vals, pacf_confint, 2)
    fig.update_layout(template="plotly_white", height=350, title="Autocorrelation")
    return fig


def stationarity_tests(series: pd.Series) -> dict:
    s = series.dropna()
    out = {}
    try:
        adf_stat, adf_p, *_ = adfuller(s, autolag="AIC")
        out["ADF"] = {
            "statistic": adf_stat,
            "p_value": adf_p,
            "stationary": adf_p < 0.05,
            "interpretation": "Stationary (reject unit root)" if adf_p < 0.05 else "Non-stationary (fail to reject unit root)",
        }
    except Exception as e:
        out["ADF"] = {"error": str(e)}

    try:
        kpss_stat, kpss_p, *_ = kpss(s, regression="c", nlags="auto")
        out["KPSS"] = {
            "statistic": kpss_stat,
            "p_value": kpss_p,
            "stationary": kpss_p > 0.05,
            "interpretation": "Stationary (fail to reject)" if kpss_p > 0.05 else "Non-stationary (reject null of stationarity)",
        }
    except Exception as e:
        out["KPSS"] = {"error": str(e)}
    return out
