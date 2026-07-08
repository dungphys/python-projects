"""
Multi-seasonal analysis utilities.

- MSTL decomposition: explore several seasonal periods in one series at once
  (e.g. daily + weekly + yearly cycles in sub-daily data), each as its own
  additive component alongside a shared trend and residual.
- Fourier terms: deterministic sin/cos features for a seasonal period,
  used to inject *additional* seasonalities into SARIMAX (as exogenous
  regressors -- SARIMAX's built-in seasonal order only handles one period)
  and into XGBoost (as extra engineered features). Prophet supports multiple
  seasonalities natively via `add_seasonality`, so it doesn't need this
  machinery -- see modules/forecasting.py.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from statsmodels.tsa.seasonal import MSTL


def mstl_decompose(series: pd.Series, periods: list[float], windows: list[int] | None = None):
    """Fit MSTL for one or more seasonal periods (expressed in number of
    time steps of the series, e.g. 7 for weekly seasonality on daily data,
    24 for daily seasonality on hourly data)."""
    s = series.dropna()
    periods = sorted({p for p in periods if p and p > 1})
    if not periods:
        raise ValueError("Provide at least one seasonal period greater than 1.")
    periods_int = [int(round(p)) for p in periods]
    model = MSTL(s, periods=periods_int, windows=windows)
    result = model.fit()
    return result, periods_int


def mstl_plot(result, periods: list[int], title: str = "Multi-Seasonal Decomposition (MSTL)") -> go.Figure:
    seasonal = result.seasonal
    n_seasonal = seasonal.shape[1] if isinstance(seasonal, pd.DataFrame) else 1
    rows = 2 + n_seasonal + 1  # observed, trend, one row per seasonal period, residual
    titles = ["Observed", "Trend"] + [f"Seasonal (period={p})" for p in periods] + ["Residual"]
    fig = make_subplots(rows=rows, cols=1, shared_xaxes=True, subplot_titles=titles, vertical_spacing=0.04)
    idx = result.observed.index

    fig.add_trace(go.Scatter(x=idx, y=result.observed, mode="lines", name="Observed", line=dict(color="#111827")), row=1, col=1)
    fig.add_trace(go.Scatter(x=idx, y=result.trend, mode="lines", name="Trend", line=dict(color="#2563eb")), row=2, col=1)

    if isinstance(seasonal, pd.DataFrame):
        for i, p in enumerate(periods):
            fig.add_trace(
                go.Scatter(x=idx, y=seasonal.iloc[:, i], mode="lines", name=f"Seasonal {p}", line=dict(color="#16a34a")),
                row=3 + i, col=1,
            )
    else:
        fig.add_trace(go.Scatter(x=idx, y=seasonal, mode="lines", name=f"Seasonal {periods[0]}", line=dict(color="#16a34a")), row=3, col=1)

    fig.add_trace(
        go.Scatter(x=idx, y=result.resid, mode="markers", name="Residual", marker=dict(size=3, color="#dc2626")),
        row=rows, col=1,
    )
    fig.update_layout(template="plotly_white", height=210 * rows, showlegend=False, title=title)
    return fig


def infer_step(index: pd.DatetimeIndex) -> pd.Timedelta:
    """Modal spacing between consecutive timestamps -- used as the 'unit'
    that seasonal periods (expressed in steps) are measured against."""
    deltas = index.to_series().diff().dropna()
    if deltas.empty:
        return pd.Timedelta(days=1)
    return deltas.mode().iloc[0]


def _step_count(index: pd.DatetimeIndex, anchor: pd.Timestamp, step: pd.Timedelta) -> np.ndarray:
    return ((index - anchor) / step).astype(float).values


def fourier_terms(
    index: pd.DatetimeIndex, period_steps: float, order: int, anchor: pd.Timestamp, step: pd.Timedelta
) -> pd.DataFrame:
    """Deterministic Fourier series terms for a seasonal period expressed in
    number of time steps of the series. `anchor`/`step` fix a shared time
    origin so that terms built for a training range and a future range stay
    phase-consistent."""
    t = _step_count(index, anchor, step)
    out = pd.DataFrame(index=index)
    p_label = int(round(period_steps))
    for k in range(1, order + 1):
        out[f"fourier_p{p_label}_sin{k}"] = np.sin(2 * np.pi * k * t / period_steps)
        out[f"fourier_p{p_label}_cos{k}"] = np.cos(2 * np.pi * k * t / period_steps)
    return out


def build_multi_fourier_exog(
    index: pd.DatetimeIndex, period_orders: list[tuple[float, int]], anchor: pd.Timestamp, step: pd.Timedelta
) -> pd.DataFrame:
    """Concatenate Fourier terms for several seasonal periods at once."""
    if not period_orders:
        return pd.DataFrame(index=index)
    frames = [fourier_terms(index, period, order, anchor, step) for period, order in period_orders]
    return pd.concat(frames, axis=1)


def parse_periods(text: str) -> list[float]:
    """Parse a comma-separated string of periods (ints or floats) into a list."""
    out = []
    for token in text.split(","):
        token = token.strip()
        if not token:
            continue
        try:
            out.append(float(token))
        except ValueError:
            continue
    return out