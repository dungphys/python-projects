"""
Multivariate anomaly detection.

Three complementary detectors, all operating on a standardized feature
matrix built from a chosen set of columns (so they genuinely look at
*joint* behavior across series, not just each column independently):

  - Isolation Forest: tree-based, good for non-linear/non-Gaussian structure.
  - PCA reconstruction error: flags points that don't fit the dominant
    correlation structure between the chosen columns.
  - Robust covariance (Elliptic Envelope / Mahalanobis distance): assumes an
    elliptical (roughly Gaussian) joint distribution and flags points far
    from its center relative to the covariance structure.

Each detector returns a DataFrame indexed like the input, with a `score`
column (higher = more anomalous) and a boolean `is_anomaly` column. An
ensemble vote combines flags across whichever detectors were run.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.covariance import EllipticEnvelope
from sklearn.decomposition import PCA


def _prepare(df: pd.DataFrame, cols: list[str]):
    sub = df[cols].dropna()
    scaler = StandardScaler()
    X = scaler.fit_transform(sub.values)
    return sub, X


def isolation_forest_scores(
    df: pd.DataFrame, cols: list[str], contamination: float = 0.05,
    n_estimators: int = 200, random_state: int = 42,
):
    sub, X = _prepare(df, cols)
    model = IsolationForest(
        contamination=contamination, n_estimators=n_estimators, random_state=random_state
    )
    labels = model.fit_predict(X)  # -1 anomaly, 1 normal
    scores = -model.score_samples(X)  # flip sign so higher = more anomalous
    result = pd.DataFrame({"score": scores, "is_anomaly": labels == -1}, index=sub.index)
    return result, model


def pca_reconstruction_scores(
    df: pd.DataFrame, cols: list[str], n_components: int | None = None, contamination: float = 0.05,
):
    sub, X = _prepare(df, cols)
    max_components = X.shape[1]
    if n_components is None:
        n_components = max(1, max_components - 1) if max_components > 1 else 1
    n_components = max(1, min(n_components, max_components))

    pca = PCA(n_components=n_components)
    transformed = pca.fit_transform(X)
    reconstructed = pca.inverse_transform(transformed)
    errors = np.sum((X - reconstructed) ** 2, axis=1)
    threshold = np.quantile(errors, 1 - contamination)
    result = pd.DataFrame({"score": errors, "is_anomaly": errors > threshold}, index=sub.index)
    return result, pca


def elliptic_envelope_scores(
    df: pd.DataFrame, cols: list[str], contamination: float = 0.05, random_state: int = 42,
):
    sub, X = _prepare(df, cols)
    model = EllipticEnvelope(contamination=contamination, random_state=random_state)
    labels = model.fit_predict(X)
    scores = -model.score_samples(X)  # negative log-likelihood-ish; higher = more anomalous
    result = pd.DataFrame({"score": scores, "is_anomaly": labels == -1}, index=sub.index)
    return result, model


def ensemble_vote(results: dict[str, pd.DataFrame], min_votes: int = 2) -> pd.DataFrame:
    """Combine per-method boolean flags into a majority vote."""
    flags = pd.DataFrame({name: res["is_anomaly"] for name, res in results.items()})
    votes = flags.sum(axis=1)
    combined = pd.DataFrame({"votes": votes, "is_anomaly": votes >= min_votes}, index=flags.index)
    return combined


def score_plot(results: dict[str, pd.DataFrame], title: str = "Anomaly Scores by Method") -> go.Figure:
    n = len(results)
    fig = make_subplots(rows=n, cols=1, shared_xaxes=True, subplot_titles=list(results.keys()), vertical_spacing=0.08)
    for i, (name, res) in enumerate(results.items(), start=1):
        fig.add_trace(
            go.Scatter(x=res.index, y=res["score"], mode="lines", name=f"{name} score", line=dict(color="#2563eb")),
            row=i, col=1,
        )
        anom = res[res["is_anomaly"]]
        fig.add_trace(
            go.Scatter(x=anom.index, y=anom["score"], mode="markers", name=f"{name} anomaly",
                       marker=dict(color="#dc2626", size=7, symbol="x")),
            row=i, col=1,
        )
    fig.update_layout(template="plotly_white", height=260 * n, title=title, showlegend=False)
    return fig


def multivariate_overlay_plot(
    df: pd.DataFrame, cols: list[str], anomaly_index: pd.DatetimeIndex, title: str = "Series with Detected Anomalies"
) -> go.Figure:
    fig = go.Figure()
    for c in cols:
        fig.add_trace(go.Scatter(x=df.index, y=df[c], mode="lines", name=c))
    if len(anomaly_index):
        for i, c in enumerate(cols):
            fig.add_trace(
                go.Scatter(
                    x=anomaly_index, y=df.loc[anomaly_index, c], mode="markers",
                    marker=dict(color="#dc2626", size=8, symbol="x"),
                    name="Anomaly" if i == 0 else None, showlegend=(i == 0),
                )
            )
    fig.update_layout(template="plotly_white", title=title, hovermode="x unified", height=450,
                       legend=dict(orientation="h", yanchor="bottom", y=1.02))
    return fig
