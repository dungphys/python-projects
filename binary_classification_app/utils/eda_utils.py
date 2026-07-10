"""
Exploratory Data Analysis plot builders. All functions return Plotly figures
so they render natively and interactively inside Streamlit.
"""
from __future__ import annotations

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go


def target_balance_fig(y: pd.Series, target_name: str) -> go.Figure:
    counts = y.value_counts().sort_index()
    labels = [f"Class {i}" for i in counts.index]
    fig = px.bar(
        x=labels, y=counts.values, text=counts.values,
        labels={"x": target_name, "y": "Count"},
        title=f"Target Class Balance: {target_name}",
        color=labels, color_discrete_sequence=px.colors.qualitative.D3
    )
    fig.update_traces(textposition="outside")
    return fig


def missing_values_fig(df: pd.DataFrame) -> go.Figure:
    missing = df.isna().sum()
    missing = missing[missing > 0].sort_values(ascending=True)
    if missing.empty:
        fig = go.Figure()
        fig.add_annotation(text="No missing values found", showarrow=False,
                            font=dict(size=16))
        fig.update_layout(title="Missing Values by Column")
        return fig
    fig = px.bar(
        x=missing.values, y=missing.index, orientation="h",
        labels={"x": "Missing Count", "y": "Column"},
        title="Missing Values by Column",
    )
    return fig


def numeric_distribution_fig(df: pd.DataFrame, col: str, y: pd.Series | None = None,
                              target_name: str | None = None) -> go.Figure:
    if y is not None and target_name is not None:
        plot_df = pd.DataFrame({col: df[col], target_name: y.astype(str)})
        fig = px.histogram(
            plot_df, x=col, color=target_name, marginal="box",
            barmode="overlay", opacity=0.65,
            title=f"Distribution of {col} by {target_name}",
        )
    else:
        fig = px.histogram(df, x=col, marginal="box", title=f"Distribution of {col}")
    return fig


def categorical_distribution_fig(df: pd.DataFrame, col: str, y: pd.Series | None = None,
                                  target_name: str | None = None, top_n: int = 15) -> go.Figure:
    vc = df[col].value_counts(dropna=False).head(top_n)
    if y is not None and target_name is not None:
        plot_df = pd.DataFrame({col: df[col].astype(str), target_name: y.astype(str)})
        plot_df = plot_df[plot_df[col].isin(vc.index.astype(str))]
        fig = px.histogram(
            plot_df, x=col, color=target_name, barmode="group",
            title=f"{col} counts by {target_name} (top {top_n})",
        )
    else:
        fig = px.bar(x=vc.index.astype(str), y=vc.values,
                      labels={"x": col, "y": "Count"},
                      title=f"{col} value counts (top {top_n})")
    return fig


def correlation_heatmap_fig(df: pd.DataFrame, numeric_cols: list[str]) -> go.Figure:
    if len(numeric_cols) < 2:
        fig = go.Figure()
        fig.add_annotation(text="Need at least 2 numeric columns for a correlation heatmap",
                            showarrow=False, font=dict(size=14))
        return fig
    corr = df[numeric_cols].corr()
    fig = px.imshow(
        corr, text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
        title="Correlation Heatmap (numeric features)",
        aspect="auto",
    )
    return fig


def boxplot_fig(df: pd.DataFrame, col: str, y: pd.Series, target_name: str) -> go.Figure:
    plot_df = pd.DataFrame({col: df[col], target_name: y.astype(str)})
    is_log_y = True if (plot_df[col] > 0).all() and (plot_df[col].max() / plot_df[col].min() > 100) else False
    fig = px.box(plot_df, x=target_name, y=col, color=target_name,
                  title=f"{col} by {target_name}", log_y=is_log_y, color_discrete_sequence=px.colors.qualitative.D3)
    return fig
