"""Forecast accuracy metrics."""
from __future__ import annotations

import numpy as np
import pandas as pd


def evaluate(actual: pd.Series, predicted: pd.Series) -> dict:
    """Align two series on their shared index and compute standard metrics."""
    df = pd.concat([actual.rename("actual"), predicted.rename("predicted")], axis=1).dropna()
    if df.empty:
        return {"MAE": np.nan, "RMSE": np.nan, "MAPE": np.nan, "R2": np.nan, "n": 0}

    a, p = df["actual"].values, df["predicted"].values
    mae = np.mean(np.abs(a - p))
    rmse = np.sqrt(np.mean((a - p) ** 2))
    nonzero = a != 0
    mape = np.mean(np.abs((a[nonzero] - p[nonzero]) / a[nonzero])) * 100 if nonzero.any() else np.nan
    ss_res = np.sum((a - p) ** 2)
    ss_tot = np.sum((a - np.mean(a)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan

    return {"MAE": mae, "RMSE": rmse, "MAPE": mape, "R2": r2, "n": len(df)}


def metrics_table(results: dict[str, dict], actual: pd.Series) -> pd.DataFrame:
    """Build a comparison table across multiple model results (test-period accuracy)."""
    rows = {}
    for name, res in results.items():
        m = evaluate(actual, res["forecast"])
        rows[name] = m
    return pd.DataFrame(rows).T[["MAE", "RMSE", "MAPE", "R2", "n"]]
