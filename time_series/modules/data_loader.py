"""
Data loading utilities: reads CSV, TSV, XLS/XLSX and returns a clean DataFrame
with a parsed datetime index, plus helper functions for frequency inference
and resampling.
"""
from __future__ import annotations

import io
import pandas as pd
import numpy as np


def load_file(uploaded_file) -> pd.DataFrame:
    """
    Load a CSV, TSV, or Excel file (from a Streamlit UploadedFile or a path)
    into a raw DataFrame (no date parsing yet -- that happens after the user
    picks the date column).
    """
    name = getattr(uploaded_file, "name", str(uploaded_file))
    lower = name.lower()

    if lower.endswith((".xlsx", ".xls")):
        df = pd.read_excel(uploaded_file)
    elif lower.endswith(".tsv"):
        df = pd.read_csv(uploaded_file, sep="\t")
    else:
        # Default to CSV, but sniff the separator in case it's actually
        # semicolon- or tab-delimited (common in exported European data).
        raw = uploaded_file.read() if hasattr(uploaded_file, "read") else None
        if raw is not None:
            if isinstance(raw, bytes):
                sample = raw[:5000].decode("utf-8", errors="ignore")
            else:
                sample = raw[:5000]
            uploaded_file.seek(0)
            sep = _sniff_separator(sample)
            df = pd.read_csv(uploaded_file, sep=sep, engine="python")
        else:
            df = pd.read_csv(uploaded_file)

    df.columns = [str(c).strip() for c in df.columns]
    return df


def _sniff_separator(sample: str) -> str:
    first_line = sample.splitlines()[0] if sample.splitlines() else ""
    counts = {sep: first_line.count(sep) for sep in [",", ";", "\t", "|"]}
    best = max(counts, key=counts.get)
    return best if counts[best] > 0 else ","


def guess_datetime_column(df: pd.DataFrame) -> str | None:
    """Heuristically guess which column holds the timestamp."""
    candidates = []
    for col in df.columns:
        lc = str(col).lower()
        if any(k in lc for k in ["date", "datetime", "time", "timestamp", "ds", "period"]):
            candidates.append(col)

    if candidates:
        return candidates[0]

    # Fall back: try parsing each column and see which one mostly succeeds.
    for col in df.columns:
        try:
            parsed = pd.to_datetime(df[col], errors="coerce")
            if parsed.notna().mean() > 0.9:
                return col
        except Exception:
            continue
    return None


def build_time_index(df: pd.DataFrame, date_col: str) -> pd.DataFrame:
    """Parse the chosen column as datetime, sort, and set as index."""
    out = df.copy()
    out[date_col] = pd.to_datetime(out[date_col], errors="coerce")
    n_bad = out[date_col].isna().sum()
    out = out.dropna(subset=[date_col])
    out = out.sort_values(date_col).set_index(date_col)
    out.index.name = "date"
    return out, n_bad

# frequency
FREQ_LABELS = {
    "D": "Daily",
    "B": "Business day",
    "W": "Weekly",
    "MS": "Monthly (start)",
    "M": "Monthly (end)",
    "Q": "Quarterly",
    "Y": "Yearly",
    "H": "Hourly",
    "T": "Minute",
}


def infer_frequency(index: pd.DatetimeIndex) -> str | None:
    """Try pandas' inference; fall back to the modal delta between points."""
    freq = pd.infer_freq(index)
    if freq:
        return freq
    if len(index) < 2:
        return None
    deltas = index.to_series().diff().dropna()
    if deltas.empty:
        return None
    modal_delta = deltas.mode().iloc[0]
    days = modal_delta.total_seconds() / 86400
    if days >= 360:
        return "Y"
    if days >= 28:
        return "MS"
    if days >= 6.5:
        return "W"
    if days >= 0.9:
        return "D"
    if days >= 1 / 24:
        return "H"
    return "T"


def numeric_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
