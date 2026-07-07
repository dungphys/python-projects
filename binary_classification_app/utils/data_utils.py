"""
Data loading, type inference, and binary-target detection/encoding utilities.
No column names are assumed in advance -- everything is inferred from the
uploaded CSV at runtime.
"""
from __future__ import annotations

import io
import pandas as pd
import numpy as np


# Common string encodings people use for a binary target.
_POSITIVE_TOKENS = {
    "yes", "y", "true", "t", "1", "positive", "pos", "good", "success",
    "churn", "default", "fraud", "attrited", "attritedcustomer",
}
_NEGATIVE_TOKENS = {
    "no", "n", "false", "f", "0", "negative", "neg", "bad", "failure",
    "nochurn", "no default", "not fraud", "existing", "existingcustomer",
}


def load_csv(uploaded_file) -> pd.DataFrame:
    """Load a CSV (file-like or path) into a DataFrame with light cleanup."""
    if hasattr(uploaded_file, "read"):
        raw = uploaded_file.read()
        if isinstance(raw, bytes):
            buffer = io.BytesIO(raw)
        else:
            buffer = io.StringIO(raw)
        df = pd.read_csv(buffer)
    else:
        df = pd.read_csv(uploaded_file)

    # Strip whitespace from column names; leave data values untouched here.
    df.columns = [str(c).strip() for c in df.columns]
    return df


def get_column_types(df: pd.DataFrame, exclude: list[str] | None = None):
    """Split columns into numeric and categorical, optionally excluding some."""
    exclude = exclude or []
    cols = [c for c in df.columns if c not in exclude]
    numeric_cols = [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]
    categorical_cols = [c for c in cols if c not in numeric_cols]
    return numeric_cols, categorical_cols


def detect_binary_candidates(df: pd.DataFrame) -> list[str]:
    """Return columns that plausibly hold a binary target (exactly 2 unique
    non-null values, or numeric columns with few unique values)."""
    candidates = []
    for col in df.columns:
        nunique = df[col].nunique(dropna=True)
        if nunique == 2:
            candidates.append(col)
    return candidates


def encode_target(series: pd.Series) -> tuple[pd.Series, dict]:
    """Encode an arbitrary 2-class series into {0, 1}.

    Returns the encoded series (int, NaNs preserved as NaN then droppable
    upstream) and a mapping dict {original_value: encoded_value} for display.
    """
    s = series.copy()
    uniques = [u for u in s.dropna().unique()]
    if len(uniques) != 2:
        raise ValueError(f"Target column must have exactly 2 classes, found {len(uniques)}.")

    # If already numeric 0/1, keep as-is (but normalize dtype).
    numeric_like = all(_is_zero_or_one(u) for u in uniques)
    if numeric_like:
        mapping = {u: int(round(float(u))) for u in uniques}
    else:
        # Try to use positive/negative token matching for a sensible 0/1 mapping.
        str_uniques = {str(u).strip().lower(): u for u in uniques}
        pos_match = [orig for key, orig in str_uniques.items() if key in _POSITIVE_TOKENS]
        neg_match = [orig for key, orig in str_uniques.items() if key in _NEGATIVE_TOKENS]

        if len(pos_match) == 1 and len(neg_match) == 1:
            mapping = {neg_match[0]: 0, pos_match[0]: 1}
        else:
            # Fallback: sort for reproducibility, first (alphabetically) -> 0
            sorted_uniques = sorted(uniques, key=lambda x: str(x))
            mapping = {sorted_uniques[0]: 0, sorted_uniques[1]: 1}

    encoded = s.map(mapping)
    return encoded.astype("Int64"), mapping


def _is_zero_or_one(val) -> bool:
    try:
        f = float(val)
        return f in (0.0, 1.0)
    except (TypeError, ValueError):
        return False


def basic_overview(df: pd.DataFrame) -> dict:
    """Compute summary stats used in the Data Overview tab."""
    n_rows, n_cols = df.shape
    missing = df.isna().sum()
    missing_pct = (missing / n_rows * 100).round(2)
    dup_count = int(df.duplicated().sum())

    dtypes = df.dtypes.astype(str)

    overview = {
        "n_rows": n_rows,
        "n_cols": n_cols,
        "duplicate_rows": dup_count,
        "missing_by_col": pd.DataFrame(
            {"missing_count": missing, "missing_pct": missing_pct}
        ).sort_values("missing_count", ascending=False),
        "dtypes": dtypes,
        "memory_mb": round(df.memory_usage(deep=True).sum() / (1024 ** 2), 3),
    }
    return overview
