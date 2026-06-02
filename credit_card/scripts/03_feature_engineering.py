from pathlib import Path
import pandas as pd


def _resolve_input_path() -> Path:
    base_dir = Path(__file__).resolve().parent
    print(base_dir)
    candidates = [
        base_dir / "cleaned_data.csv",
        base_dir.parent / "cleaned_data.csv",
        base_dir / "data" / "cleaned_data.csv",
        base_dir.parent / "data" / "cleaned_data.csv",
        Path("cleaned_data.csv"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "Could not find cleaned_data.csv in the current project structure. "
        "Place the file in the data directory."
    )


def safe_ratio(df: pd.DataFrame, numerator: str, denominator: str, output: str) -> None:
    if numerator not in df.columns or denominator not in df.columns:
        return
    denom = df[denominator].replace({0: pd.NA})
    df[output] = df[numerator] / denom


def add_boolean_feature(df: pd.DataFrame, source: str, threshold: float, output: str, greater: bool = True) -> None:
    if source not in df.columns:
        return
    if greater:
        df[output] = df[source] > threshold
    else:
        df[output] = df[source] < threshold


def add_feature_columns(df: pd.DataFrame) -> pd.DataFrame:
    # Common credit card derived ratios and flags
    safe_ratio(df, "total_trans_amt", "total_trans_ct", "avg_transaction_amount")
    safe_ratio(df, "current_balance", "credit_limit", "credit_utilization")
    safe_ratio(df, "current_balance", "monthly_income", "balance_to_income")
    safe_ratio(df, "total_payment_amt", "monthly_income", "payment_to_income_ratio")
    safe_ratio(df, "total_trans_amt", "monthly_income", "spend_to_income_ratio")

    add_boolean_feature(df, "credit_utilization", 0.85, "high_credit_utilization")
    add_boolean_feature(df, "balance_to_income", 0.5, "high_balance_to_income")
    add_boolean_feature(df, "payment_to_income_ratio", 0.35, "high_payment_to_income")

    if "months_since_last_activity" in df.columns:
        df["recent_activity_flag"] = df["months_since_last_activity"] <= 3

    if "num_late_payments" in df.columns and "months_on_book" in df.columns:
        safe_ratio(df, "num_late_payments", "months_on_book", "late_payment_rate")
        add_boolean_feature(df, "late_payment_rate", 0.1, "high_late_payment_rate")

    if "total_trans_amt" in df.columns and "total_trans_ct" in df.columns:
        df["transaction_frequency"] = df["total_trans_ct"]

    if "avg_utilization_ratio" not in df.columns and "credit_utilization" in df.columns:
        df["avg_utilization_ratio"] = df["credit_utilization"]

    if "monthly_income" in df.columns and "age" in df.columns:
        df["income_per_age_year"] = df["monthly_income"] / df["age"].replace({0: pd.NA})

    return df


def main() -> None:
    input_path = _resolve_input_path()
    df = pd.read_csv(input_path)
    df = add_feature_columns(df)
    output_path = input_path.parent / "fe_data.csv"
    df.to_csv(output_path, index=False)
    print(f"Feature engineering complete. Output saved to: {output_path}")


if __name__ == "__main__":
    main()
