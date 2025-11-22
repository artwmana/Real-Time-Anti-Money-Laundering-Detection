"""
Utility helpers to load AMLNet transactions and flatten the nested metadata
payload into tabular columns that downstream feature engineering can consume.
The logic mirrors the parsing pipeline used in notebooks/01_eda.ipynb.
"""

from __future__ import annotations

import ast
import os
import re
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import pandas as pd
from dotenv import load_dotenv


DATETIME_PATTERN = re.compile(r"datetime\.datetime\((.*?)\)")


def load_raw_transactions(
    relative_csv_path: str, nrows: Optional[int] = None
) -> pd.DataFrame:
    """
    Load the AMLNet CSV using the DATA_PATH environment variable.
    """
    load_dotenv()
    data_root = Path(os.getenv("DATA_PATH", "data")).expanduser()
    csv_path = data_root / relative_csv_path
    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset not found at {csv_path}")
    return pd.read_csv(csv_path, nrows=nrows)


def normalize_python_json_string(value: str) -> str:
    """
    Replace python datetime constructors with ISO strings so that ast.literal_eval
    works without executing arbitrary code.
    """

    def repl(match: re.Match[str]) -> str:
        args = match.group(1).split(",")
        nums = [int(a.strip()) for a in args]
        return f"'{pd.Timestamp(*nums).isoformat()}'"

    return DATETIME_PATTERN.sub(repl, value)


def parse_metadata_payload(value: Any) -> Dict[str, Any]:
    """
    Convert the metadata column (either dict or string) into a dictionary.
    """
    if isinstance(value, dict):
        return value
    if not isinstance(value, str):
        return {}
    normalized = normalize_python_json_string(value)
    return ast.literal_eval(normalized)


def expand_nested_dict(series: pd.Series, prefix: str) -> pd.DataFrame:
    """
    Expand nested dictionaries into a DataFrame and prefix column names.
    """
    expanded = (
        series.apply(lambda x: x if isinstance(x, dict) else {})
        .apply(pd.Series)
        .add_prefix(f"{prefix}_")
    )
    return expanded


def fill_object_nan(df: pd.DataFrame, fill_value: str = "Unknown") -> pd.DataFrame:
    """
    Replace NaNs in object columns for consistent downstream encoding.
    """
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].fillna(fill_value)
    return df


def coerce_numeric_columns(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    """
    Cast selected columns to numeric, ignoring errors gracefully.
    """
    for col in columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def flatten_metadata(df: pd.DataFrame) -> pd.DataFrame:
    """
    Flatten the metadata JSON into structured columns and return a new DataFrame.
    """
    if "metadata" not in df.columns:
        raise KeyError("Expected column 'metadata' to flatten payloads.")

    meta = df.pop("metadata").apply(parse_metadata_payload)
    meta_df = meta.apply(pd.Series)

    location = fill_object_nan(expand_nested_dict(meta_df.get("location"), "loc"))
    device = fill_object_nan(expand_nested_dict(meta_df.get("device_info"), "device"))
    merchant = fill_object_nan(
        expand_nested_dict(meta_df.get("merchant_info"), "merch")
    )
    risk = fill_object_nan(expand_nested_dict(meta_df.get("risk_indicators"), "risk"))

    coerce_numeric_columns(
        location,
        ["loc_postcode"],
    )
    coerce_numeric_columns(
        merchant,
        ["merch_avg_transaction"],
    )
    coerce_numeric_columns(
        risk,
        ["risk_amount_vs_average", "risk_customer_risk_score", "risk_risk_score"],
    )

    meta_subset = meta_df[["timestamp", "payment_method"]].copy()
    meta_subset["timestamp"] = pd.to_datetime(meta_subset["timestamp"], errors="coerce")
    meta_subset["payment_method"] = meta_subset["payment_method"].fillna("Unknown")

    flattened = pd.concat(
        [
            df,
            location,
            device,
            merchant,
            risk,
            meta_subset,
        ],
        axis=1,
    )
    return flattened


def transform_transactions(
    relative_csv_path: str, nrows: Optional[int] = None
) -> pd.DataFrame:
    """
    Load the raw dataset and return a DataFrame with flattened metadata.
    """
    raw_df = load_raw_transactions(relative_csv_path, nrows=nrows)
    return flatten_metadata(raw_df)


def main() -> None:
    """
    Quick manual test: read the default CSV and print resulting columns.
    """
    df = transform_transactions("raw/AMLNet_August_2025.csv", nrows=10_000)
    print(f"Flattened columns ({len(df.columns)}):")
    print(sorted(df.columns))


if __name__ == "__main__":
    main()
