"""Deterministic dataset profiling utilities.

This module calculates dataset structure, completeness, uniqueness,
column statistics, and an initial quality score without using an LLM.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def profile_dataframe(dataframe: pd.DataFrame) -> dict[str, Any]:
    """Generate a deterministic profile for a Pandas DataFrame.

    Args:
        dataframe: Dataset to profile.

    Returns:
        Dictionary containing row and column counts, duplicate counts,
        missing values, column profiles, and an initial quality score.

    Raises:
        TypeError: If dataframe is not a Pandas DataFrame.
    """

    if not isinstance(dataframe, pd.DataFrame):
        raise TypeError("dataframe must be a pandas DataFrame")

    row_count = int(len(dataframe))
    column_count = int(len(dataframe.columns))
    duplicate_rows = int(dataframe.duplicated().sum())

    missing_by_column = {
        str(column): int(missing_count)
        for column, missing_count in dataframe.isna().sum().items()
    }

    total_missing_values = int(dataframe.isna().sum().sum())

    column_profiles = [
        profile_column(
            series=dataframe[column],
            column_name=str(column),
        )
        for column in dataframe.columns
    ]

    total_cells = row_count * column_count

    if total_cells > 0:
        completeness_score = 1.0 - (
            total_missing_values / total_cells
        )
    else:
        completeness_score = 0.0

    if row_count > 0:
        uniqueness_score = 1.0 - (
            duplicate_rows / row_count
        )
    else:
        uniqueness_score = 0.0

    quality_score = round(
        100.0
        * (
            0.70 * completeness_score
            + 0.30 * uniqueness_score
        ),
        2,
    )

    return {
        "row_count": row_count,
        "column_count": column_count,
        "duplicate_rows": duplicate_rows,
        "total_missing_values": total_missing_values,
        "quality_score": quality_score,
        "missing_by_column": missing_by_column,
        "columns": column_profiles,
    }


def profile_column(
    series: pd.Series,
    column_name: str,
) -> dict[str, Any]:
    """Generate a deterministic profile for one DataFrame column."""

    non_null_series = series.dropna()
    row_count = int(len(series))
    null_count = int(series.isna().sum())

    if row_count > 0:
        null_percentage = round(
            float(null_count / row_count * 100.0),
            2,
        )
    else:
        null_percentage = 0.0

    profile: dict[str, Any] = {
        "column": column_name,
        "dtype": str(series.dtype),
        "semantic_type": infer_semantic_type(
            series=series,
            column_name=column_name,
        ),
        "null_count": null_count,
        "null_percentage": null_percentage,
        "unique_count": int(series.nunique(dropna=True)),
        "sample_values": [
            make_json_safe(value)
            for value in non_null_series.head(3).tolist()
        ],
    }

    if pd.api.types.is_numeric_dtype(series):
        profile.update(
            profile_numeric_column(non_null_series)
        )

    date_statistics = profile_date_column(
        series=series,
        column_name=column_name,
    )

    if date_statistics:
        profile.update(date_statistics)

    return profile


def profile_numeric_column(series: pd.Series) -> dict[str, Any]:
    """Calculate descriptive statistics for a numeric column."""

    if series.empty:
        return empty_numeric_profile()

    numeric_series = pd.to_numeric(
        series,
        errors="coerce",
    ).dropna()

    if numeric_series.empty:
        return empty_numeric_profile()

    standard_deviation = (
        numeric_series.std()
        if len(numeric_series) > 1
        else 0.0
    )

    return {
        "minimum": make_json_safe(numeric_series.min()),
        "maximum": make_json_safe(numeric_series.max()),
        "mean": make_json_safe(numeric_series.mean()),
        "median": make_json_safe(numeric_series.median()),
        "standard_deviation": make_json_safe(
            standard_deviation
        ),
    }


def empty_numeric_profile() -> dict[str, None]:
    """Return empty numeric statistics."""

    return {
        "minimum": None,
        "maximum": None,
        "mean": None,
        "median": None,
        "standard_deviation": None,
    }


def profile_date_column(
    series: pd.Series,
    column_name: str,
) -> dict[str, Any]:
    """Return date-range statistics for likely date columns."""

    semantic_type = infer_semantic_type(
        series=series,
        column_name=column_name,
    )

    if semantic_type != "date":
        return {}

    parsed_dates = pd.to_datetime(
        series,
        errors="coerce",
    ).dropna()

    if parsed_dates.empty:
        return {
            "minimum_date": None,
            "maximum_date": None,
            "valid_date_count": 0,
        }

    return {
        "minimum_date": make_json_safe(parsed_dates.min()),
        "maximum_date": make_json_safe(parsed_dates.max()),
        "valid_date_count": int(len(parsed_dates)),
    }


def infer_semantic_type(
    series: pd.Series,
    column_name: str,
) -> str:
    """Infer an initial semantic role for a dataset column."""

    normalized_name = normalize_column_name(column_name)

    identifier_names = {
        "id",
        "order_id",
        "customer_id",
        "product_id",
        "transaction_id",
        "invoice_id",
        "record_id",
    }

    if (
        normalized_name in identifier_names
        or normalized_name.endswith("_id")
    ):
        return "identifier"

    date_keywords = [
        "date",
        "time",
        "timestamp",
        "datetime",
        "year",
        "month",
        "day",
    ]

    if any(
        keyword in normalized_name
        for keyword in date_keywords
    ):
        return "date"

    if pd.api.types.is_datetime64_any_dtype(series):
        return "date"

    if pd.api.types.is_bool_dtype(series):
        return "boolean"

    if pd.api.types.is_numeric_dtype(series):
        return "measure"

    row_count = max(int(len(series)), 1)
    unique_count = int(series.nunique(dropna=True))
    unique_ratio = unique_count / row_count

    if unique_count <= 50:
        return "category"

    if unique_ratio <= 0.20:
        return "category"

    return "text"


def normalize_column_name(column_name: str) -> str:
    """Normalize a column name for simple semantic checks."""

    normalized = str(column_name).strip().lower()
    normalized = normalized.replace(" ", "_")
    normalized = normalized.replace("-", "_")

    while "__" in normalized:
        normalized = normalized.replace("__", "_")

    return normalized.strip("_")


def make_json_safe(value: Any) -> Any:
    """Convert Pandas and NumPy values to JSON-safe Python values."""

    if value is None:
        return None

    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass

    if isinstance(value, np.integer):
        return int(value)

    if isinstance(value, np.floating):
        return float(value)

    if isinstance(value, np.bool_):
        return bool(value)

    if isinstance(value, pd.Timestamp):
        return value.isoformat()

    if isinstance(value, np.datetime64):
        return pd.Timestamp(value).isoformat()

    if isinstance(value, np.ndarray):
        return [make_json_safe(item) for item in value.tolist()]

    if isinstance(value, (list, tuple)):
        return [make_json_safe(item) for item in value]

    if isinstance(value, dict):
        return {
            str(key): make_json_safe(item)
            for key, item in value.items()
        }

    return value


if __name__ == "__main__":
    sample = pd.DataFrame(
        {
            "order_id": [1, 2, 2],
            "order_date": [
                "2026-01-01",
                "2026-01-02",
                "2026-01-02",
            ],
            "product": ["Laptop", "Mouse", "Mouse"],
            "revenue": [1000.0, 50.0, 50.0],
            "region": ["Yangon", None, None],
        }
    )

    print(profile_dataframe(sample))
