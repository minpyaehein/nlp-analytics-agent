"""Quality gates for safe deterministic revenue and profit calculations."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Iterable

import pandas as pd


@dataclass
class ColumnQualityResult:
    """Quality measurements for one required numeric column."""

    column: str
    exists: bool
    total_rows: int
    numeric_count: int
    missing_count: int
    invalid_numeric_count: int
    usable_ratio: float
    missing_ratio: float
    minimum_value: float | None
    maximum_value: float | None
    negative_count: int
    zero_count: int
    passed: bool
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class AnalyticsQualityGateResult:
    """Complete quality-gate result for an analytical calculation."""

    status: str
    analytics_ready: bool
    gate_name: str
    row_count: int
    required_numeric_columns: list[str]
    missing_columns: list[str]
    unusable_numeric_columns: list[str]
    usable_numeric_ratios: dict[str, float]
    minimum_usable_ratio: float
    maximum_missing_ratio: float
    duplicate_rows: int
    total_missing_values: int
    column_results: list[ColumnQualityResult]
    warnings: list[str]
    errors: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def failure_message(self) -> str:
        """Return a readable reason when calculation is blocked."""

        if self.analytics_ready:
            return "The dataset passed the analytics quality gate."

        reasons: list[str] = []

        if self.missing_columns:
            reasons.append(
                "missing required columns: "
                + ", ".join(self.missing_columns)
            )

        if self.unusable_numeric_columns:
            reasons.append(
                "unusable numeric columns: "
                + ", ".join(self.unusable_numeric_columns)
            )

        if not reasons:
            reasons.append("one or more quality checks failed")

        return (
            "The dataset is not safe for the requested calculation because "
            + "; ".join(reasons)
            + ". Review the source data or OCR extraction before continuing."
        )

    def raise_if_failed(self) -> None:
        """Raise ValueError when the dataset is not analytics-ready."""

        if not self.analytics_ready:
            raise ValueError(self.failure_message())


def validate_analytics_readiness(
    dataframe: pd.DataFrame,
    required_numeric_columns: Iterable[str],
    minimum_usable_ratio: float = 0.80,
    maximum_missing_ratio: float = 0.20,
    allow_negative_values: bool = False,
    allow_zero_values: bool = True,
    gate_name: str = "analytics_quality_gate",
) -> AnalyticsQualityGateResult:
    """Validate required numerical evidence before calculation."""

    if not isinstance(dataframe, pd.DataFrame):
        raise TypeError("dataframe must be a Pandas DataFrame")

    _validate_ratio(minimum_usable_ratio, "minimum_usable_ratio")
    _validate_ratio(maximum_missing_ratio, "maximum_missing_ratio")

    required_columns = _normalize_required_columns(
        required_numeric_columns
    )
    row_count = int(len(dataframe))
    missing_columns: list[str] = []
    unusable_columns: list[str] = []
    ratios: dict[str, float] = {}
    column_results: list[ColumnQualityResult] = []
    warnings: list[str] = []
    errors: list[str] = []

    if row_count == 0:
        errors.append("The dataset contains no rows.")

    for column in required_columns:
        if column not in dataframe.columns:
            missing_columns.append(column)
            unusable_columns.append(column)
            ratios[column] = 0.0
            errors.append(f"Required numeric column '{column}' is missing.")
            column_results.append(
                ColumnQualityResult(
                    column=column,
                    exists=False,
                    total_rows=row_count,
                    numeric_count=0,
                    missing_count=row_count,
                    invalid_numeric_count=0,
                    usable_ratio=0.0,
                    missing_ratio=1.0 if row_count else 0.0,
                    minimum_value=None,
                    maximum_value=None,
                    negative_count=0,
                    zero_count=0,
                    passed=False,
                    warnings=["The required column is missing."],
                )
            )
            continue

        result = _validate_numeric_column(
            series=dataframe[column],
            column_name=column,
            row_count=row_count,
            minimum_usable_ratio=minimum_usable_ratio,
            maximum_missing_ratio=maximum_missing_ratio,
            allow_negative_values=allow_negative_values,
            allow_zero_values=allow_zero_values,
        )
        column_results.append(result)
        ratios[column] = result.usable_ratio

        if not result.passed:
            unusable_columns.append(column)
            errors.append(
                f"Column '{column}' failed with usable ratio "
                f"{result.usable_ratio:.1%}."
            )

        warnings.extend(
            f"{column}: {warning}" for warning in result.warnings
        )

    duplicate_rows = int(dataframe.duplicated().sum())
    if duplicate_rows:
        warnings.append(
            f"The dataset contains {duplicate_rows} duplicate row(s)."
        )

    analytics_ready = bool(
        row_count > 0
        and not missing_columns
        and not unusable_columns
    )

    return AnalyticsQualityGateResult(
        status="passed" if analytics_ready else "failed",
        analytics_ready=analytics_ready,
        gate_name=gate_name,
        row_count=row_count,
        required_numeric_columns=required_columns,
        missing_columns=missing_columns,
        unusable_numeric_columns=unusable_columns,
        usable_numeric_ratios=ratios,
        minimum_usable_ratio=float(minimum_usable_ratio),
        maximum_missing_ratio=float(maximum_missing_ratio),
        duplicate_rows=duplicate_rows,
        total_missing_values=int(dataframe.isna().sum().sum()),
        column_results=column_results,
        warnings=warnings,
        errors=errors,
    )


def validate_revenue_readiness(
    dataframe: pd.DataFrame,
    minimum_usable_ratio: float = 0.80,
    maximum_missing_ratio: float = 0.20,
) -> AnalyticsQualityGateResult:
    """Validate quantity and unit_price for revenue calculation."""

    return validate_analytics_readiness(
        dataframe=dataframe,
        required_numeric_columns=["quantity", "unit_price"],
        minimum_usable_ratio=minimum_usable_ratio,
        maximum_missing_ratio=maximum_missing_ratio,
        allow_negative_values=False,
        allow_zero_values=True,
        gate_name="revenue_quality_gate",
    )


def validate_profit_readiness(
    dataframe: pd.DataFrame,
    minimum_usable_ratio: float = 0.80,
    maximum_missing_ratio: float = 0.20,
) -> AnalyticsQualityGateResult:
    """Validate quantity, unit_price, and unit_cost for profit calculation."""

    return validate_analytics_readiness(
        dataframe=dataframe,
        required_numeric_columns=[
            "quantity",
            "unit_price",
            "unit_cost",
        ],
        minimum_usable_ratio=minimum_usable_ratio,
        maximum_missing_ratio=maximum_missing_ratio,
        allow_negative_values=False,
        allow_zero_values=True,
        gate_name="profit_quality_gate",
    )


def require_revenue_ready(
    dataframe: pd.DataFrame,
    minimum_usable_ratio: float = 0.80,
    maximum_missing_ratio: float = 0.20,
) -> AnalyticsQualityGateResult:
    """Validate revenue inputs and raise if unsafe."""

    result = validate_revenue_readiness(
        dataframe,
        minimum_usable_ratio,
        maximum_missing_ratio,
    )
    result.raise_if_failed()
    return result


def require_profit_ready(
    dataframe: pd.DataFrame,
    minimum_usable_ratio: float = 0.80,
    maximum_missing_ratio: float = 0.20,
) -> AnalyticsQualityGateResult:
    """Validate profit inputs and raise if unsafe."""

    result = validate_profit_readiness(
        dataframe,
        minimum_usable_ratio,
        maximum_missing_ratio,
    )
    result.raise_if_failed()
    return result


def _validate_numeric_column(
    series: pd.Series,
    column_name: str,
    row_count: int,
    minimum_usable_ratio: float,
    maximum_missing_ratio: float,
    allow_negative_values: bool,
    allow_zero_values: bool,
) -> ColumnQualityResult:
    """Measure quality for one required numeric column."""

    normalized = _normalize_numeric_text(series)
    non_null_mask = normalized.notna()
    numeric = pd.to_numeric(normalized, errors="coerce")
    invalid_numeric_count = int((non_null_mask & numeric.isna()).sum())
    numeric_count = int(numeric.notna().sum())
    missing_count = int(numeric.isna().sum())
    usable_ratio = float(numeric_count / row_count) if row_count else 0.0
    missing_ratio = float(missing_count / row_count) if row_count else 0.0
    valid = numeric.dropna()
    minimum_value = float(valid.min()) if not valid.empty else None
    maximum_value = float(valid.max()) if not valid.empty else None
    negative_count = int((valid < 0).sum())
    zero_count = int((valid == 0).sum())
    column_warnings: list[str] = []
    passed = True

    if usable_ratio < minimum_usable_ratio:
        passed = False
        column_warnings.append(
            f"Usable ratio {usable_ratio:.1%} is below "
            f"{minimum_usable_ratio:.1%}."
        )

    if missing_ratio > maximum_missing_ratio:
        passed = False
        column_warnings.append(
            f"Missing ratio {missing_ratio:.1%} exceeds "
            f"{maximum_missing_ratio:.1%}."
        )

    if invalid_numeric_count:
        column_warnings.append(
            f"{invalid_numeric_count} non-empty value(s) are not numeric."
        )

    if not allow_negative_values and negative_count:
        passed = False
        column_warnings.append(
            f"{negative_count} negative value(s) were detected."
        )

    if not allow_zero_values and zero_count:
        passed = False
        column_warnings.append(
            f"{zero_count} zero value(s) were detected."
        )

    return ColumnQualityResult(
        column=column_name,
        exists=True,
        total_rows=row_count,
        numeric_count=numeric_count,
        missing_count=missing_count,
        invalid_numeric_count=invalid_numeric_count,
        usable_ratio=round(usable_ratio, 4),
        missing_ratio=round(missing_ratio, 4),
        minimum_value=minimum_value,
        maximum_value=maximum_value,
        negative_count=negative_count,
        zero_count=zero_count,
        passed=passed,
        warnings=column_warnings,
    )


def _normalize_numeric_text(series: pd.Series) -> pd.Series:
    """Normalize common spreadsheet and OCR numeric formatting."""

    if pd.api.types.is_numeric_dtype(series):
        return series.copy()

    normalized = series.astype("string").str.strip()
    normalized = normalized.replace(
        {
            "": pd.NA,
            "nan": pd.NA,
            "NaN": pd.NA,
            "none": pd.NA,
            "None": pd.NA,
            "null": pd.NA,
            "NULL": pd.NA,
        }
    )
    normalized = normalized.str.replace(",", "", regex=False)
    normalized = normalized.str.replace(
        r"^\((.+)\)$",
        r"-\1",
        regex=True,
    )
    normalized = normalized.str.replace(
        r"[$€£¥]",
        "",
        regex=True,
    )
    return normalized


def _normalize_required_columns(
    required_numeric_columns: Iterable[str],
) -> list[str]:
    """Return non-empty, unique required-column names."""

    normalized: list[str] = []

    for column in required_numeric_columns:
        cleaned = str(column).strip()
        if not cleaned:
            raise ValueError("Required column names cannot be empty.")
        if cleaned not in normalized:
            normalized.append(cleaned)

    if not normalized:
        raise ValueError(
            "At least one required numeric column must be provided."
        )

    return normalized


def _validate_ratio(value: float, name: str) -> None:
    """Validate a ratio between zero and one."""

    if not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric")
    if not 0 <= float(value) <= 1:
        raise ValueError(f"{name} must be between 0 and 1")
