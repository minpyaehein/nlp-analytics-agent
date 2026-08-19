"""Apply validated filters before running the existing analytics executor."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import pandas as pd

from app.models.analysis_plan import AnalysisPlan, FilterCondition
from app.services.analysis_executor import ExecutionResult, execute_analysis_plan
from app.services.schema_linker import link_field


SUPPORTED_OPERATORS = {
    "equals",
    "not_equals",
    "greater_than",
    "greater_than_or_equal",
    "less_than",
    "less_than_or_equal",
    "contains",
    "in",
    "between",
}


@dataclass
class FilterAudit:
    """Evidence describing how one filter changed the dataset."""

    requested_column: str
    source_column: str
    operator: str
    value: Any
    rows_before: int
    rows_after: int
    matched_rows: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class FilteredExecutionResult:
    """Combined filter audit and deterministic analysis result."""

    success: bool
    source_rows: int
    filtered_rows: int
    applied_filters: list[dict[str, Any]]
    filter_warnings: list[str]
    analysis: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _normalize_operator(operator: str) -> str:
    """Normalize common operator aliases to an approved operator."""

    normalized = str(operator).strip().lower().replace(" ", "_")

    aliases = {
        "=": "equals",
        "==": "equals",
        "eq": "equals",
        "!=": "not_equals",
        "ne": "not_equals",
        ">": "greater_than",
        ">=": "greater_than_or_equal",
        "<": "less_than",
        "<=": "less_than_or_equal",
    }

    normalized = aliases.get(normalized, normalized)

    if normalized not in SUPPORTED_OPERATORS:
        raise ValueError(
            f"Unsupported filter operator: '{operator}'."
        )

    return normalized


def _resolve_filter_column(
    df: pd.DataFrame,
    requested_column: str,
) -> str:
    """Resolve a semantic filter field to an actual dataset column."""

    if requested_column in df.columns:
        return requested_column

    link = link_field(
        semantic_field=requested_column,
        df=df,
        allow_derived=False,
    )

    if link.source_column is None:
        raise ValueError(
            f"Filter column '{requested_column}' could not be linked "
            "to the dataset."
        )

    return link.source_column


def _text_equals_mask(
    series: pd.Series,
    value: Any,
) -> pd.Series:
    """Perform trimmed, case-insensitive text equality."""

    return (
        series.astype("string")
        .str.strip()
        .str.casefold()
        .eq(str(value).strip().casefold())
        .fillna(False)
    )


def _numeric_series(series: pd.Series, column: str) -> pd.Series:
    """Convert a series to numeric and reject unusable columns."""

    numeric = pd.to_numeric(series, errors="coerce")

    if numeric.notna().sum() == 0:
        raise ValueError(
            f"Column '{column}' does not contain usable numeric values."
        )

    return numeric


def _build_filter_mask(
    df: pd.DataFrame,
    source_column: str,
    operator: str,
    value: Any,
) -> pd.Series:
    """Build a Boolean mask for one approved filter condition."""

    series = df[source_column]

    if operator == "equals":
        if pd.api.types.is_numeric_dtype(series):
            numeric = _numeric_series(series, source_column)
            return numeric.eq(float(value)).fillna(False)

        return _text_equals_mask(series, value)

    if operator == "not_equals":
        if pd.api.types.is_numeric_dtype(series):
            numeric = _numeric_series(series, source_column)
            return numeric.ne(float(value)).fillna(False)

        return (~_text_equals_mask(series, value)).fillna(False)

    if operator == "contains":
        return (
            series.astype("string")
            .str.casefold()
            .str.contains(
                str(value).casefold(),
                regex=False,
                na=False,
            )
        )

    if operator == "in":
        values = value if isinstance(value, list) else [value]

        if pd.api.types.is_numeric_dtype(series):
            numeric_values = [float(item) for item in values]
            numeric = _numeric_series(series, source_column)
            return numeric.isin(numeric_values).fillna(False)

        normalized_values = {
            str(item).strip().casefold()
            for item in values
        }
        return (
            series.astype("string")
            .str.strip()
            .str.casefold()
            .isin(normalized_values)
            .fillna(False)
        )

    numeric = _numeric_series(series, source_column)

    if operator == "greater_than":
        return numeric.gt(float(value)).fillna(False)

    if operator == "greater_than_or_equal":
        return numeric.ge(float(value)).fillna(False)

    if operator == "less_than":
        return numeric.lt(float(value)).fillna(False)

    if operator == "less_than_or_equal":
        return numeric.le(float(value)).fillna(False)

    if operator == "between":
        if not isinstance(value, list) or len(value) != 2:
            raise ValueError(
                "The 'between' operator requires a two-item list."
            )

        lower = float(value[0])
        upper = float(value[1])

        if lower > upper:
            lower, upper = upper, lower

        return numeric.between(lower, upper, inclusive="both").fillna(False)

    raise ValueError(f"Unsupported operator: '{operator}'.")


def apply_plan_filters(
    df: pd.DataFrame,
    filters: list[FilterCondition],
) -> tuple[pd.DataFrame, list[FilterAudit], list[str]]:
    """Apply all validated filters sequentially and return audit evidence."""

    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame")

    filtered_df = df.copy()
    audits: list[FilterAudit] = []
    warnings: list[str] = []

    for condition in filters:
        operator = _normalize_operator(condition.operator)
        source_column = _resolve_filter_column(
            filtered_df,
            condition.column,
        )

        rows_before = len(filtered_df)
        mask = _build_filter_mask(
            filtered_df,
            source_column,
            operator,
            condition.value,
        )
        matched_rows = int(mask.sum())
        filtered_df = filtered_df.loc[mask].copy()
        rows_after = len(filtered_df)

        audits.append(
            FilterAudit(
                requested_column=condition.column,
                source_column=source_column,
                operator=operator,
                value=condition.value,
                rows_before=rows_before,
                rows_after=rows_after,
                matched_rows=matched_rows,
            )
        )

        if matched_rows == 0:
            warnings.append(
                "Filter matched no rows: "
                f"{condition.column} {operator} {condition.value}."
            )

    return filtered_df, audits, warnings


def execute_filtered_analysis(
    df: pd.DataFrame,
    plan: AnalysisPlan,
) -> FilteredExecutionResult:
    """Apply plan filters, then execute the existing deterministic analysis."""

    source_rows = len(df)
    filtered_df, audits, filter_warnings = apply_plan_filters(
        df,
        plan.filters,
    )

    if filtered_df.empty:
        raise ValueError(
            "No dataset rows remained after applying the requested filters."
        )

    # The copy keeps the original plan unchanged for evidence and UI display.
    execution_plan = plan.model_copy(
        update={"filters": []}
    )

    analysis_result: ExecutionResult = execute_analysis_plan(
        filtered_df,
        execution_plan,
    )

    analysis_dict = analysis_result.to_dict()
    analysis_dict["validation"]["source_rows_before_filters"] = source_rows
    analysis_dict["validation"]["rows_after_filters"] = len(filtered_df)
    analysis_dict["validation"]["applied_filter_count"] = len(audits)

    return FilteredExecutionResult(
        success=analysis_result.success,
        source_rows=source_rows,
        filtered_rows=len(filtered_df),
        applied_filters=[audit.to_dict() for audit in audits],
        filter_warnings=filter_warnings,
        analysis=analysis_dict,
    )
