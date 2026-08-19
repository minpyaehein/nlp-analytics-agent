"""Validated bilingual filter execution for InsightFlow AI."""

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


FILTER_VALUE_ALIASES = {
    "ရန်ကုန်": "Yangon",
    "ရန်ကုန်ဒေသ": "Yangon",
    "ရန်ကုန်တိုင်း": "Yangon",
    "ရန်ကုန်တိုင်းဒေသကြီး": "Yangon",
    "မန္တလေး": "Mandalay",
    "မန္တလေးဒေသ": "Mandalay",
    "မန္တလေးတိုင်း": "Mandalay",
    "မန္တလေးတိုင်းဒေသကြီး": "Mandalay",
    "နေပြည်တော်": "Naypyidaw",
    "နေပြည်တော်ဒေသ": "Naypyidaw",
    "ပဲခူး": "Bago",
    "ပဲခူးတိုင်း": "Bago",
    "ပဲခူးတိုင်းဒေသကြီး": "Bago",
    "မွန်": "Mon",
    "မွန်ပြည်နယ်": "Mon",
    "ကရင်": "Kayin",
    "ကရင်ပြည်နယ်": "Kayin",
    "ရှမ်း": "Shan",
    "ရှမ်းပြည်နယ်": "Shan",
    "ရခိုင်": "Rakhine",
    "ရခိုင်ပြည်နယ်": "Rakhine",
    "ဧရာဝတီ": "Ayeyarwady",
    "ဧရာဝတီတိုင်း": "Ayeyarwady",
    "ဧရာဝတီတိုင်းဒေသကြီး": "Ayeyarwady",
    "စစ်ကိုင်း": "Sagaing",
    "စစ်ကိုင်းတိုင်း": "Sagaing",
    "စစ်ကိုင်းတိုင်းဒေသကြီး": "Sagaing",
    "မကွေး": "Magway",
    "မကွေးတိုင်း": "Magway",
    "မကွေးတိုင်းဒေသကြီး": "Magway",
    "တနင်္သာရီ": "Tanintharyi",
    "တနင်္သာရီတိုင်း": "Tanintharyi",
    "တနင်္သာရီတိုင်းဒေသကြီး": "Tanintharyi",
    "ကချင်": "Kachin",
    "ကချင်ပြည်နယ်": "Kachin",
    "ချင်း": "Chin",
    "ချင်းပြည်နယ်": "Chin",
    "ကယား": "Kayah",
    "ကယားပြည်နယ်": "Kayah",
}


@dataclass
class FilterAudit:
    """Evidence describing how one filter changed the dataset."""

    requested_column: str
    source_column: str
    operator: str
    original_value: Any
    normalized_value: Any
    rows_before: int
    rows_after: int
    matched_rows: int

    def to_dict(self) -> dict[str, Any]:
        """Return a serializable filter-audit dictionary."""

        return asdict(self)


@dataclass
class FilteredExecutionResult:
    """Combined filter evidence and deterministic analysis output."""

    success: bool
    source_rows: int
    filtered_rows: int
    applied_filters: list[dict[str, Any]]
    filter_warnings: list[str]
    analysis: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Return a serializable execution-result dictionary."""

        return asdict(self)


def normalize_filter_value(value: Any) -> Any:
    """Normalize known Myanmar location values to canonical English values."""

    if isinstance(value, list):
        return [
            normalize_filter_value(item)
            for item in value
        ]

    if isinstance(value, tuple):
        return [
            normalize_filter_value(item)
            for item in value
        ]

    if not isinstance(value, str):
        return value

    cleaned_value = value.strip()

    return FILTER_VALUE_ALIASES.get(
        cleaned_value,
        cleaned_value,
    )


def _normalize_operator(operator: str) -> str:
    """Normalize common operator aliases to approved operator names."""

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
    dataframe: pd.DataFrame,
    requested_column: str,
) -> str:
    """Resolve a semantic filter field to an actual dataset column."""

    if requested_column in dataframe.columns:
        return requested_column

    casefold_map = {
        str(column).strip().casefold(): str(column)
        for column in dataframe.columns
    }

    direct_match = casefold_map.get(
        str(requested_column).strip().casefold()
    )

    if direct_match is not None:
        return direct_match

    link = link_field(
        semantic_field=requested_column,
        df=dataframe,
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
    """Perform trimmed, case-insensitive and null-safe text equality."""

    return (
        series.astype("string")
        .str.strip()
        .str.casefold()
        .eq(str(value).strip().casefold())
        .fillna(False)
    )


def _numeric_series(
    series: pd.Series,
    column_name: str,
) -> pd.Series:
    """Convert a series to numeric and reject unusable columns."""

    numeric = pd.to_numeric(series, errors="coerce")

    if numeric.notna().sum() == 0:
        raise ValueError(
            f"Column '{column_name}' does not contain usable "
            "numeric values."
        )

    return numeric


def _build_filter_mask(
    dataframe: pd.DataFrame,
    source_column: str,
    operator: str,
    value: Any,
) -> pd.Series:
    """Build a Boolean mask for one approved filter operation."""

    series = dataframe[source_column]

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

        return numeric.between(
            lower,
            upper,
            inclusive="both",
        ).fillna(False)

    raise ValueError(
        f"Unsupported filter operator: '{operator}'."
    )


def get_available_values(
    dataframe: pd.DataFrame,
    source_columns: list[str],
    maximum_values: int = 25,
) -> dict[str, list[str]]:
    """Return sample category values for useful no-match error messages."""

    available_values: dict[str, list[str]] = {}

    for source_column in source_columns:
        if source_column not in dataframe.columns:
            continue

        values = (
            dataframe[source_column]
            .dropna()
            .astype(str)
            .str.strip()
            .drop_duplicates()
            .head(maximum_values)
            .tolist()
        )

        available_values[source_column] = values

    return available_values


def apply_plan_filters(
    dataframe: pd.DataFrame,
    filters: list[FilterCondition],
) -> tuple[pd.DataFrame, list[FilterAudit], list[str]]:
    """Apply validated filters sequentially and produce audit evidence."""

    if not isinstance(dataframe, pd.DataFrame):
        raise TypeError("dataframe must be a pandas DataFrame")

    filtered_dataframe = dataframe.copy()
    audits: list[FilterAudit] = []
    warnings: list[str] = []

    for condition in filters:
        operator = _normalize_operator(condition.operator)
        source_column = _resolve_filter_column(
            filtered_dataframe,
            condition.column,
        )
        normalized_value = normalize_filter_value(
            condition.value
        )

        rows_before = len(filtered_dataframe)

        mask = _build_filter_mask(
            filtered_dataframe,
            source_column,
            operator,
            normalized_value,
        )

        matched_rows = int(mask.sum())
        filtered_dataframe = filtered_dataframe.loc[mask].copy()
        rows_after = len(filtered_dataframe)

        audits.append(
            FilterAudit(
                requested_column=condition.column,
                source_column=source_column,
                operator=operator,
                original_value=condition.value,
                normalized_value=normalized_value,
                rows_before=rows_before,
                rows_after=rows_after,
                matched_rows=matched_rows,
            )
        )

        if matched_rows == 0:
            warnings.append(
                "Filter matched no rows: "
                f"{condition.column} {operator} "
                f"{normalized_value}."
            )

    return filtered_dataframe, audits, warnings


def execute_filtered_analysis(
    dataframe: pd.DataFrame,
    plan: AnalysisPlan,
) -> FilteredExecutionResult:
    """Apply plan filters and run deterministic analysis on matching rows."""

    if not isinstance(dataframe, pd.DataFrame):
        raise TypeError("dataframe must be a pandas DataFrame")

    if dataframe.empty:
        raise ValueError("The dataset contains no data rows.")

    source_rows = len(dataframe)

    filtered_dataframe, audits, filter_warnings = apply_plan_filters(
        dataframe,
        plan.filters,
    )

    if filtered_dataframe.empty:
        available_values = get_available_values(
            dataframe,
            [audit.source_column for audit in audits],
        )

        requested_filters = [
            {
                "column": audit.requested_column,
                "operator": audit.operator,
                "original_value": audit.original_value,
                "normalized_value": audit.normalized_value,
            }
            for audit in audits
        ]

        raise ValueError(
            "No dataset rows remained after applying the requested "
            f"filters. Requested filters: {requested_filters}. "
            f"Available values: {available_values}."
        )

    execution_plan = plan.model_copy(
        update={"filters": []}
    )

    analysis_result: ExecutionResult = execute_analysis_plan(
        filtered_dataframe,
        execution_plan,
    )

    analysis_dict = analysis_result.to_dict()
    validation = analysis_dict.setdefault("validation", {})
    validation["source_rows_before_filters"] = source_rows
    validation["rows_after_filters"] = len(filtered_dataframe)
    validation["applied_filter_count"] = len(audits)
    validation["filters_passed"] = all(
        audit.matched_rows > 0
        for audit in audits
    )

    return FilteredExecutionResult(
        success=analysis_result.success,
        source_rows=source_rows,
        filtered_rows=len(filtered_dataframe),
        applied_filters=[
            audit.to_dict()
            for audit in audits
        ],
        filter_warnings=filter_warnings,
        analysis=analysis_dict,
    )
