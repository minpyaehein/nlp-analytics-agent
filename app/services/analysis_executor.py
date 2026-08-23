"""Deterministic Pandas execution engine for validated analysis plans."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import pandas as pd

from app.models.analysis_plan import (
    AnalysisIntent,
    AnalysisPlan,
    SortDirection,
)
from app.services.schema_linker import link_analysis_plan

from core.analytics_quality_gate import (
    require_profit_ready,
    require_revenue_ready,
)



@dataclass
class ExecutionResult:
    """Serializable result returned by the analytics execution engine."""

    success: bool
    intent: str
    metric_name: str | None
    dimension_name: str | None
    result: list[dict[str, Any]]
    row_count: int
    calculation: str | None
    warnings: list[str]
    validation: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Convert the execution result into a serializable dictionary."""

        return asdict(self)


def _make_serializable(value: Any) -> Any:
    """Convert Pandas values into JSON-friendly Python values."""

    if pd.isna(value):
        return None

    if isinstance(value, pd.Timestamp):
        return value.isoformat()

    if hasattr(value, "item"):
        try:
            return value.item()
        except (ValueError, AttributeError):
            pass

    return value


def dataframe_to_records(df: pd.DataFrame) -> list[dict[str, Any]]:
    """Convert a DataFrame to JSON-friendly row records."""

    records: list[dict[str, Any]] = []

    for row in df.to_dict(orient="records"):
        records.append(
            {
                str(key): _make_serializable(value)
                for key, value in row.items()
            }
        )

    return records


def _ensure_numeric(
    series: pd.Series,
    column_name: str,
) -> pd.Series:
    """Convert a source series to numeric and reject unusable columns."""

    numeric_series = pd.to_numeric(
        series,
        errors="coerce",
    )

    if numeric_series.notna().sum() == 0:
        raise ValueError(
            f"Column '{column_name}' does not contain usable numeric values."
        )

    return numeric_series


def create_metric_series(
    df: pd.DataFrame,
    metric_link: dict[str, Any],
) -> tuple[pd.Series, str]:
    """Create a numeric metric series from a direct or derived link."""

    match_type = metric_link["match_type"]
    metric_name = metric_link["semantic_field"]

    if match_type in {"exact", "alias"}:
        source_column = metric_link["source_column"]

        if source_column not in df.columns:
            raise ValueError(
                f"Metric column '{source_column}' does not exist."
            )

        metric_series = _ensure_numeric(
            df[source_column],
            source_column,
        )

        return metric_series, source_column

    if match_type == "derived":
        required_columns = metric_link.get("required_columns") or []

        missing_columns = [
            column
            for column in required_columns
            if column not in df.columns
        ]

        if missing_columns:
            raise ValueError(
                "Derived metric is missing source columns: "
                + ", ".join(missing_columns)
            )

        if metric_name == "revenue":

            require_revenue_ready(

                dataframe=df,

                minimum_usable_ratio=0.80,

                maximum_missing_ratio=0.20,

            )

            quantity_column, price_column = required_columns

            quantity = _ensure_numeric(
                df[quantity_column],
                quantity_column,
            )
            unit_price = _ensure_numeric(
                df[price_column],
                price_column,
            )

            return (
                quantity * unit_price,
                f"{quantity_column} * {price_column}",
            )

        if metric_name == "profit":

            require_profit_ready(

                dataframe=df,

                minimum_usable_ratio=0.80,

                maximum_missing_ratio=0.20,

            )

            quantity_column, price_column, cost_column = required_columns

            quantity = _ensure_numeric(
                df[quantity_column],
                quantity_column,
            )
            unit_price = _ensure_numeric(
                df[price_column],
                price_column,
            )
            unit_cost = _ensure_numeric(
                df[cost_column],
                cost_column,
            )

            return (
                quantity * (unit_price - unit_cost),
                (
                    f"{quantity_column} * "
                    f"({price_column} - {cost_column})"
                ),
            )

        raise ValueError(
            f"Unsupported derived metric: '{metric_name}'."
        )

    raise ValueError(
        f"Metric '{metric_name}' could not be resolved."
    )


def validate_execution(
    source_metric: pd.Series,
    result_df: pd.DataFrame,
    metric_name: str,
    intent: AnalysisIntent,
) -> dict[str, Any]:
    """Perform deterministic consistency checks on an analysis result."""

    validation: dict[str, Any] = {
        "status": "passed",
        "source_non_null_values": int(source_metric.notna().sum()),
        "result_rows": int(len(result_df)),
        "checks": [],
    }

    if result_df.empty:
        validation["status"] = "failed"
        validation["checks"].append(
            "The analysis produced an empty result."
        )
        return validation

    if metric_name in result_df.columns:
        result_numeric = pd.to_numeric(
            result_df[metric_name],
            errors="coerce",
        )

        if result_numeric.isna().all():
            validation["status"] = "failed"
            validation["checks"].append(
                "The calculated metric contains no valid numeric values."
            )
        else:
            validation["checks"].append(
                "The result metric contains valid numeric values."
            )

    if intent in {
        AnalysisIntent.RANKING,
        AnalysisIntent.COMPARISON,
        AnalysisIntent.TREND,
    }:
        source_total = float(source_metric.sum(skipna=True))

        if metric_name in result_df.columns:
            grouped_total = float(
                pd.to_numeric(
                    result_df[metric_name],
                    errors="coerce",
                ).sum(skipna=True)
            )

            validation["source_total"] = round(source_total, 6)
            validation["result_total"] = round(grouped_total, 6)

            if intent == AnalysisIntent.RANKING:
                validation["checks"].append(
                    "Ranking totals may be lower than the source total "
                    "when a result limit is applied."
                )
            elif abs(source_total - grouped_total) <= 1e-6:
                validation["checks"].append(
                    "Grouped metric total matches the source metric total."
                )
            else:
                validation["status"] = "warning"
                validation["checks"].append(
                    "Grouped total differs from the source total."
                )

    return validation


def execute_data_quality(
    df: pd.DataFrame,
    plan: AnalysisPlan,
) -> ExecutionResult:
    """Execute a deterministic data-quality analysis."""

    quality_rows: list[dict[str, Any]] = []

    for column in df.columns:
        quality_rows.append(
            {
                "column": str(column),
                "dtype": str(df[column].dtype),
                "missing_count": int(df[column].isna().sum()),
                "missing_percentage": round(
                    float(df[column].isna().mean() * 100),
                    2,
                ),
                "unique_count": int(df[column].nunique(dropna=True)),
            }
        )

    duplicate_count = int(df.duplicated().sum())

    return ExecutionResult(
        success=True,
        intent=plan.intent.value,
        metric_name=None,
        dimension_name=None,
        result=quality_rows,
        row_count=len(quality_rows),
        calculation=None,
        warnings=[],
        validation={
            "status": "passed",
            "dataset_rows": int(len(df)),
            "dataset_columns": int(len(df.columns)),
            "duplicate_rows": duplicate_count,
            "total_missing_values": int(df.isna().sum().sum()),
        },
    )


def execute_correlation(
    df: pd.DataFrame,
    plan: AnalysisPlan,
) -> ExecutionResult:
    """Calculate a numeric correlation matrix."""

    numeric_df = df.select_dtypes(include="number")

    if numeric_df.shape[1] < 2:
        raise ValueError(
            "Correlation requires at least two numeric columns."
        )

    correlation_matrix = numeric_df.corr(numeric_only=True)
    result_df = (
        correlation_matrix
        .rename_axis("variable")
        .reset_index()
    )

    return ExecutionResult(
        success=True,
        intent=plan.intent.value,
        metric_name=plan.metric,
        dimension_name=plan.dimension,
        result=dataframe_to_records(result_df),
        row_count=len(result_df),
        calculation="Pearson correlation matrix",
        warnings=[
            "Correlation indicates association, not causation."
        ],
        validation={
            "status": "passed",
            "numeric_columns": list(numeric_df.columns),
        },
    )


def execute_analysis_plan(
    df: pd.DataFrame,
    plan: AnalysisPlan,
) -> ExecutionResult:
    """Execute a validated analysis plan against a Pandas DataFrame."""

    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame")

    if df.empty:
        raise ValueError("The dataset contains no data rows.")

    if plan.intent == AnalysisIntent.DATA_QUALITY:
        return execute_data_quality(df, plan)

    if plan.intent == AnalysisIntent.CORRELATION:
        return execute_correlation(df, plan)

    schema_links = link_analysis_plan(
        metric=plan.metric,
        dimension=plan.dimension,
        df=df,
    )

    if not schema_links["resolved"]:
        raise ValueError(
            "The analysis plan could not be linked to the dataset: "
            + " ".join(schema_links["warnings"])
        )

    metric_link = schema_links["metric"]
    dimension_link = schema_links["dimension"]

    if metric_link is None:
        raise ValueError(
            "The analysis plan does not specify a metric."
        )

    working_df = df.copy()
    metric_name = str(plan.metric)

    metric_series, calculation = create_metric_series(
        working_df,
        metric_link,
    )
    working_df[metric_name] = metric_series

    warnings = list(schema_links["warnings"])

    if plan.intent == AnalysisIntent.SUMMARY:
        result_df = pd.DataFrame(
            [
                {
                    "metric": metric_name,
                    "aggregation": plan.aggregation,
                    "value": float(
                        working_df[metric_name].sum(skipna=True)
                    ),
                }
            ]
        )

        validation = {
            "status": "passed",
            "source_non_null_values": int(
                working_df[metric_name].notna().sum()
            ),
        }

        return ExecutionResult(
            success=True,
            intent=plan.intent.value,
            metric_name=metric_name,
            dimension_name=None,
            result=dataframe_to_records(result_df),
            row_count=len(result_df),
            calculation=calculation,
            warnings=warnings,
            validation=validation,
        )

    if dimension_link is None:
        raise ValueError(
            "This analytical intent requires a dimension."
        )

    dimension_column = dimension_link["source_column"]

    if dimension_column is None or dimension_column not in working_df.columns:
        raise ValueError(
            "The analysis dimension could not be resolved."
        )

    dimension_name = str(plan.dimension)

    if plan.intent == AnalysisIntent.DISTRIBUTION:
        numeric_values = working_df[metric_name].dropna()
        frequencies = pd.cut(
            numeric_values,
            bins=10,
            duplicates="drop",
        ).value_counts(sort=False)

        result_df = frequencies.rename("frequency").reset_index()
        result_df.columns = ["range", "frequency"]
        result_df["range"] = result_df["range"].astype(str)

    else:
        group_series = working_df[dimension_column]

        if plan.intent == AnalysisIntent.TREND:
            if dimension_name in {"month", "year", "order_date"}:
                date_candidates = [
                    dimension_column,
                    "order_date",
                    "date",
                ]
                date_column = next(
                    (
                        column
                        for column in date_candidates
                        if column in working_df.columns
                    ),
                    None,
                )

                if date_column is None:
                    raise ValueError(
                        "Trend analysis requires a date column."
                    )

                dates = pd.to_datetime(
                    working_df[date_column],
                    errors="coerce",
                )

                if dates.notna().sum() == 0:
                    raise ValueError(
                        f"Column '{date_column}' has no valid dates."
                    )

                if dimension_name == "year":
                    group_series = dates.dt.to_period("Y").astype(str)
                else:
                    group_series = dates.dt.to_period("M").astype(str)

        grouped_df = pd.DataFrame(
            {
                dimension_name: group_series,
                metric_name: working_df[metric_name],
            }
        )

        result_df = (
            grouped_df
            .dropna(subset=[dimension_name])
            .groupby(
                dimension_name,
                dropna=False,
                as_index=False,
            )[metric_name]
            .sum()
        )

        if plan.intent == AnalysisIntent.TREND:
            result_df = result_df.sort_values(
                by=dimension_name,
                ascending=True,
            )
        else:
            ascending = (
                plan.sort_direction == SortDirection.ASCENDING
            )
            result_df = result_df.sort_values(
                by=metric_name,
                ascending=ascending,
            )

        if plan.intent == AnalysisIntent.RANKING and plan.limit:
            result_df = result_df.head(plan.limit)

    validation = validate_execution(
        source_metric=working_df[metric_name],
        result_df=result_df,
        metric_name=metric_name,
        intent=plan.intent,
    )

    return ExecutionResult(
        success=validation["status"] != "failed",
        intent=plan.intent.value,
        metric_name=metric_name,
        dimension_name=dimension_name,
        result=dataframe_to_records(result_df),
        row_count=len(result_df),
        calculation=calculation,
        warnings=warnings,
        validation=validation,
    )
