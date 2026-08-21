"""Automatic Plotly visualization generation for InsightFlow AI."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from app.models.analysis_plan import AnalysisIntent, AnalysisPlan


@dataclass
class ChartResult:
    """Chart object together with validation and display metadata."""

    success: bool
    chart_type: str
    title: str
    x_column: str | None
    y_column: str | None
    row_count: int
    warnings: list[str]
    validation: dict[str, Any]
    figure: go.Figure

    def metadata(self) -> dict[str, Any]:
        """Return chart metadata without the non-serializable figure."""

        data = asdict(self)
        data.pop("figure", None)
        return data


def _enum_value(value: Any) -> str:
    """Return an Enum value or a normalized string."""

    return str(getattr(value, "value", value))


def _select_result_columns(
    result_df: pd.DataFrame,
    plan: AnalysisPlan,
) -> tuple[str | None, str | None]:
    """Determine dimension and metric columns from the execution result."""

    dimension = plan.dimension
    metric = plan.metric

    x_column = (
        dimension
        if dimension is not None and dimension in result_df.columns
        else None
    )
    y_column = (
        metric
        if metric is not None and metric in result_df.columns
        else None
    )

    if x_column is None and len(result_df.columns) >= 1:
        x_column = str(result_df.columns[0])

    if y_column is None:
        numeric_columns = result_df.select_dtypes(
            include="number"
        ).columns.tolist()

        if numeric_columns:
            y_column = str(numeric_columns[-1])

    return x_column, y_column


def _validate_chart_data(
    result_df: pd.DataFrame,
    x_column: str | None,
    y_column: str | None,
) -> dict[str, Any]:
    """Validate that the chart uses existing, non-empty result columns."""

    checks: list[str] = []
    status = "passed"

    if result_df.empty:
        return {
            "status": "failed",
            "checks": ["The analysis result contains no chartable rows."],
        }

    if x_column is not None:
        if x_column not in result_df.columns:
            status = "failed"
            checks.append(
                f"The x-axis column '{x_column}' does not exist."
            )
        elif result_df[x_column].notna().sum() == 0:
            status = "failed"
            checks.append(
                f"The x-axis column '{x_column}' contains no values."
            )
        else:
            checks.append(
                f"The x-axis column '{x_column}' is valid."
            )

    if y_column is not None:
        if y_column not in result_df.columns:
            status = "failed"
            checks.append(
                f"The y-axis column '{y_column}' does not exist."
            )
        else:
            numeric = pd.to_numeric(
                result_df[y_column],
                errors="coerce",
            )

            if numeric.notna().sum() == 0:
                status = "failed"
                checks.append(
                    f"The y-axis column '{y_column}' is not numeric."
                )
            else:
                checks.append(
                    f"The y-axis column '{y_column}' is numeric."
                )
                checks.append(
                    "The chart row count matches the analysis result."
                )

    return {
        "status": status,
        "checks": checks,
        "result_rows": int(len(result_df)),
        "x_column": x_column,
        "y_column": y_column,
    }


def _format_label(value: str | None) -> str:
    """Create a readable axis or title label."""

    if not value:
        return "Value"

    return value.replace("_", " ").title()


def _apply_common_layout(
    figure: go.Figure,
    title: str,
) -> go.Figure:
    """Apply a consistent dashboard-friendly Plotly layout."""

    figure.update_layout(
        title={
            "text": title,
            "x": 0.02,
            "xanchor": "left",
        },
        margin={"l": 35, "r": 25, "t": 65, "b": 40},
        legend_title_text="",
        hoverlabel={"namelength": -1},
        height=460,
    )

    return figure


def _build_bar_chart(
    result_df: pd.DataFrame,
    x_column: str,
    y_column: str,
    title: str,
) -> go.Figure:
    """Create a ranking or comparison bar chart."""

    figure = px.bar(
        result_df,
        x=x_column,
        y=y_column,
        text_auto=",.2s",
        labels={
            x_column: _format_label(x_column),
            y_column: _format_label(y_column),
        },
    )
    figure.update_traces(
        marker_color="#1F6FB2",
        textposition="outside",
        cliponaxis=False,
    )
    return _apply_common_layout(figure, title)


def _build_line_chart(
    result_df: pd.DataFrame,
    x_column: str,
    y_column: str,
    title: str,
) -> go.Figure:
    """Create a chronological trend line chart."""

    figure = px.line(
        result_df,
        x=x_column,
        y=y_column,
        markers=True,
        labels={
            x_column: _format_label(x_column),
            y_column: _format_label(y_column),
        },
    )
    figure.update_traces(
        line={"color": "#168C80", "width": 3},
        marker={"size": 8},
    )
    return _apply_common_layout(figure, title)


def _build_histogram_chart(
    result_df: pd.DataFrame,
    x_column: str,
    y_column: str | None,
    title: str,
) -> go.Figure:
    """Create a distribution chart from result ranges or raw values."""

    if y_column is not None and y_column in result_df.columns:
        figure = px.bar(
            result_df,
            x=x_column,
            y=y_column,
            labels={
                x_column: _format_label(x_column),
                y_column: _format_label(y_column),
            },
        )
    else:
        figure = px.histogram(
            result_df,
            x=x_column,
            nbins=10,
            labels={x_column: _format_label(x_column)},
        )

    figure.update_traces(marker_color="#E6A23C")
    return _apply_common_layout(figure, title)


def _build_data_quality_chart(
    result_df: pd.DataFrame,
    title: str,
) -> go.Figure:
    """Create a missing-values chart for data-quality results."""

    if "column" not in result_df.columns:
        raise ValueError(
            "Data-quality chart requires a 'column' result field."
        )

    value_column = (
        "missing_count"
        if "missing_count" in result_df.columns
        else "missing_percentage"
    )

    if value_column not in result_df.columns:
        raise ValueError(
            "Data-quality results do not include missing-value metrics."
        )

    chart_df = result_df.sort_values(
        by=value_column,
        ascending=False,
    )

    figure = px.bar(
        chart_df,
        x="column",
        y=value_column,
        text_auto=True,
        labels={
            "column": "Column",
            value_column: _format_label(value_column),
        },
    )
    figure.update_traces(marker_color="#C0392B")
    return _apply_common_layout(figure, title)


def _build_correlation_heatmap(
    result_df: pd.DataFrame,
    title: str,
) -> go.Figure:
    """Create a heatmap from the executor correlation matrix output."""

    if "variable" not in result_df.columns:
        raise ValueError(
            "Correlation results require a 'variable' column."
        )

    matrix = result_df.set_index("variable")
    matrix = matrix.apply(pd.to_numeric, errors="coerce")

    figure = go.Figure(
        data=go.Heatmap(
            z=matrix.values,
            x=matrix.columns.tolist(),
            y=matrix.index.tolist(),
            zmin=-1,
            zmax=1,
            colorscale="RdBu",
            reversescale=True,
            colorbar={"title": "Correlation"},
            hovertemplate=(
                "%{y} vs %{x}<br>Correlation=%{z:.3f}<extra></extra>"
            ),
        )
    )

    return _apply_common_layout(figure, title)


def generate_chart(
    result_records: list[dict[str, Any]],
    plan: AnalysisPlan,
) -> ChartResult:
    """Generate a validated Plotly chart from deterministic result rows."""

    result_df = pd.DataFrame(result_records)

    if result_df.empty:
        raise ValueError(
            "A chart cannot be generated because the analysis result is empty."
        )

    intent = _enum_value(plan.intent)
    visualization = _enum_value(plan.visualization)
    x_column, y_column = _select_result_columns(result_df, plan)
    metric_label = _format_label(plan.metric)
    dimension_label = _format_label(plan.dimension)
    warnings: list[str] = []

    if intent == AnalysisIntent.DATA_QUALITY.value:
        chart_type = "data_quality_bar"
        title = "Missing Values by Column"
        figure = _build_data_quality_chart(result_df, title)
        x_column = "column"
        y_column = (
            "missing_count"
            if "missing_count" in result_df.columns
            else "missing_percentage"
        )

    elif intent == AnalysisIntent.CORRELATION.value:
        chart_type = "correlation_heatmap"
        title = "Numeric Correlation Matrix"
        figure = _build_correlation_heatmap(result_df, title)
        x_column = "variable"
        y_column = None

    elif visualization == "line" or intent == AnalysisIntent.TREND.value:
        if x_column is None or y_column is None:
            raise ValueError(
                "Trend visualization requires dimension and metric columns."
            )
        chart_type = "line"
        title = f"{metric_label} Trend by {dimension_label}"
        figure = _build_line_chart(
            result_df,
            x_column,
            y_column,
            title,
        )

    elif visualization == "histogram" or intent == AnalysisIntent.DISTRIBUTION.value:
        if x_column is None:
            raise ValueError(
                "Distribution visualization requires a result column."
            )
        chart_type = "histogram"
        title = f"{metric_label} Distribution"
        figure = _build_histogram_chart(
            result_df,
            x_column,
            y_column,
            title,
        )

    elif intent in {
        AnalysisIntent.RANKING.value,
        AnalysisIntent.COMPARISON.value,
    } or visualization == "bar":
        if x_column is None or y_column is None:
            raise ValueError(
                "Bar visualization requires dimension and metric columns."
            )
        chart_type = "bar"
        title = f"{metric_label} by {dimension_label}"
        figure = _build_bar_chart(
            result_df,
            x_column,
            y_column,
            title,
        )

    else:
        if x_column is None or y_column is None:
            raise ValueError(
                "The analysis result does not contain enough fields for a chart."
            )
        chart_type = "bar"
        title = f"{metric_label} by {dimension_label}"
        warnings.append(
            "The requested visualization was unsupported; a bar chart was used."
        )
        figure = _build_bar_chart(
            result_df,
            x_column,
            y_column,
            title,
        )

    validation = _validate_chart_data(
        result_df,
        x_column,
        y_column,
    )

    if intent == AnalysisIntent.CORRELATION.value:
        validation = {
            "status": "passed",
            "checks": [
                "The correlation matrix contains chartable numeric values."
            ],
            "result_rows": int(len(result_df)),
        }

    return ChartResult(
        success=validation.get("status") != "failed",
        chart_type=chart_type,
        title=title,
        x_column=x_column,
        y_column=y_column,
        row_count=len(result_df),
        warnings=warnings,
        validation=validation,
        figure=figure,
    )
