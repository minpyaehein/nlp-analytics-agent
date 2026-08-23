"""InsightFlow AI Streamlit application with Power BI publication.

Run from the repository root:
    python -m streamlit run app/frontend/streamlit_app.py

Required project modules:
    app.agents.ai_planner
    app.services.powerbi_exporter

The app keeps financial calculations deterministic. Qwen creates only the
analysis plan; Pandas calculates results after schema and quality validation.
"""

from __future__ import annotations

import hashlib
import io
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd
import plotly.express as px
import streamlit as st

from app.agents.ai_planner import AIPlannerResult, create_ai_plan
from app.services.powerbi_exporter import (
    PowerBIPublication,
    PowerBIQualityEvidence,
    PowerBIResponse,
    PowerBIResultRow,
    PowerBISourceFile,
    publish_to_powerbi_workbook,
)

try:
    from core.unified_file_loader import process_uploaded_file
except ImportError:
    process_uploaded_file = None


POWERBI_PATH = Path("powerbi_output") / "insightflow_powerbi.xlsx"
SUPPORTED_EXTENSIONS = ["csv", "xlsx", "xls", "json", "txt", "tsv", "docx", "pdf"]


@dataclass
class LoadedDataset:
    dataframe: pd.DataFrame
    evidence: dict[str, Any]
    source_bytes: bytes


@dataclass
class ExecutionResult:
    result_frame: pd.DataFrame
    result_rows: list[PowerBIResultRow]
    filtered_frame: pd.DataFrame
    quality: PowerBIQualityEvidence
    answer: str
    primary_value: float | None


def _enum_text(value: Any) -> str | None:
    if value is None:
        return None
    return str(getattr(value, "value", value))


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _uploaded_bytes(uploaded_file: Any) -> bytes:
    if hasattr(uploaded_file, "getvalue"):
        return uploaded_file.getvalue()
    value = uploaded_file.read()
    uploaded_file.seek(0)
    return value


def _extract_loader_result(result: Any) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Accept DataFrame, tuple, mapping, or project result object."""

    if isinstance(result, pd.DataFrame):
        return result, {}

    if isinstance(result, tuple):
        dataframe = next((item for item in result if isinstance(item, pd.DataFrame)), None)
        evidence = next((item for item in result if isinstance(item, Mapping)), {})
        if dataframe is not None:
            return dataframe, dict(evidence)

    if isinstance(result, Mapping):
        for key in ("dataframe", "df", "table", "data"):
            if isinstance(result.get(key), pd.DataFrame):
                evidence = dict(result.get("evidence") or result.get("processing_evidence") or {})
                return result[key], evidence

    for attribute in ("dataframe", "df", "table", "data"):
        dataframe = getattr(result, attribute, None)
        if isinstance(dataframe, pd.DataFrame):
            evidence = getattr(result, "evidence", None)
            if evidence is None:
                evidence = getattr(result, "processing_evidence", {})
            if hasattr(evidence, "model_dump"):
                evidence = evidence.model_dump(mode="json")
            return dataframe, dict(evidence or {})

    raise TypeError("The unified loader did not return a recognizable DataFrame result.")


def load_uploaded_dataset(uploaded_file: Any) -> LoadedDataset:
    data = _uploaded_bytes(uploaded_file)
    suffix = Path(uploaded_file.name).suffix.casefold()
    evidence: dict[str, Any] = {
        "filename": uploaded_file.name,
        "file_type": suffix.lstrip("."),
        "file_size_bytes": len(data),
        "source_sha256": _sha256(data),
        "ocr_executed": False,
    }

    if suffix == ".csv":
        frame = pd.read_csv(io.BytesIO(data))
    elif suffix in {".xlsx", ".xls"}:
        engine = "openpyxl" if suffix == ".xlsx" else "xlrd"
        frame = pd.read_excel(io.BytesIO(data), engine=engine)
    elif suffix == ".json":
        frame = pd.read_json(io.BytesIO(data))
    elif suffix in {".txt", ".tsv"}:
        delimiter = "\t" if suffix == ".tsv" else None
        frame = pd.read_csv(io.BytesIO(data), sep=delimiter, engine="python")
    else:
        if process_uploaded_file is None:
            raise RuntimeError(
                "DOCX/PDF loading requires core.unified_file_loader.process_uploaded_file()."
            )
        frame, loader_evidence = _extract_loader_result(
            process_uploaded_file(
                data,
                uploaded_file.name,
            )
        )
        evidence.update(loader_evidence)

    frame.columns = [str(column).strip() for column in frame.columns]
    frame = frame.dropna(how="all").reset_index(drop=True)
    evidence.setdefault("row_count", int(len(frame)))
    evidence.setdefault("column_count", int(len(frame.columns)))
    evidence.setdefault("duplicate_rows", int(frame.duplicated().sum()))
    evidence.setdefault("missing_values", int(frame.isna().sum().sum()))
    return LoadedDataset(frame, evidence, data)


def _column_map(frame: pd.DataFrame) -> dict[str, str]:
    return {str(column).strip().casefold(): str(column) for column in frame.columns}


def _derived_metric(frame: pd.DataFrame, metric: str) -> tuple[pd.Series, tuple[str, ...]]:
    columns = _column_map(frame)
    metric_key = metric.casefold()

    if metric_key in columns:
        column = columns[metric_key]
        return pd.to_numeric(frame[column], errors="coerce"), (column,)

    if metric_key == "revenue":
        required = ("quantity", "unit_price")
        if not all(name in columns for name in required):
            raise ValueError("Revenue requires quantity and unit_price columns.")
        quantity = pd.to_numeric(frame[columns["quantity"]], errors="coerce")
        unit_price = pd.to_numeric(frame[columns["unit_price"]], errors="coerce")
        return quantity * unit_price, (columns["quantity"], columns["unit_price"])

    if metric_key == "profit":
        required = ("quantity", "unit_price", "unit_cost")
        if not all(name in columns for name in required):
            raise ValueError("Profit requires quantity, unit_price, and unit_cost columns.")
        quantity = pd.to_numeric(frame[columns["quantity"]], errors="coerce")
        unit_price = pd.to_numeric(frame[columns["unit_price"]], errors="coerce")
        unit_cost = pd.to_numeric(frame[columns["unit_cost"]], errors="coerce")
        return quantity * (unit_price - unit_cost), (
            columns["quantity"], columns["unit_price"], columns["unit_cost"]
        )

    raise ValueError(f"Metric '{metric}' is not grounded to the dataset.")


def _filter_value(series: pd.Series, operator: str, value: Any) -> pd.Series:
    if operator == "equals":
        return series.astype(str).str.casefold() == str(value).casefold()
    if operator == "not_equals":
        return series.astype(str).str.casefold() != str(value).casefold()
    if operator == "contains":
        return series.astype(str).str.contains(str(value), case=False, na=False, regex=False)
    if operator == "in":
        values = value if isinstance(value, Sequence) and not isinstance(value, str) else [value]
        normalized = {str(item).casefold() for item in values}
        return series.astype(str).str.casefold().isin(normalized)

    numeric = pd.to_numeric(series, errors="coerce")
    if operator == "greater_than":
        return numeric > float(value)
    if operator == "greater_than_or_equal":
        return numeric >= float(value)
    if operator == "less_than":
        return numeric < float(value)
    if operator == "less_than_or_equal":
        return numeric <= float(value)
    if operator == "between":
        low, high = value
        return numeric.between(float(low), float(high), inclusive="both")
    raise ValueError(f"Unsupported filter operator: {operator}")


def apply_plan_filters(frame: pd.DataFrame, filters: Sequence[Any]) -> pd.DataFrame:
    filtered = frame.copy()
    columns = _column_map(filtered)
    for condition in filters:
        condition_data = (
            condition.model_dump(mode="python")
            if hasattr(condition, "model_dump")
            else dict(condition)
        )
        column_key = str(condition_data["column"]).casefold()
        if column_key not in columns:
            raise ValueError(f"Filter column '{condition_data['column']}' does not exist.")
        column = columns[column_key]
        mask = _filter_value(
            filtered[column],
            str(condition_data.get("operator", "equals")),
            condition_data.get("value"),
        )
        filtered = filtered.loc[mask].copy()
    return filtered.reset_index(drop=True)


def validate_metric_quality(
    frame: pd.DataFrame,
    metric_values: pd.Series,
    required_columns: Sequence[str],
    metric: str,
) -> PowerBIQualityEvidence:
    total_rows = len(frame)
    usable_rows = int(metric_values.notna().sum())
    usable_ratio = usable_rows / total_rows if total_rows else 0.0
    ready = total_rows > 0 and usable_ratio >= 0.80
    warnings: list[str] = []
    duplicate_rows = int(frame.duplicated().sum())
    missing_values = int(frame[list(required_columns)].isna().sum().sum())

    if duplicate_rows:
        warnings.append(f"{duplicate_rows} duplicate source row(s) were retained.")
    if usable_ratio < 1.0:
        warnings.append(
            f"Only {usable_ratio:.1%} of rows contain usable numeric evidence for {metric}."
        )

    return PowerBIQualityEvidence(
        gate_name=metric,
        ready=ready,
        duplicate_rows=duplicate_rows,
        missing_values=missing_values,
        usable_numeric_ratio=usable_ratio,
        validation_status="passed" if ready else "blocked",
        warnings=tuple(warnings),
        evidence={
            "required_columns": list(required_columns),
            "source_rows": total_rows,
            "usable_rows": usable_rows,
        },
    )


def execute_validated_plan(frame: pd.DataFrame, plan: Any) -> ExecutionResult:
    if bool(getattr(plan, "requires_clarification", False)):
        raise ValueError(getattr(plan, "clarification_question", "Clarification is required."))

    filtered = apply_plan_filters(frame, getattr(plan, "filters", []))
    metric = _enum_text(getattr(plan, "metric", None))
    if not metric:
        raise ValueError("The plan does not contain a metric.")

    metric_values, required_columns = _derived_metric(filtered, metric)
    quality = validate_metric_quality(filtered, metric_values, required_columns, metric)
    if not quality.ready:
        raise ValueError(
            f"{metric.title()} calculation was blocked because the current DataFrame "
            "does not contain enough usable numeric evidence."
        )

    working = filtered.copy()
    working["__metric_value__"] = metric_values
    intent = _enum_text(getattr(plan, "intent", "summary")) or "summary"
    aggregation = _enum_text(getattr(plan, "aggregation", "sum")) or "sum"
    dimension = _enum_text(getattr(plan, "dimension", None))

    aggregate_function = {
        "sum": "sum",
        "mean": "mean",
        "count": "count",
        "min": "min",
        "max": "max",
    }.get(aggregation)
    if not aggregate_function:
        raise ValueError(f"Unsupported aggregation: {aggregation}")

    if intent == "summary":
        value = getattr(working["__metric_value__"], aggregate_function)()
        result_frame = pd.DataFrame(
            [{"label": f"Total {metric}", "value": float(value), "rank": None}]
        )
        answer = f"Validated {aggregation} {metric}: {float(value):,.2f}."
        primary_value = float(value)
    elif intent in {"ranking", "comparison", "trend"}:
        if not dimension:
            raise ValueError(f"Intent '{intent}' requires a dimension.")
        columns = _column_map(working)
        dimension_key = dimension.casefold()
        if dimension_key not in columns:
            if intent == "trend" and "order_date" in columns:
                dimension_key = "order_date"
            else:
                raise ValueError(f"Dimension '{dimension}' does not exist in the dataset.")
        dimension_column = columns[dimension_key]
        grouped = (
            working.groupby(dimension_column, dropna=False)["__metric_value__"]
            .agg(aggregate_function)
            .reset_index()
        )
        grouped.columns = ["label", "value"]
        ascending = _enum_text(getattr(plan, "sort_direction", None)) == "ascending"
        if intent == "trend":
            grouped = grouped.sort_values("label", ascending=True)
        else:
            grouped = grouped.sort_values("value", ascending=ascending)
        limit = getattr(plan, "limit", None)
        if limit:
            grouped = grouped.head(int(limit))
        grouped = grouped.reset_index(drop=True)
        grouped["rank"] = range(1, len(grouped) + 1)
        result_frame = grouped
        top = result_frame.iloc[0]
        answer = f"Top validated {dimension}: {top['label']} with {float(top['value']):,.2f} {metric}."
        primary_value = float(top["value"])
    elif intent == "distribution":
        values = working["__metric_value__"].dropna()
        counts, edges = pd.cut(values, bins=min(10, max(1, values.nunique())), duplicates="drop", retbins=True)
        result_frame = counts.value_counts(sort=False).rename_axis("label").reset_index(name="value")
        result_frame["label"] = result_frame["label"].astype(str)
        result_frame["rank"] = range(1, len(result_frame) + 1)
        answer = f"Generated a validated distribution for {metric}."
        primary_value = None
    else:
        raise ValueError(f"Intent '{intent}' is not supported by this Streamlit executor.")

    result_rows = [
        PowerBIResultRow(
            label=str(row["label"]),
            value=float(row["value"]),
            rank=int(row["rank"]) if pd.notna(row.get("rank")) else None,
        )
        for row in result_frame.to_dict(orient="records")
    ]
    return ExecutionResult(result_frame, result_rows, filtered, quality, answer, primary_value)


def render_chart(result: ExecutionResult, plan: Any) -> None:
    visualization = _enum_text(getattr(plan, "visualization", "table")) or "table"
    metric = _enum_text(getattr(plan, "metric", "value")) or "value"

    if visualization == "kpi" and result.primary_value is not None:
        st.metric(f"Validated {metric.title()}", f"{result.primary_value:,.2f}")
    elif visualization in {"bar", "histogram"}:
        figure = px.bar(
            result.result_frame,
            x="label",
            y="value",
            title=f"Validated {metric.title()} Analysis",
        )
        st.plotly_chart(figure, use_container_width=True)
    elif visualization == "line":
        figure = px.line(
            result.result_frame,
            x="label",
            y="value",
            markers=True,
            title=f"Validated {metric.title()} Trend",
        )
        st.plotly_chart(figure, use_container_width=True)

    st.dataframe(result.result_frame, use_container_width=True, hide_index=True)


def source_file_record(loaded: LoadedDataset) -> PowerBISourceFile:
    evidence = loaded.evidence
    return PowerBISourceFile(
        filename=str(evidence.get("filename", "")),
        file_type=str(evidence.get("file_type", "")),
        file_size_bytes=evidence.get("file_size_bytes"),
        page_count=evidence.get("page_count"),
        ocr_executed=bool(evidence.get("ocr_executed", False)),
        ocr_languages=str(evidence.get("ocr_languages", "")),
        extraction_strategy=str(evidence.get("extraction_strategy", "")),
        extraction_confidence=evidence.get("extraction_confidence"),
        source_sha256=str(evidence.get("source_sha256", "")),
    )


def publish_current_result(
    question: str,
    planner_result: AIPlannerResult,
    execution: ExecutionResult,
    loaded: LoadedDataset,
) -> Any:
    plan = planner_result.plan
    tool_steps = [
        step.model_dump(mode="json") if hasattr(step, "model_dump") else dict(step)
        for step in getattr(plan, "tool_steps", [])
    ]
    publication = PowerBIPublication(
        question=question,
        planner_source=planner_result.source,
        model_name=planner_result.model,
        intent=_enum_text(plan.intent) or "unknown",
        metric=_enum_text(plan.metric),
        dimension=_enum_text(plan.dimension),
        aggregation=_enum_text(plan.aggregation) or "sum",
        sort_direction=_enum_text(plan.sort_direction),
        result_limit=plan.limit,
        visualization=_enum_text(plan.visualization) or "table",
        confidence=plan.confidence,
        source_rows=len(loaded.dataframe),
        filtered_rows=len(execution.filtered_frame),
        validation_status=execution.quality.validation_status,
        quality_ready=execution.quality.ready,
        results=tuple(execution.result_rows),
        ai_response=PowerBIResponse(
            language=_enum_text(plan.language) or "en",
            answer=execution.answer,
            primary_finding=execution.answer,
            calculation=(
                "Revenue = quantity * unit_price"
                if _enum_text(plan.metric) == "revenue"
                else "Profit = quantity * (unit_price - unit_cost)"
                if _enum_text(plan.metric) == "profit"
                else f"{_enum_text(plan.aggregation)}({_enum_text(plan.metric)})"
            ),
            limitations="; ".join(execution.quality.warnings),
        ),
        quality=(execution.quality,),
        source_files=(source_file_record(loaded),),
        reasoning_summary=str(plan.reasoning_summary),
        assumptions=tuple(plan.assumptions),
        tool_steps=tuple(tool_steps),
    )
    return publish_to_powerbi_workbook(publication, POWERBI_PATH)


def initialize_state() -> None:
    defaults = {
        "loaded_dataset": None,
        "planner_result": None,
        "execution_result": None,
        "current_question": "",
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def main() -> None:
    st.set_page_config(
        page_title="InsightFlow AI",
        page_icon="📊",
        layout="wide",
    )
    initialize_state()

    st.title("InsightFlow AI")
    st.caption(
        "Bilingual document analytics with local Qwen planning, deterministic "
        "execution, OCR quality gates, and Power BI publication."
    )

    with st.sidebar:
        st.header("Data Source")
        uploaded_file = st.file_uploader(
            "Upload a data or document file",
            type=SUPPORTED_EXTENSIONS,
        )
        if uploaded_file and st.button("Process file", use_container_width=True):
            try:
                with st.spinner("Extracting and validating the uploaded file..."):
                    st.session_state.loaded_dataset = load_uploaded_dataset(uploaded_file)
                    st.session_state.planner_result = None
                    st.session_state.execution_result = None
                st.success("File processed.")
            except Exception as error:
                st.error(f"File processing failed: {error}")

        st.divider()
        st.header("Power BI")
        st.code(str(POWERBI_PATH), language=None)
        if POWERBI_PATH.exists():
            st.download_button(
                "Download Power BI workbook",
                data=POWERBI_PATH.read_bytes(),
                file_name=POWERBI_PATH.name,
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True,
            )

    loaded: LoadedDataset | None = st.session_state.loaded_dataset
    if loaded is None:
        st.info("Upload a supported file and select Process file to begin.")
        return

    profile_col, evidence_col = st.columns([2, 1])
    with profile_col:
        st.subheader("Dataset Preview")
        st.dataframe(loaded.dataframe.head(100), use_container_width=True, hide_index=True)
    with evidence_col:
        st.subheader("Processing Evidence")
        st.metric("Rows", len(loaded.dataframe))
        st.metric("Columns", len(loaded.dataframe.columns))
        st.metric("Duplicate rows", int(loaded.dataframe.duplicated().sum()))
        with st.expander("Evidence JSON"):
            st.json(loaded.evidence)

    st.divider()
    st.subheader("Ask InsightFlow AI")
    question = st.text_input(
        "English, Myanmar, or mixed-language analytical question",
        value=st.session_state.current_question,
        placeholder="Show the top 5 products by revenue.",
    )

    if st.button("Analyze", type="primary"):
        try:
            with st.spinner("Creating a grounded plan and calculating deterministically..."):
                planner_result = create_ai_plan(question, loaded.dataframe)
                if planner_result.plan.requires_clarification:
                    st.session_state.planner_result = planner_result
                    st.session_state.execution_result = None
                else:
                    execution = execute_validated_plan(loaded.dataframe, planner_result.plan)
                    st.session_state.planner_result = planner_result
                    st.session_state.execution_result = execution
                st.session_state.current_question = question
        except Exception as error:
            st.error(f"Analysis failed: {error}")

    planner_result: AIPlannerResult | None = st.session_state.planner_result
    execution: ExecutionResult | None = st.session_state.execution_result
    if planner_result is None:
        return

    plan = planner_result.plan
    if plan.requires_clarification:
        st.warning(plan.clarification_question or "Please clarify the analytical request.")
        return

    st.success(execution.answer if execution else "Validated analysis completed.")
    plan_col, quality_col = st.columns(2)
    with plan_col:
        st.subheader("Grounded Plan")
        st.json(planner_result.metadata())
    with quality_col:
        st.subheader("Quality Gate")
        st.json(execution.quality.__dict__ if execution else {})

    if execution:
        render_chart(execution, plan)

        st.divider()
        st.subheader("Power BI Publication")
        st.caption(
            "Only the deterministic result and validation evidence are written "
            "to the Power BI workbook."
        )
        if st.button("Publish to Power BI", type="primary", use_container_width=True):
            try:
                published = publish_current_result(
                    st.session_state.current_question,
                    planner_result,
                    execution,
                    loaded,
                )
                if published.duplicate:
                    st.info(f"This result is already published. Run ID: {published.run_id}")
                else:
                    st.success(
                        f"Published {published.result_rows_written} result row(s). "
                        f"Run ID: {published.run_id}"
                    )
                st.code(str(published.workbook_path), language=None)
                st.caption("Open Power BI Desktop and select Home > Refresh.")
            except Exception as error:
                st.error(f"Power BI publication failed: {error}")


if __name__ == "__main__":
    main()
