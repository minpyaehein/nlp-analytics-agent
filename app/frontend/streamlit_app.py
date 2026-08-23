"""InsightFlow AI Streamlit app with CSV, XLSX, and PDF ingestion."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from core.pdf_analysis_state import (
    state_matches_dataframe,
    validate_before_metric,
    validate_current_pdf_dataframe,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.models.analysis_plan import (
    AnalysisIntent,
    AnalysisPlan,
    FilterCondition,
    SortDirection,
    VisualizationType,
)
from app.services.chart_generator import generate_chart
from app.services.filtered_executor import execute_filtered_analysis
from app.services.hybrid_parser import parse_question_hybrid
from app.services.text_normalizer import detect_language, normalize_question
from core.data_profiler import profile_dataframe
from core.analytics_quality_gate import (
    validate_profit_readiness,
    validate_revenue_readiness,
)
from core.unified_file_loader import process_uploaded_file

st.set_page_config(
    page_title="InsightFlow AI",
    page_icon="📊",
    layout="wide",
)

INTENT_MAP = {
    "summary": AnalysisIntent.SUMMARY,
    "trend": AnalysisIntent.TREND,
    "comparison": AnalysisIntent.COMPARISON,
    "ranking": AnalysisIntent.RANKING,
    "distribution": AnalysisIntent.DISTRIBUTION,
    "data_quality": AnalysisIntent.DATA_QUALITY,
    "correlation": AnalysisIntent.CORRELATION,
    "unknown": AnalysisIntent.UNKNOWN,
    "anomaly": AnalysisIntent.UNKNOWN,
}

VISUALIZATION_MAP = {
    "kpi": VisualizationType.KPI,
    "line": VisualizationType.LINE,
    "bar": VisualizationType.BAR,
    "histogram": VisualizationType.HISTOGRAM,
    "scatter": VisualizationType.SCATTER,
    "table": VisualizationType.TABLE,
}

SORT_MAP = {
    "ascending": SortDirection.ASCENDING,
    "descending": SortDirection.DESCENDING,
}

EXAMPLE_QUESTIONS = {
    "Top products by revenue": "Show the top 5 products by revenue.",
    "Total revenue": "Show total revenue.",
    "Monthly revenue trend": "Show monthly revenue trend.",
    "Data quality": "Analyze missing values and duplicate rows.",
    "Myanmar profit ranking": "အမြတ်အများဆုံး ကုန်ပစ္စည်း ၅ ခုကို ပြပါ",
    "Myanmar Yangon revenue": "ရန်ကုန်ဒေသ၏ ဝင်ငွေကို ပြပါ",
}

RESPONSE_LANGUAGES = [
    "Same as question",
    "English",
    "မြန်မာ",
    "Bilingual",
]


def make_arrow_safe(value: Any) -> Any:
    """Convert nested Python values into Streamlit/PyArrow-safe values."""
    if isinstance(value, set):
        value = list(value)
    if isinstance(value, (list, dict, tuple)):
        return json.dumps(value, ensure_ascii=False, default=str)
    if isinstance(value, Path):
        return str(value)
    return value


def arrow_safe_dataframe(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Return a display-safe DataFrame without changing analytical data."""
    output = dataframe.copy()
    for column in output.columns:
        if output[column].dtype == "object":
            output[column] = output[column].map(make_arrow_safe)
    return output


def convert_hybrid_plan(
    question: str,
    hybrid_result: dict[str, Any],
) -> AnalysisPlan:
    """Convert deterministic or local-Qwen output into AnalysisPlan."""
    plan_data = hybrid_result["plan"]
    if "original_question" in plan_data:
        return AnalysisPlan.model_validate(plan_data)

    intent_name = str(plan_data.get("intent", "unknown"))
    visualization_name = str(plan_data.get("visualization", "table"))
    sort_name = plan_data.get("sort_direction")
    metric = plan_data.get("metric")
    dimension = plan_data.get("dimension")
    warnings: list[str] = []

    if metric is None and intent_name not in {"data_quality", "correlation"}:
        warnings.append("The local model did not identify a metric.")
    if intent_name in {"ranking", "trend", "comparison"} and dimension is None:
        warnings.append("The local model did not identify a dimension.")
    if intent_name == "anomaly":
        warnings.append("Anomaly execution is not implemented yet.")

    return AnalysisPlan(
        original_question=question.strip(),
        normalized_question=normalize_question(question),
        language=detect_language(question),
        intent=INTENT_MAP.get(intent_name, AnalysisIntent.UNKNOWN),
        metric=metric,
        dimension=dimension,
        aggregation=plan_data.get("aggregation", "sum"),
        sort_direction=(
            SORT_MAP.get(str(sort_name)) if sort_name is not None else None
        ),
        limit=plan_data.get("limit"),
        visualization=VISUALIZATION_MAP.get(
            visualization_name,
            VisualizationType.TABLE,
        ),
        filters=[
            FilterCondition.model_validate(item)
            for item in plan_data.get("filters", [])
        ],
        confidence=1.0 if not warnings else 0.65,
        warnings=warnings,
    )


def readiness_is_ready(readiness: dict[str, Any]) -> bool:
    """Read readiness state consistently across quality-gate versions."""
    return bool(
        readiness.get("ready", readiness.get("success", False))
    )


def enforce_metric_quality_gate(
    plan: AnalysisPlan,
    dataframe: pd.DataFrame,
    upload_result: Any,
) -> dict[str, Any] | None:
    """Validate the exact current DataFrame before financial calculation.

    PDF/OCR readiness stored in ``upload_result`` is retained as provenance,
    but it is not trusted as the final calculation gate. This function always
    recalculates readiness from ``dataframe``, which is the same object passed
    to the deterministic executor.
    """

    metric = (plan.metric or "").strip().casefold()

    if metric not in {"revenue", "profit"}:
        return None

    if metric == "revenue":
        gate = validate_revenue_readiness(dataframe)
    else:
        gate = validate_profit_readiness(dataframe)

    if not gate.analytics_ready:
        source_label = str(
            getattr(upload_result, "source_type", "dataset")
        ).upper()

        raise ValueError(
            f"{source_label} extraction is not ready for {metric} "
            "calculation. Current DataFrame quality gate: "
            f"{gate.to_dict()}"
        )

    return gate.to_dict()


def display_upload_evidence(upload_result: Any) -> None:
    """Display PDF/OCR/extraction provenance and readiness evidence."""
    with st.expander("File Processing Evidence", expanded=False):
        st.write("**Processing steps**")
        for index, step in enumerate(upload_result.processing_steps, start=1):
            st.write(f"{index}. {step}")

        if upload_result.warnings:
            st.write("**Warnings**")
            for warning in upload_result.warnings:
                st.warning(warning)

        if upload_result.source_type == "pdf":
            source_tab, extraction_tab, readiness_tab = st.tabs(
                ["PDF/OCR", "Table Extraction", "Quality Gates"]
            )
            with source_tab:
                st.json(upload_result.source_metadata)
            with extraction_tab:
                st.json(upload_result.extraction_metadata)
            with readiness_tab:
                col_1, col_2 = st.columns(2)
                with col_1:
                    st.write("**Revenue readiness**")
                    st.json(upload_result.revenue_readiness)
                with col_2:
                    st.write("**Profit readiness**")
                    st.json(upload_result.profit_readiness)

    if (
        upload_result.source_type == "pdf"
        and upload_result.searchable_pdf_bytes
    ):
        st.download_button(
            "Download Searchable PDF",
            data=upload_result.searchable_pdf_bytes,
            file_name=(
                Path(upload_result.filename).stem + "_searchable.pdf"
            ),
            mime="application/pdf",
            width="content",
        )


def display_chart(
    records: list[dict[str, Any]],
    plan: AnalysisPlan,
) -> None:
    """Generate, validate, display, and export a Plotly chart."""
    if not records or plan.visualization == VisualizationType.TABLE:
        return

    try:
        chart = generate_chart(records, plan)
    except Exception as error:
        st.warning(f"Chart generation was skipped: {error}")
        return

    if not chart.success or chart.figure is None:
        for warning in chart.warnings:
            st.warning(warning)
        return

    st.subheader("Automatic Visualization")
    st.plotly_chart(
        chart.figure,
        width="stretch",
        config={"displaylogo": False, "responsive": True},
    )
    st.download_button(
        "Download Chart HTML",
        data=chart.figure.to_html(
            full_html=True,
            include_plotlyjs="cdn",
        ).encode("utf-8"),
        file_name="insightflow_chart.html",
        mime="text/html",
        width="content",
    )
    with st.expander("Chart Validation"):
        st.json(chart.metadata())


def display_analysis_result(
    result: Any,
    plan: AnalysisPlan,
    parser_source: str,
) -> None:
    """Display results, charts, filters, evidence, and downloads."""
    analysis = result.analysis
    records = analysis.get("result", [])
    result_dataframe = arrow_safe_dataframe(pd.DataFrame(records))
    source_label = {
        "rule_based": "Deterministic rule parser",
        "local_llm": "Local Qwen3 4B fallback",
        "rule_based_fallback": "Rule parser after LLM failure",
    }.get(parser_source, parser_source)

    st.subheader("Analysis Result")
    col_1, col_2, col_3, col_4 = st.columns(4)
    col_1.metric("Parser source", source_label)
    col_2.metric("Intent", plan.intent.value)
    col_3.metric("Source rows", f"{result.source_rows:,}")
    col_4.metric("Filtered rows", f"{result.filtered_rows:,}")

    if len(records) == 1 and isinstance(records[0].get("value"), (int, float)):
        value = float(records[0]["value"])
        st.metric(
            (plan.metric or "Result").replace("_", " ").title(),
            f"{value:,.0f}" if value.is_integer() else f"{value:,.2f}",
        )

    if not result_dataframe.empty:
        st.dataframe(result_dataframe, width="stretch", hide_index=True)
        st.download_button(
            "Download Result CSV",
            data=result_dataframe.to_csv(index=False).encode("utf-8-sig"),
            file_name="insightflow_analysis_result.csv",
            mime="text/csv",
            width="content",
        )

    display_chart(records, plan)

    if result.applied_filters:
        st.subheader("Applied Filters")
        st.dataframe(
            arrow_safe_dataframe(pd.DataFrame(result.applied_filters)),
            width="stretch",
            hide_index=True,
        )

    for warning in result.filter_warnings:
        st.warning(warning)
    for warning in analysis.get("warnings", []):
        st.warning(warning)

    with st.expander("Calculation and Validation Evidence"):
        st.write("**Calculation:**", analysis.get("calculation") or "N/A")
        st.json(analysis.get("validation", {}))
    with st.expander("Detected Analysis Plan"):
        st.json(plan.model_dump(mode="json"))


st.title("InsightFlow AI")
st.subheader("Bilingual NLP-Based Data Analytics Agent")
st.write(
    "Upload CSV, Excel, or PDF. Scanned PDFs are OCR-processed "
    "automatically before validated table analytics."
)

with st.sidebar:
    st.header("PDF Processing")
    ocr_language = st.selectbox(
        "OCR language",
        ["eng+mya", "eng", "mya"],
        help="Use eng+mya for bilingual English and Myanmar PDFs.",
        width="stretch",
    )
    st.caption(
        "PDF calculations are blocked when required OCR fields fail "
        "the revenue or profit quality gate."
    )

uploaded_file = st.file_uploader(
    "Upload a dataset or report",
    type=["csv", "xlsx", "pdf"],
    help="Supported formats: CSV, XLSX, searchable PDF, scanned PDF.",
    width="stretch",
)

if uploaded_file is None:
    st.info("Upload a CSV, XLSX, or PDF file to begin.")
    st.stop()

try:
    spinner_text = (
        "Inspecting the PDF, running OCR if required, and extracting tables..."
        if uploaded_file.name.casefold().endswith(".pdf")
        else "Loading and validating the tabular dataset..."
    )
    with st.spinner(spinner_text):
        upload_result = process_uploaded_file(
            uploaded_file,
            uploaded_file.name,
            ocr_language=ocr_language,
            timeout_seconds=600,
        )
except Exception as error:
    st.error(f"File processing failed: {error}")
    st.stop()

if not upload_result.success or upload_result.dataframe.empty:
    st.error("The uploaded file did not produce a usable table.")
    st.stop()

dataframe = upload_result.dataframe
profile = profile_dataframe(dataframe)

st.success(
    f"Loaded {upload_result.filename} as {upload_result.source_type.upper()}."
)

metric_1, metric_2, metric_3, metric_4 = st.columns(4)
metric_1.metric("Rows", f"{profile['row_count']:,}")
metric_2.metric("Columns", f"{profile['column_count']:,}")
metric_3.metric("Duplicate rows", f"{profile['duplicate_rows']:,}")
metric_4.metric("Quality score", f"{profile['quality_score']}%")

display_upload_evidence(upload_result)

preview_tab, quality_tab, ask_tab = st.tabs(
    ["Dataset Preview", "Data Quality", "Ask InsightFlow AI"]
)

with preview_tab:
    st.dataframe(
        arrow_safe_dataframe(dataframe.head(100)),
        width="stretch",
        hide_index=True,
    )

with quality_tab:
    st.write("### Detected Schema")
    st.dataframe(
        arrow_safe_dataframe(pd.DataFrame(profile["columns"])),
        width="stretch",
        hide_index=True,
    )
    missing_dataframe = pd.DataFrame(
        {
            "column": profile["missing_by_column"].keys(),
            "missing_count": profile["missing_by_column"].values(),
        }
    )
    missing_dataframe["missing_percentage"] = (
        missing_dataframe["missing_count"]
        / max(profile["row_count"], 1)
        * 100
    ).round(2)
    st.write("### Missing Values")
    st.dataframe(
        missing_dataframe.sort_values("missing_count", ascending=False),
        width="stretch",
        hide_index=True,
    )
    with st.expander("Complete Profile JSON"):
        st.json(profile)

with ask_tab:
    selected_example = st.selectbox(
        "Example question",
        list(EXAMPLE_QUESTIONS),
        width="stretch",
    )
    if "question_text" not in st.session_state:
        st.session_state.question_text = EXAMPLE_QUESTIONS[selected_example]
    if st.button("Use Selected Example", width="content"):
        st.session_state.question_text = EXAMPLE_QUESTIONS[selected_example]

    question = st.text_area(
        "Describe your analytical problem",
        key="question_text",
        height=120,
        placeholder="Example: ရန်ကုန်ဒေသ၏ ဝင်ငွေကို ပြပါ",
        width="stretch",
    )
    response_language = st.selectbox(
        "Response language",
        RESPONSE_LANGUAGES,
        width="stretch",
    )

    if st.button("Analyze", type="primary", width="content"):
        if not question.strip():
            st.warning("Enter an analytical problem before continuing.")
        else:
            try:
                with st.spinner("Planning and executing validated analysis..."):
                    hybrid_result = parse_question_hybrid(question)
                    plan = convert_hybrid_plan(question, hybrid_result)
                    if plan.intent == AnalysisIntent.UNKNOWN:
                        raise ValueError(
                            "The request could not be converted into a "
                            "supported analytical operation."
                        )
                    current_quality_gate = enforce_metric_quality_gate(
                        plan=plan,
                        dataframe=dataframe,
                        upload_result=upload_result,
                    )

                    if current_quality_gate is not None:
                        if not hasattr(upload_result, "evidence"):
                            pass
                        elif isinstance(upload_result.evidence, dict):
                            upload_result.evidence[
                                "pre_analysis_quality_gate"
                            ] = current_quality_gate

                    execution_result = execute_filtered_analysis(
                        dataframe,
                        plan,
                    )

                st.success("Analysis completed successfully.")
                st.caption(f"Selected response language: {response_language}")
                display_analysis_result(
                    execution_result,
                    plan,
                    hybrid_result["source"],
                )
                if hybrid_result.get("llm_error"):
                    st.warning(
                        "The local LLM was unavailable; the rule-based plan "
                        f"was used. Details: {hybrid_result['llm_error']}"
                    )
            except ValueError as error:
                message = str(error)

                quality_gate_failure = (
                    "not safe for the requested calculation" in message
                    or "PDF extraction is not ready" in message
                    or "quality_gate" in message
                    or "analytics_ready" in message
                )

                if quality_gate_failure:
                    st.error(
                        "Revenue or profit calculation was blocked because "
                        "the reconstructed PDF table does not contain enough "
                        "usable numeric evidence."
                    )
                    st.warning(
                        "Reset the previous OCR result, upload the high-resolution "
                        "scan, rerun OCR, and confirm that the Revenue and Profit "
                        "quality gates are ready before analyzing."
                    )
                    with st.expander("Technical quality-gate details"):
                        st.code(message)
                else:
                    st.error(f"Analysis failed: {message}")

            except Exception as error:
                st.error(f"Analysis failed: {error}")
                st.info(
                    "Inspect File Processing Evidence and Quality Gates, "
                    "then use a clearer supported question."
                )
