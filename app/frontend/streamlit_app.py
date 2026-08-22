"""InsightFlow AI Streamlit application with DOCX table analytics."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

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
from app.services.insight_generator import generate_insights
from app.services.text_normalizer import detect_language, normalize_question
from core.data_profiler import profile_dataframe
from core.docx_loader import extract_docx
from core.file_loader import list_excel_sheets, load_uploaded_file_with_metadata

st.set_page_config(
    page_title="InsightFlow AI",
    page_icon="📊",
    layout="wide",
)

SUPPORTED_UPLOAD_TYPES = ["csv", "xlsx", "json", "txt", "tsv", "docx"]
RESPONSE_LANGUAGE_OPTIONS = [
    "Same as question",
    "English",
    "မြန်မာ",
    "Bilingual",
]
INSIGHT_LANGUAGE_MAP = {
    "Same as question": "same",
    "English": "en",
    "မြန်မာ": "my",
    "Bilingual": "bilingual",
}
EXAMPLE_QUESTIONS = {
    "English product ranking": "Show the top 5 products by revenue.",
    "English summary": "Show total revenue.",
    "English Yangon summary": "Show total revenue for Yangon.",
    "English monthly trend": "Show monthly revenue trend.",
    "English data quality": "Analyze missing values and duplicate rows.",
    "Myanmar profit ranking": "အမြတ်အများဆုံး ကုန်ပစ္စည်း ၅ ခုကို ပြပါ",
    "Myanmar Yangon revenue": "ရန်ကုန်ဒေသ၏ ဝင်ငွေကို ပြပါ",
    "Mixed language": "Mandalay region အတွက် top 5 products by revenue ပြပါ",
}
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


def make_arrow_safe(value: Any) -> Any:
    """Convert nested values into Streamlit/PyArrow-safe values."""
    if isinstance(value, set):
        value = list(value)
    if isinstance(value, (list, dict, tuple)):
        return json.dumps(value, ensure_ascii=False, default=str)
    if isinstance(value, Path):
        return str(value)
    return value


def make_dataframe_arrow_safe(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Return a display-safe copy of a DataFrame."""
    safe = dataframe.copy()
    for column in safe.columns:
        if safe[column].dtype == "object":
            safe[column] = safe[column].apply(make_arrow_safe)
    return safe


def json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        indent=2,
        default=str,
    ).encode("utf-8")


def format_number(value: float | int) -> str:
    number = float(value)
    return f"{int(number):,}" if number.is_integer() else f"{number:,.2f}"


def resolve_insight_language(selected: str, question_language: str) -> str:
    requested = INSIGHT_LANGUAGE_MAP.get(selected, "bilingual")
    if requested != "same":
        return requested
    if question_language == "my":
        return "my"
    if question_language == "en":
        return "en"
    return "bilingual"


def convert_hybrid_plan(
    question: str,
    hybrid_result: dict[str, Any],
) -> AnalysisPlan:
    """Convert rule-based or Qwen output to the shared AnalysisPlan."""
    data = hybrid_result["plan"]
    if "original_question" in data:
        return AnalysisPlan.model_validate(data)

    intent_name = str(data.get("intent", "unknown"))
    visualization_name = str(data.get("visualization", "table"))
    sort_name = data.get("sort_direction")
    filters = [
        FilterCondition.model_validate(item)
        for item in data.get("filters", [])
    ]
    metric = data.get("metric")
    dimension = data.get("dimension")
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
        aggregation=data.get("aggregation", "sum"),
        sort_direction=(
            SORT_MAP.get(str(sort_name)) if sort_name is not None else None
        ),
        limit=data.get("limit"),
        visualization=VISUALIZATION_MAP.get(
            visualization_name,
            VisualizationType.TABLE,
        ),
        filters=filters,
        confidence=1.0 if not warnings else 0.65,
        warnings=warnings,
    )


def extract_summary_value(records: list[dict[str, Any]]) -> float | int | None:
    if len(records) != 1:
        return None
    value = records[0].get("value")
    return value if isinstance(value, (int, float)) else None


def show_standard_file_metadata(metadata: Any) -> None:
    """Display metadata for CSV, XLSX, JSON, TXT, or TSV loading."""
    col_1, col_2, col_3 = st.columns(3)
    col_1.metric("Input format", metadata.format.upper())
    col_2.metric("Loaded rows", f"{metadata.row_count:,}")
    col_3.metric("Loaded columns", f"{metadata.column_count:,}")
    with st.expander("File Loading Metadata"):
        st.json(metadata.to_dict())
    for warning in metadata.warnings or []:
        st.warning(warning)


def show_docx_metadata(extracted: Any) -> None:
    """Display DOCX document-level extraction metadata."""
    metadata = extracted.metadata
    col_1, col_2, col_3, col_4 = st.columns(4)
    col_1.metric("Paragraphs", f"{metadata.non_empty_paragraph_count:,}")
    col_2.metric("Headings", f"{metadata.heading_count:,}")
    col_3.metric("Word tables", f"{metadata.table_count:,}")
    col_4.metric("Usable tables", f"{metadata.extracted_table_count:,}")

    with st.expander("DOCX Metadata"):
        st.json(metadata.to_dict())

    for warning in metadata.warnings:
        st.warning(warning)


def show_docx_narrative(extracted: Any) -> None:
    """Display DOCX headings and paragraph evidence."""
    st.write("### Narrative Text Preview")
    records = extracted.text_records()
    if not records:
        st.info("No non-empty narrative paragraphs were extracted.")
        return

    preview = make_dataframe_arrow_safe(pd.DataFrame(records))
    preferred_columns = [
        "block_id",
        "block_type",
        "heading_level",
        "section_path",
        "text",
    ]
    available = [column for column in preferred_columns if column in preview.columns]
    st.dataframe(
        preview[available].head(200),
        width="stretch",
        hide_index=True,
    )

    combined_text = extracted.combined_text()
    with st.expander("Combined Narrative Text"):
        st.text_area(
            "Extracted text",
            value=combined_text[:30000],
            height=350,
            disabled=True,
        )
        if len(combined_text) > 30000:
            st.caption("Preview limited to the first 30,000 characters.")

    st.download_button(
        "Download Narrative TXT",
        data=combined_text.encode("utf-8-sig"),
        file_name="docx_narrative.txt",
        mime="text/plain",
        width="content",
    )


def show_dataset_profile(dataframe: pd.DataFrame, profile: dict[str, Any]) -> None:
    metric_1, metric_2, metric_3, metric_4 = st.columns(4)
    metric_1.metric("Rows", f"{profile['row_count']:,}")
    metric_2.metric("Columns", f"{profile['column_count']:,}")
    metric_3.metric("Duplicate rows", f"{profile['duplicate_rows']:,}")
    metric_4.metric("Quality score", f"{profile['quality_score']}%")
    st.write("### Dataset Preview")
    st.dataframe(
        make_dataframe_arrow_safe(dataframe.head(100)),
        width="stretch",
        hide_index=True,
    )


def show_data_quality(profile: dict[str, Any]) -> None:
    schema = make_dataframe_arrow_safe(pd.DataFrame(profile["columns"]))
    st.write("### Detected Schema")
    st.dataframe(schema, width="stretch", hide_index=True)

    missing = pd.DataFrame({
        "column": profile["missing_by_column"].keys(),
        "missing_count": profile["missing_by_column"].values(),
    })
    missing["missing_percentage"] = (
        missing["missing_count"] / max(profile["row_count"], 1) * 100
    ).round(2)
    st.write("### Missing Values")
    st.dataframe(
        missing.sort_values("missing_count", ascending=False),
        width="stretch",
        hide_index=True,
    )
    with st.expander("Complete Profile JSON"):
        st.json(profile)


def show_chart(records: list[dict[str, Any]], plan: AnalysisPlan) -> None:
    try:
        chart = generate_chart(result_records=records, plan=plan)
        st.subheader("Automatic Visualization")
        st.plotly_chart(chart.figure, width="stretch")
        for warning in chart.warnings:
            st.warning(warning)
        with st.expander("Chart Validation"):
            st.json(chart.validation)
    except ValueError as error:
        st.info(f"A chart was not generated. Reason: {error}")


def show_insights(
    records: list[dict[str, Any]],
    plan: AnalysisPlan,
    language: str,
) -> None:
    try:
        insight = generate_insights(
            result_records=records,
            plan=plan,
            language=language,
        )
        st.subheader("Verified Insights")
        st.markdown(f"### {insight.headline}")
        for finding in insight.findings:
            st.markdown(f"- {finding}")
        for warning in insight.warnings:
            st.warning(warning)

        evidence_tab, validation_tab = st.tabs(
            ["Insight Evidence", "Insight Validation"]
        )
        with evidence_tab:
            st.json(insight.evidence)
        with validation_tab:
            if insight.validation.get("status") == "passed":
                st.success("Insight validation passed.")
            st.json(insight.validation)
    except ValueError as error:
        st.info(f"Verified insights were not generated. Reason: {error}")


def show_analysis_result(
    result: Any,
    plan: AnalysisPlan,
    parser_source: str,
    insight_language: str,
    source_evidence: dict[str, Any] | None,
) -> None:
    """Display results, chart, insights, evidence, and exports."""
    analysis = result.analysis
    records = analysis.get("result", [])
    result_dataframe = make_dataframe_arrow_safe(pd.DataFrame(records))

    st.subheader("Analysis Result")
    source_label = {
        "rule_based": "Deterministic rule parser",
        "local_llm": "Local Qwen3 4B fallback",
        "rule_based_fallback": "Rule parser after LLM failure",
    }.get(parser_source, parser_source)

    col_1, col_2, col_3, col_4 = st.columns(4)
    col_1.metric("Parser source", source_label)
    col_2.metric("Intent", plan.intent.value)
    col_3.metric("Source rows", f"{result.source_rows:,}")
    col_4.metric("Filtered rows", f"{result.filtered_rows:,}")

    summary = extract_summary_value(records)
    if summary is not None:
        st.metric(
            (plan.metric or "Result").replace("_", " ").title(),
            format_number(summary),
        )

    if result_dataframe.empty:
        st.warning("The analysis produced no result rows.")
        return

    st.dataframe(result_dataframe, width="stretch", hide_index=True)
    show_chart(records, plan)
    show_insights(records, plan, insight_language)

    if result.applied_filters:
        st.subheader("Applied Filters")
        st.dataframe(
            make_dataframe_arrow_safe(pd.DataFrame(result.applied_filters)),
            width="stretch",
            hide_index=True,
        )

    for warning in result.filter_warnings:
        st.warning(warning)
    for warning in analysis.get("warnings", []):
        st.warning(warning)

    evidence_tab, plan_tab, source_tab, raw_tab = st.tabs(
        [
            "Calculation Evidence",
            "Analysis Plan",
            "Source Evidence",
            "Raw Evidence",
        ]
    )
    with evidence_tab:
        st.write("**Calculation:**", analysis.get("calculation") or "N/A")
        st.json(analysis.get("validation", {}))
    with plan_tab:
        st.json(plan.model_dump(mode="json"))
    with source_tab:
        if source_evidence:
            st.json(source_evidence)
        else:
            st.info("No separate document source evidence was required.")
    with raw_tab:
        st.json(result.to_dict())

    st.subheader("Export Results")
    download_1, download_2 = st.columns(2)
    with download_1:
        st.download_button(
            "Download Result CSV",
            data=result_dataframe.to_csv(index=False).encode("utf-8-sig"),
            file_name="insightflow_analysis_result.csv",
            mime="text/csv",
            width="stretch",
        )
    with download_2:
        export_evidence = {
            "analysis": result.to_dict(),
            "source_evidence": source_evidence,
        }
        st.download_button(
            "Download Evidence JSON",
            data=json_bytes(export_evidence),
            file_name="insightflow_analysis_evidence.json",
            mime="application/json",
            width="stretch",
        )


st.title("InsightFlow AI")
st.subheader("Bilingual NLP-Based Data Analytics Agent")
st.write(
    "Upload CSV, XLSX, JSON, TXT, TSV, or DOCX data. DOCX files can "
    "provide both narrative evidence and selectable analytical tables."
)

uploaded_file = st.file_uploader(
    "Upload a dataset or DOCX report",
    type=SUPPORTED_UPLOAD_TYPES,
    help=(
        "Supported formats: CSV, XLSX, JSON, TXT, TSV, and DOCX. "
        "TXT files must contain delimited tabular data."
    ),
)

if uploaded_file is None:
    st.info("Upload a supported file to begin.")
    st.stop()

is_docx = uploaded_file.name.casefold().endswith(".docx")
dataframe: pd.DataFrame
source_evidence: dict[str, Any] | None = None

if is_docx:
    try:
        extracted_docx = extract_docx(uploaded_file, uploaded_file.name)
    except Exception as error:
        st.error(f"DOCX extraction failed: {error}")
        st.stop()

    st.success(f"Successfully extracted: {uploaded_file.name}")
    show_docx_metadata(extracted_docx)

    doc_preview_tab, doc_table_tab = st.tabs(
        ["DOCX Narrative", "DOCX Tables"]
    )
    with doc_preview_tab:
        show_docx_narrative(extracted_docx)

    if not extracted_docx.tables:
        with doc_table_tab:
            st.warning(
                "No usable DOCX table was found. Narrative extraction "
                "succeeded, but tabular analytics requires a table."
            )
        st.stop()

    table_labels = [
        (
            f"{table.table_index}: {table.title} "
            f"({table.row_count} rows × {table.column_count} columns)"
        )
        for table in extracted_docx.tables
    ]
    with doc_table_tab:
        selected_label = st.selectbox(
            "Select a DOCX table for analytics",
            options=table_labels,
        )
        selected_index = table_labels.index(selected_label)
        selected_table = extracted_docx.tables[selected_index]
        dataframe = selected_table.dataframe.copy()

        st.write("### Selected DOCX Table")
        st.dataframe(
            make_dataframe_arrow_safe(dataframe.head(100)),
            width="stretch",
            hide_index=True,
        )
        with st.expander("Selected Table Metadata"):
            st.json(selected_table.metadata())
        for warning in selected_table.warnings:
            st.warning(warning)

    source_evidence = {
        "source_type": "docx_table",
        "document": extracted_docx.metadata.to_dict(),
        "selected_table": selected_table.metadata(),
        "narrative_block_count": len(extracted_docx.text_blocks),
        "narrative_references": [
            block.block_id for block in extracted_docx.text_blocks[:25]
        ],
    }

else:
    selected_sheet: str | None = None
    if uploaded_file.name.casefold().endswith(".xlsx"):
        try:
            worksheets = list_excel_sheets(uploaded_file, uploaded_file.name)
            selected_sheet = st.selectbox(
                "Select an Excel worksheet",
                options=worksheets,
            )
        except Exception as error:
            st.error(f"Unable to inspect Excel worksheets: {error}")
            st.stop()

    try:
        loaded = load_uploaded_file_with_metadata(
            uploaded_file=uploaded_file,
            filename=uploaded_file.name,
            sheet_name=selected_sheet,
        )
        dataframe = loaded.dataframe
        file_metadata = loaded.metadata
    except Exception as error:
        st.error(f"File loading failed: {error}")
        st.stop()

    st.success(f"Successfully loaded: {uploaded_file.name}")
    show_standard_file_metadata(file_metadata)
    source_evidence = {
        "source_type": "tabular_file",
        "file_metadata": file_metadata.to_dict(),
    }

if dataframe.empty:
    st.warning("The selected data contains no rows.")
    st.stop()

profile = profile_dataframe(dataframe)
preview_tab, quality_tab, analysis_tab = st.tabs(
    ["Dataset Preview", "Data Quality", "Ask InsightFlow AI"]
)

with preview_tab:
    show_dataset_profile(dataframe, profile)

with quality_tab:
    show_data_quality(profile)

with analysis_tab:
    st.write(
        "The deterministic parser handles common requests. Local Qwen3 4B "
        "is used only for low-confidence or unresolved requests."
    )
    selected_example = st.selectbox(
        "Example question",
        options=list(EXAMPLE_QUESTIONS.keys()),
    )
    user_problem = st.text_area(
        "Describe your analytical problem",
        value=EXAMPLE_QUESTIONS[selected_example],
        height=120,
        placeholder="Example: ရန်ကုန်ဒေသ၏ ဝင်ငွေကို ပြပါ",
    )
    response_language = st.selectbox(
        "Response language",
        options=RESPONSE_LANGUAGE_OPTIONS,
    )
    analyze_clicked = st.button("Analyze", type="primary", width="content")

    if analyze_clicked:
        if not user_problem.strip():
            st.warning("Enter an analytical problem before continuing.")
        else:
            try:
                with st.spinner("Understanding and executing the request..."):
                    hybrid_result = parse_question_hybrid(user_problem)
                    plan = convert_hybrid_plan(user_problem, hybrid_result)
                    for warning in plan.warnings:
                        st.warning(warning)
                    if plan.intent == AnalysisIntent.UNKNOWN:
                        raise ValueError(
                            "The request could not be converted into a "
                            "supported analytical operation."
                        )
                    result = execute_filtered_analysis(dataframe, plan)
                    insight_language = resolve_insight_language(
                        response_language,
                        plan.language,
                    )

                st.success("Analysis completed successfully.")
                show_analysis_result(
                    result=result,
                    plan=plan,
                    parser_source=hybrid_result["source"],
                    insight_language=insight_language,
                    source_evidence=source_evidence,
                )
                if hybrid_result.get("llm_error"):
                    st.warning(
                        "The local LLM was unavailable, so rule-based output "
                        f"was used. Details: {hybrid_result['llm_error']}"
                    )
            except Exception as error:
                st.error(f"Analysis failed: {error}")
                st.info(
                    "Try a clear request such as 'Show total revenue', "
                    "'Show the top 5 products by profit', or "
                    "'ရန်ကုန်ဒေသ၏ ဝင်ငွေကို ပြပါ'."
                )
