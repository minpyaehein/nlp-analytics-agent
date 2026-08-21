"""InsightFlow AI Streamlit application.

Supported input formats:
- CSV
- TSV
- Delimited TXT
- JSON, including common nested record containers
- XLSX with worksheet selection

The application connects:
file loading -> profiling -> hybrid parsing -> filtered execution ->
validated results -> Plotly visualization -> verified bilingual insights.
"""

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
from core.file_loader import (
    list_excel_sheets,
    load_uploaded_file_with_metadata,
)


st.set_page_config(
    page_title="InsightFlow AI",
    page_icon="📊",
    layout="wide",
)


SUPPORTED_UPLOAD_TYPES = [
    "csv",
    "xlsx",
    "json",
    "txt",
    "tsv",
]


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
    "English product ranking": (
        "Show the top 5 products by revenue."
    ),
    "English summary": (
        "Show total revenue."
    ),
    "English filtered summary": (
        "Show total revenue for Yangon."
    ),
    "English trend": (
        "Show monthly revenue trend."
    ),
    "English data quality": (
        "Analyze missing values and duplicate rows."
    ),
    "Myanmar profit ranking": (
        "အမြတ်အများဆုံး ကုန်ပစ္စည်း ၅ ခုကို ပြပါ"
    ),
    "Myanmar Yangon revenue": (
        "ရန်ကုန်ဒေသ၏ ဝင်ငွေကို ပြပါ"
    ),
    "Mixed language": (
        "Mandalay region အတွက် top 5 products by revenue ပြပါ"
    ),
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
        return json.dumps(
            value,
            ensure_ascii=False,
            default=str,
        )

    if isinstance(value, Path):
        return str(value)

    return value


def make_dataframe_arrow_safe(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """Return a display-safe copy of a DataFrame."""

    safe_dataframe = dataframe.copy()

    for column in safe_dataframe.columns:
        if safe_dataframe[column].dtype == "object":
            safe_dataframe[column] = (
                safe_dataframe[column].apply(make_arrow_safe)
            )

    return safe_dataframe


def json_bytes(value: Any) -> bytes:
    """Serialize a value as UTF-8 JSON bytes."""

    return json.dumps(
        value,
        ensure_ascii=False,
        indent=2,
        default=str,
    ).encode("utf-8")


def enum_value(value: Any) -> str:
    """Return an Enum value or normalized string."""

    return str(getattr(value, "value", value))


def format_number(value: float | int) -> str:
    """Format a numerical KPI value."""

    number = float(value)

    if number.is_integer():
        return f"{int(number):,}"

    return f"{number:,.2f}"


def resolve_insight_language(
    selected_language: str,
    question_language: str,
) -> str:
    """Resolve English, Myanmar, or bilingual insight output."""

    requested = INSIGHT_LANGUAGE_MAP.get(
        selected_language,
        "bilingual",
    )

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
    """Convert rule-based or Qwen output to shared AnalysisPlan."""

    plan_data = hybrid_result["plan"]

    if "original_question" in plan_data:
        return AnalysisPlan.model_validate(plan_data)

    intent_name = str(plan_data.get("intent", "unknown"))
    visualization_name = str(
        plan_data.get("visualization", "table")
    )
    sort_name = plan_data.get("sort_direction")

    filters = [
        FilterCondition.model_validate(filter_data)
        for filter_data in plan_data.get("filters", [])
    ]

    metric = plan_data.get("metric")
    dimension = plan_data.get("dimension")
    warnings: list[str] = []

    if metric is None and intent_name not in {
        "data_quality",
        "correlation",
    }:
        warnings.append(
            "The local model did not identify a metric."
        )

    if (
        intent_name in {"ranking", "trend", "comparison"}
        and dimension is None
    ):
        warnings.append(
            "The local model did not identify a dimension."
        )

    if intent_name == "anomaly":
        warnings.append(
            "Anomaly execution is not implemented yet."
        )

    return AnalysisPlan(
        original_question=question.strip(),
        normalized_question=normalize_question(question),
        language=detect_language(question),
        intent=INTENT_MAP.get(
            intent_name,
            AnalysisIntent.UNKNOWN,
        ),
        metric=metric,
        dimension=dimension,
        aggregation=plan_data.get("aggregation", "sum"),
        sort_direction=(
            SORT_MAP.get(str(sort_name))
            if sort_name is not None
            else None
        ),
        limit=plan_data.get("limit"),
        visualization=VISUALIZATION_MAP.get(
            visualization_name,
            VisualizationType.TABLE,
        ),
        filters=filters,
        confidence=1.0 if not warnings else 0.65,
        warnings=warnings,
    )


def extract_summary_value(
    result_records: list[dict[str, Any]],
) -> float | int | None:
    """Extract a one-row summary result value."""

    if len(result_records) != 1:
        return None

    value = result_records[0].get("value")

    if isinstance(value, (int, float)):
        return value

    return None


def show_file_metadata(metadata: Any) -> None:
    """Display file-loading audit metadata and warnings."""

    metadata_dict = metadata.to_dict()

    col_1, col_2, col_3 = st.columns(3)
    col_1.metric("Input format", metadata.format.upper())
    col_2.metric("Loaded rows", f"{metadata.row_count:,}")
    col_3.metric("Loaded columns", f"{metadata.column_count:,}")

    with st.expander("File Loading Metadata"):
        st.json(metadata_dict)

    for warning in metadata.warnings or []:
        st.warning(warning)


def show_dataset_profile(
    dataframe: pd.DataFrame,
    profile: dict[str, Any],
) -> None:
    """Display dataset preview and deterministic profile."""

    metric_1, metric_2, metric_3, metric_4 = st.columns(4)
    metric_1.metric("Rows", f"{profile['row_count']:,}")
    metric_2.metric("Columns", f"{profile['column_count']:,}")
    metric_3.metric(
        "Duplicate rows",
        f"{profile['duplicate_rows']:,}",
    )
    metric_4.metric(
        "Quality score",
        f"{profile['quality_score']}%",
    )

    st.write("### Dataset Preview")
    st.dataframe(
        make_dataframe_arrow_safe(dataframe.head(100)),
        width="stretch",
        hide_index=True,
    )


def show_data_quality(profile: dict[str, Any]) -> None:
    """Display schema, missing values, and profile JSON."""

    schema_dataframe = make_dataframe_arrow_safe(
        pd.DataFrame(profile["columns"])
    )

    st.write("### Detected Schema")
    st.dataframe(
        schema_dataframe,
        width="stretch",
        hide_index=True,
    )

    missing_dataframe = pd.DataFrame(
        {
            "column": profile["missing_by_column"].keys(),
            "missing_count": profile[
                "missing_by_column"
            ].values(),
        }
    )
    missing_dataframe["missing_percentage"] = (
        missing_dataframe["missing_count"]
        / max(profile["row_count"], 1)
        * 100
    ).round(2)

    st.write("### Missing Values")
    st.dataframe(
        missing_dataframe.sort_values(
            by="missing_count",
            ascending=False,
        ),
        width="stretch",
        hide_index=True,
    )

    with st.expander("Complete Profile JSON"):
        st.json(profile)


def show_chart(
    result_records: list[dict[str, Any]],
    analysis_plan: AnalysisPlan,
) -> None:
    """Generate and display a validated Plotly chart."""

    try:
        chart_result = generate_chart(
            result_records=result_records,
            plan=analysis_plan,
        )

        st.subheader("Automatic Visualization")
        st.plotly_chart(
            chart_result.figure,
            width="stretch",
        )

        for warning in chart_result.warnings:
            st.warning(warning)

        with st.expander("Chart Validation"):
            st.json(chart_result.validation)

    except ValueError as chart_error:
        st.info(
            "A chart was not generated for this result. "
            f"Reason: {chart_error}"
        )


def show_verified_insights(
    result_records: list[dict[str, Any]],
    analysis_plan: AnalysisPlan,
    insight_language: str,
) -> None:
    """Generate and display deterministic bilingual insights."""

    try:
        insight_result = generate_insights(
            result_records=result_records,
            plan=analysis_plan,
            language=insight_language,
        )

        st.subheader("Verified Insights")
        st.markdown(f"### {insight_result.headline}")

        for finding in insight_result.findings:
            st.markdown(f"- {finding}")

        for warning in insight_result.warnings:
            st.warning(warning)

        evidence_tab, validation_tab = st.tabs(
            [
                "Insight Evidence",
                "Insight Validation",
            ]
        )

        with evidence_tab:
            st.json(insight_result.evidence)

        with validation_tab:
            if (
                insight_result.validation.get("status")
                == "passed"
            ):
                st.success("Insight validation passed.")
            else:
                st.warning(
                    "Insight validation returned warnings."
                )

            st.json(insight_result.validation)

    except ValueError as insight_error:
        st.info(
            "Verified insights were not generated for this "
            f"result. Reason: {insight_error}"
        )


def show_analysis_result(
    result: Any,
    analysis_plan: AnalysisPlan,
    parser_source: str,
    insight_language: str,
) -> None:
    """Display analytics result, chart, insights, and evidence."""

    analysis = result.analysis
    result_records = analysis.get("result", [])
    result_dataframe = make_dataframe_arrow_safe(
        pd.DataFrame(result_records)
    )

    st.subheader("Analysis Result")

    source_label = {
        "rule_based": "Deterministic rule parser",
        "local_llm": "Local Qwen3 4B fallback",
        "rule_based_fallback": (
            "Rule parser after LLM failure"
        ),
    }.get(parser_source, parser_source)

    col_1, col_2, col_3, col_4 = st.columns(4)
    col_1.metric("Parser source", source_label)
    col_2.metric("Intent", analysis_plan.intent.value)
    col_3.metric("Source rows", f"{result.source_rows:,}")
    col_4.metric("Filtered rows", f"{result.filtered_rows:,}")

    summary_value = extract_summary_value(result_records)

    if summary_value is not None:
        st.metric(
            label=(
                analysis_plan.metric or "Result"
            ).replace("_", " ").title(),
            value=format_number(summary_value),
        )

    if result_dataframe.empty:
        st.warning("The analysis produced no result rows.")
        return

    st.dataframe(
        result_dataframe,
        width="stretch",
        hide_index=True,
    )

    show_chart(
        result_records=result_records,
        analysis_plan=analysis_plan,
    )

    show_verified_insights(
        result_records=result_records,
        analysis_plan=analysis_plan,
        insight_language=insight_language,
    )

    if result.applied_filters:
        st.subheader("Applied Filters")
        st.dataframe(
            make_dataframe_arrow_safe(
                pd.DataFrame(result.applied_filters)
            ),
            width="stretch",
            hide_index=True,
        )

    for warning in result.filter_warnings:
        st.warning(warning)

    for warning in analysis.get("warnings", []):
        st.warning(warning)

    evidence_tab, plan_tab, raw_tab = st.tabs(
        [
            "Calculation Evidence",
            "Analysis Plan",
            "Raw Evidence",
        ]
    )

    with evidence_tab:
        st.write(
            "**Calculation:**",
            analysis.get("calculation") or "Not applicable",
        )
        st.json(analysis.get("validation", {}))

    with plan_tab:
        st.json(analysis_plan.model_dump(mode="json"))

    with raw_tab:
        st.json(result.to_dict())

    st.subheader("Export Results")

    download_1, download_2 = st.columns(2)

    with download_1:
        st.download_button(
            label="Download Result CSV",
            data=result_dataframe.to_csv(
                index=False
            ).encode("utf-8-sig"),
            file_name="insightflow_analysis_result.csv",
            mime="text/csv",
            width="stretch",
        )

    with download_2:
        st.download_button(
            label="Download Evidence JSON",
            data=json_bytes(result.to_dict()),
            file_name="insightflow_analysis_evidence.json",
            mime="application/json",
            width="stretch",
        )


st.title("InsightFlow AI")
st.subheader("Bilingual NLP-Based Data Analytics Agent")

st.write(
    "Upload CSV, XLSX, JSON, TXT, or TSV data. Then ask an "
    "analytical question in English, Myanmar, or mixed language."
)


uploaded_file = st.file_uploader(
    "Upload a dataset",
    type=SUPPORTED_UPLOAD_TYPES,
    help=(
        "Supported formats: CSV, XLSX, JSON, TXT, and TSV. "
        "TXT files must contain delimited tabular data."
    ),
)


if uploaded_file is None:
    st.info(
        "Upload a CSV, XLSX, JSON, TXT, or TSV file to begin."
    )
    st.stop()


selected_sheet: str | None = None

if uploaded_file.name.casefold().endswith(".xlsx"):
    try:
        worksheets = list_excel_sheets(
            uploaded_file=uploaded_file,
            filename=uploaded_file.name,
        )

        selected_sheet = st.selectbox(
            "Select an Excel worksheet",
            options=worksheets,
            help=(
                "Only the selected worksheet will be analyzed."
            ),
        )

    except Exception as error:
        st.error(f"Unable to inspect Excel worksheets: {error}")
        st.stop()


try:
    loaded_dataset = load_uploaded_file_with_metadata(
        uploaded_file=uploaded_file,
        filename=uploaded_file.name,
        sheet_name=selected_sheet,
    )
    dataframe = loaded_dataset.dataframe
    file_metadata = loaded_dataset.metadata

except Exception as error:
    st.error(f"File loading failed: {error}")
    st.stop()


if dataframe.empty:
    st.warning("The uploaded file contains no data rows.")
    st.stop()


profile = profile_dataframe(dataframe)

st.success(f"Successfully loaded: {uploaded_file.name}")
show_file_metadata(file_metadata)


preview_tab, quality_tab, analysis_tab = st.tabs(
    [
        "Dataset Preview",
        "Data Quality",
        "Ask InsightFlow AI",
    ]
)


with preview_tab:
    show_dataset_profile(
        dataframe=dataframe,
        profile=profile,
    )


with quality_tab:
    show_data_quality(profile)


with analysis_tab:
    st.write(
        "The deterministic parser handles common requests. "
        "Local Qwen3 4B is used only when the rule parser has "
        "low confidence or unresolved warnings."
    )

    selected_example = st.selectbox(
        "Example question",
        options=list(EXAMPLE_QUESTIONS.keys()),
    )

    example_question = EXAMPLE_QUESTIONS[selected_example]

    user_problem = st.text_area(
        "Describe your analytical problem",
        value=example_question,
        height=120,
        placeholder=(
            "Example: ရန်ကုန်ဒေသ၏ ဝင်ငွေကို ပြပါ"
        ),
    )

    response_language = st.selectbox(
        "Response language",
        options=RESPONSE_LANGUAGE_OPTIONS,
    )

    analyze_clicked = st.button(
        "Analyze",
        type="primary",
        width="content",
    )

    if analyze_clicked:
        if not user_problem.strip():
            st.warning(
                "Enter an analytical problem before continuing."
            )
        else:
            try:
                with st.spinner(
                    "Understanding the question and running "
                    "the analysis..."
                ):
                    hybrid_result = parse_question_hybrid(
                        user_problem
                    )
                    analysis_plan = convert_hybrid_plan(
                        question=user_problem,
                        hybrid_result=hybrid_result,
                    )

                    for warning in analysis_plan.warnings:
                        st.warning(warning)

                    if (
                        analysis_plan.intent
                        == AnalysisIntent.UNKNOWN
                    ):
                        raise ValueError(
                            "The request could not be converted "
                            "into a supported analytical operation."
                        )

                    execution_result = execute_filtered_analysis(
                        dataframe=dataframe,
                        plan=analysis_plan,
                    )

                    insight_language = resolve_insight_language(
                        selected_language=response_language,
                        question_language=analysis_plan.language,
                    )

                st.success("Analysis completed successfully.")

                display_analysis_result(
                    result=execution_result,
                    analysis_plan=analysis_plan,
                    parser_source=hybrid_result["source"],
                    insight_language=insight_language,
                )

                if hybrid_result.get("llm_error"):
                    st.warning(
                        "The local LLM was unavailable, so the "
                        "rule-based result was used. "
                        f"Details: {hybrid_result['llm_error']}"
                    )

            except Exception as error:
                st.error(f"Analysis failed: {error}")
                st.info(
                    "Try a more explicit question such as "
                    "'Show total revenue', "
                    "'Show the top 5 products by profit', or "
                    "'ရန်ကုန်ဒေသ၏ ဝင်ငွေကို ပြပါ'."
                )
