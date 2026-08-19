"""InsightFlow AI Streamlit application."""

from __future__ import annotations

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
from app.services.filtered_executor import execute_filtered_analysis
from app.services.hybrid_parser import parse_question_hybrid
from app.services.text_normalizer import detect_language, normalize_question
from core.data_profiler import profile_dataframe
from core.file_loader import load_uploaded_file


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


RESPONSE_LANGUAGE_OPTIONS = {
    "Same as question": "same",
    "English": "en",
    "မြန်မာ": "my",
    "Bilingual": "bilingual",
}


def convert_hybrid_plan(
    question: str,
    hybrid_result: dict[str, Any],
) -> AnalysisPlan:
    """Convert rule-based or local-LLM output to the shared AnalysisPlan."""

    plan_data = hybrid_result["plan"]

    if "original_question" in plan_data:
        return AnalysisPlan.model_validate(plan_data)

    intent_name = str(plan_data.get("intent", "unknown"))
    visualization_name = str(plan_data.get("visualization", "table"))
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

    if intent_name in {
        "ranking",
        "trend",
        "comparison",
    } and dimension is None:
        warnings.append(
            "The local model did not identify a dimension."
        )

    if intent_name == "anomaly":
        warnings.append(
            "Anomaly execution is not implemented yet. "
            "The plan requires manual review."
        )

    confidence = 1.0 if not warnings else 0.65

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
        confidence=confidence,
        warnings=warnings,
    )


def metric_value_from_result(
    result_records: list[dict[str, Any]],
) -> float | int | None:
    """Return a summary value when an execution result contains one."""

    if len(result_records) != 1:
        return None

    value = result_records[0].get("value")

    if isinstance(value, (int, float)):
        return value

    return None


def format_number(value: float | int) -> str:
    """Format a number for a KPI card."""

    if isinstance(value, float) and not value.is_integer():
        return f"{value:,.2f}"

    return f"{value:,.0f}"


def display_analysis_result(
    result: Any,
    analysis_plan: AnalysisPlan,
    source: str,
) -> None:
    """Display analysis output, evidence, warnings, and CSV export."""

    analysis = result.analysis
    result_records = analysis["result"]
    result_df = pd.DataFrame(result_records)

    st.subheader("Analysis result")

    source_label = {
        "rule_based": "Deterministic rule parser",
        "local_llm": "Local Qwen3 4B fallback",
        "rule_based_fallback": "Rule parser after LLM failure",
    }.get(source, source)

    col_1, col_2, col_3, col_4 = st.columns(4)
    col_1.metric("Parser source", source_label)
    col_2.metric("Intent", analysis_plan.intent.value)
    col_3.metric("Source rows", f"{result.source_rows:,}")
    col_4.metric("Filtered rows", f"{result.filtered_rows:,}")

    summary_value = metric_value_from_result(result_records)

    if summary_value is not None:
        st.metric(
            label=(analysis_plan.metric or "Result").replace("_", " ").title(),
            value=format_number(summary_value),
        )

    if not result_df.empty:
        st.dataframe(
            result_df,
            use_container_width=True,
            hide_index=True,
        )

        csv_bytes = result_df.to_csv(index=False).encode("utf-8-sig")

        st.download_button(
            label="Download result CSV",
            data=csv_bytes,
            file_name="insightflow_analysis_result.csv",
            mime="text/csv",
        )

    if result.applied_filters:
        st.subheader("Applied filters")
        st.dataframe(
            pd.DataFrame(result.applied_filters),
            use_container_width=True,
            hide_index=True,
        )

    for warning in result.filter_warnings:
        st.warning(warning)

    for warning in analysis.get("warnings", []):
        st.warning(warning)

    with st.expander("Calculation and validation evidence"):
        st.write(
            "**Calculation:**",
            analysis.get("calculation") or "Not applicable",
        )
        st.json(analysis.get("validation", {}))

    with st.expander("Detected analysis plan"):
        st.json(analysis_plan.model_dump(mode="json"))


st.title("InsightFlow AI")
st.subheader("Bilingual NLP-Based Data Analytics Agent")

st.write(
    "Upload a CSV or Excel dataset, then describe the analytical "
    "problem in English, Myanmar, or mixed language."
)


uploaded_file = st.file_uploader(
    "Upload a dataset",
    type=["csv", "xlsx"],
    help="The current MVP supports CSV and XLSX files.",
)


if uploaded_file is None:
    st.info("Upload a CSV or XLSX file to begin.")
    st.stop()


try:
    dataframe = load_uploaded_file(
        uploaded_file=uploaded_file,
        filename=uploaded_file.name,
    )
except Exception as error:
    st.error(f"File loading failed: {error}")
    st.stop()


if dataframe.empty:
    st.warning("The uploaded file contains no data rows.")
    st.stop()


profile = profile_dataframe(dataframe)

st.success(f"Successfully loaded: {uploaded_file.name}")

metric_1, metric_2, metric_3, metric_4 = st.columns(4)
metric_1.metric("Rows", f"{profile['row_count']:,}")
metric_2.metric("Columns", f"{profile['column_count']:,}")
metric_3.metric("Duplicate rows", f"{profile['duplicate_rows']:,}")
metric_4.metric("Quality score", f"{profile['quality_score']}%")


preview_tab, quality_tab, ask_tab = st.tabs(
    [
        "Dataset Preview",
        "Data Quality",
        "Ask InsightFlow AI",
    ]
)


with preview_tab:
    st.dataframe(
        dataframe.head(100),
        use_container_width=True,
    )


with quality_tab:
    schema_df = pd.DataFrame(profile["columns"])
    st.write("### Detected schema")
    st.dataframe(
        schema_df,
        use_container_width=True,
        hide_index=True,
    )

    missing_df = pd.DataFrame(
        {
            "column": profile["missing_by_column"].keys(),
            "missing_count": profile["missing_by_column"].values(),
        }
    )
    missing_df["missing_percentage"] = (
        missing_df["missing_count"]
        / max(profile["row_count"], 1)
        * 100
    ).round(2)

    st.write("### Missing values")
    st.dataframe(
        missing_df.sort_values(
            by="missing_count",
            ascending=False,
        ),
        use_container_width=True,
        hide_index=True,
    )


with ask_tab:
    st.write(
        "Ask a clear analytical question. The rule parser handles "
        "common requests, while local Qwen3 4B handles ambiguous ones."
    )

    user_problem = st.text_area(
        "Describe your analytical problem",
        value="Show the top 5 products by revenue.",
        height=120,
        placeholder=(
            "Example: ရန်ကုန်ဒေသ၏ ဝင်ငွေကို ပြပါ"
        ),
    )

    response_language = st.selectbox(
        "Response language",
        options=list(RESPONSE_LANGUAGE_OPTIONS.keys()),
    )

    analyze_clicked = st.button(
        "Analyze",
        type="primary",
        use_container_width=False,
    )

    if analyze_clicked:
        if not user_problem.strip():
            st.warning("Enter an analytical problem before continuing.")
        else:
            try:
                with st.spinner(
                    "Understanding the question and running the analysis..."
                ):
                    hybrid_result = parse_question_hybrid(user_problem)
                    analysis_plan = convert_hybrid_plan(
                        user_problem,
                        hybrid_result,
                    )

                    if analysis_plan.warnings:
                        for warning in analysis_plan.warnings:
                            st.warning(warning)

                    if analysis_plan.intent == AnalysisIntent.UNKNOWN:
                        raise ValueError(
                            "The request could not be converted into a "
                            "supported analytical operation."
                        )

                    result = execute_filtered_analysis(
                        dataframe,
                        analysis_plan,
                    )

                st.success("Analysis completed successfully.")
                st.caption(
                    "Selected response language: "
                    f"{response_language}"
                )

                display_analysis_result(
                    result=result,
                    analysis_plan=analysis_plan,
                    source=hybrid_result["source"],
                )

                if hybrid_result.get("llm_error"):
                    st.warning(
                        "The local LLM was unavailable, so the rule-based "
                        "plan was used. "
                        f"Details: {hybrid_result['llm_error']}"
                    )

            except Exception as error:
                st.error(f"Analysis failed: {error}")
                st.info(
                    "Try a more explicit question such as: "
                    "'Show total revenue', "
                    "'Show the top 5 products by profit', or "
                    "'ရန်ကုန်ဒေသ၏ ဝင်ငွေကို ပြပါ'."
                )
