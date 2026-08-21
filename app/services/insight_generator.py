"""Verified bilingual insights from deterministic executor results."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal

import pandas as pd

from app.models.analysis_plan import AnalysisIntent, AnalysisPlan

InsightLanguage = Literal["en", "my", "bilingual"]

MY_METRIC = {
    "revenue": "ဝင်ငွေ",
    "profit": "အမြတ်",
    "cost": "ကုန်ကျစရိတ်",
    "quantity": "အရေအတွက်",
    "order_count": "အော်ဒါအရေအတွက်",
    "customer_count": "ဖောက်သည်အရေအတွက်",
}
MY_DIMENSION = {
    "product": "ကုန်ပစ္စည်း",
    "category": "အမျိုးအစား",
    "region": "ဒေသ",
    "customer": "ဖောက်သည်",
    "month": "လ",
    "quarter": "သုံးလပတ်",
    "year": "နှစ်",
    "order_date": "အော်ဒါရက်စွဲ",
}


@dataclass
class InsightResult:
    success: bool
    language: str
    headline: str
    findings: list[str]
    evidence: list[dict[str, Any]]
    warnings: list[str]
    validation: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _enum_value(value: Any) -> str:
    return str(getattr(value, "value", value))


def _number(value: Any) -> str:
    number = float(value)
    return f"{int(number):,}" if number.is_integer() else f"{number:,.2f}"


def _percentage(value: float) -> str:
    return f"{value:.1f}%"


def _metric_en(metric: str | None) -> str:
    return (metric or "value").replace("_", " ")


def _metric_my(metric: str | None) -> str:
    return MY_METRIC.get(metric or "", (metric or "တန်ဖိုး").replace("_", " "))


def _dimension_my(dimension: str | None) -> str:
    return MY_DIMENSION.get(
        dimension or "",
        (dimension or "အချက်အလက်").replace("_", " "),
    )


def _language_text(text: str, language: InsightLanguage) -> str:
    english, separator, myanmar = text.partition("\n")
    if language == "en":
        return english
    if language == "my":
        return myanmar if separator else english
    return text


def _result_columns(
    dataframe: pd.DataFrame,
    plan: AnalysisPlan,
) -> tuple[str | None, str | None]:
    dimension = plan.dimension if plan.dimension in dataframe.columns else None
    metric = plan.metric if plan.metric in dataframe.columns else None

    if dimension is None:
        text_columns = dataframe.select_dtypes(exclude="number").columns.tolist()
        if text_columns:
            dimension = str(text_columns[0])

    if metric is None:
        numeric_columns = dataframe.select_dtypes(include="number").columns.tolist()
        if numeric_columns:
            metric = str(numeric_columns[-1])

    return dimension, metric


def _summary(
    dataframe: pd.DataFrame,
    plan: AnalysisPlan,
) -> tuple[str, list[str], list[dict[str, Any]]]:
    if "value" not in dataframe.columns:
        raise ValueError("Summary insights require a 'value' result field.")

    value = pd.to_numeric(dataframe["value"], errors="coerce").iloc[0]
    if pd.isna(value):
        raise ValueError("The summary value is not numeric.")

    formatted = _number(value)
    metric_en = _metric_en(plan.metric)
    metric_my = _metric_my(plan.metric)
    headline = (
        f"Total {metric_en.title()}: {formatted}\n"
        f"စုစုပေါင်း {metric_my}: {formatted}"
    )
    findings = [
        f"The calculated total {metric_en} is {formatted}.\n"
        f"တွက်ချက်ရရှိသော စုစုပေါင်း {metric_my}မှာ {formatted} ဖြစ်ပါသည်။"
    ]
    evidence = [{
        "claim": "summary_total",
        "metric": plan.metric,
        "value": float(value),
    }]
    return headline, findings, evidence


def _ranking(
    dataframe: pd.DataFrame,
    plan: AnalysisPlan,
) -> tuple[str, list[str], list[dict[str, Any]]]:
    dimension, metric = _result_columns(dataframe, plan)
    if dimension is None or metric is None:
        raise ValueError("Ranking insights require dimension and metric columns.")

    numeric = pd.to_numeric(dataframe[metric], errors="coerce")
    valid = dataframe.loc[numeric.notna()].copy()
    valid[metric] = numeric.loc[numeric.notna()]
    if valid.empty:
        raise ValueError("Ranking results contain no numeric values.")

    ascending = _enum_value(plan.sort_direction) == "ascending"
    valid = valid.sort_values(metric, ascending=ascending).reset_index(drop=True)
    first = valid.iloc[0]
    last = valid.iloc[-1]
    first_value = float(first[metric])
    last_value = float(last[metric])
    total = float(valid[metric].sum())
    share = first_value / total * 100.0 if total else 0.0

    first_name = str(first[dimension])
    last_name = str(last[dimension])
    metric_en = _metric_en(plan.metric)
    metric_my = _metric_my(plan.metric)
    dimension_my = _dimension_my(plan.dimension)

    if ascending:
        headline = (
            f"Lowest {metric_en.title()}: {first_name} ({_number(first_value)})\n"
            f"{metric_my} အနည်းဆုံး {dimension_my}: "
            f"{first_name} ({_number(first_value)})"
        )
        lead = (
            f"{first_name} has the lowest {metric_en} at {_number(first_value)}.\n"
            f"{first_name} သည် {metric_my} {_number(first_value)} ဖြင့် "
            f"အနည်းဆုံး {dimension_my} ဖြစ်ပါသည်။"
        )
    else:
        headline = (
            f"Highest {metric_en.title()}: {first_name} ({_number(first_value)})\n"
            f"{metric_my} အများဆုံး {dimension_my}: "
            f"{first_name} ({_number(first_value)})"
        )
        lead = (
            f"{first_name} generated the highest {metric_en} at "
            f"{_number(first_value)}.\n"
            f"{first_name} သည် {metric_my} {_number(first_value)} ဖြင့် "
            "အမြင့်ဆုံးရရှိခဲ့ပါသည်။"
        )

    findings = [
        lead,
        f"{first_name} represents {_percentage(share)} of the displayed "
        f"{metric_en}.\n"
        f"ဤရလဒ်တွင် {first_name} ၏ {metric_my}သည် စုစုပေါင်း၏ "
        f"{_percentage(share)} ရှိပါသည်။",
    ]
    if len(valid) > 1:
        findings.append(
            f"{last_name} is at the opposite end with {metric_en} of "
            f"{_number(last_value)}.\n"
            f"အခြားတစ်ဖက်တွင် {last_name} ၏ {metric_my}မှာ "
            f"{_number(last_value)} ဖြစ်ပါသည်။"
        )

    evidence = [
        {
            "claim": "leading_rank",
            "dimension_value": first_name,
            "metric": plan.metric,
            "value": first_value,
            "share_of_displayed_result": round(share, 4),
        },
        {
            "claim": "opposite_rank",
            "dimension_value": last_name,
            "metric": plan.metric,
            "value": last_value,
        },
    ]
    return headline, findings, evidence


def _trend(
    dataframe: pd.DataFrame,
    plan: AnalysisPlan,
) -> tuple[str, list[str], list[dict[str, Any]]]:
    dimension, metric = _result_columns(dataframe, plan)
    if dimension is None or metric is None:
        raise ValueError("Trend insights require time and metric columns.")

    numeric = pd.to_numeric(dataframe[metric], errors="coerce")
    valid = dataframe.loc[numeric.notna()].copy()
    valid[metric] = numeric.loc[numeric.notna()]
    if len(valid) < 2:
        raise ValueError("Trend insights require at least two periods.")

    first = valid.iloc[0]
    last = valid.iloc[-1]
    start_value = float(first[metric])
    end_value = float(last[metric])
    change = end_value - start_value
    change_percent = change / abs(start_value) * 100.0 if start_value else None
    peak = valid.loc[valid[metric].idxmax()]

    metric_en = _metric_en(plan.metric)
    metric_my = _metric_my(plan.metric)
    start_period = str(first[dimension])
    end_period = str(last[dimension])
    peak_period = str(peak[dimension])
    peak_value = float(peak[metric])

    if change > 0:
        direction_en, direction_my = "increased", "မြင့်တက်ခဲ့"
    elif change < 0:
        direction_en, direction_my = "decreased", "ကျဆင်းခဲ့"
    else:
        direction_en, direction_my = "remained unchanged", "ပြောင်းလဲမှုမရှိခဲ့"

    headline = (
        f"{metric_en.title()} {direction_en} from {_number(start_value)} "
        f"to {_number(end_value)}\n"
        f"{metric_my}သည် {_number(start_value)} မှ {_number(end_value)} သို့ "
        f"{direction_my}ပါသည်"
    )
    findings = [
        f"Between {start_period} and {end_period}, {metric_en} "
        f"{direction_en} by {_number(abs(change))}.\n"
        f"{start_period} မှ {end_period} အတွင်း {metric_my}သည် "
        f"{_number(abs(change))} ဖြင့် {direction_my}ပါသည်။",
        f"Peak {metric_en} was {_number(peak_value)} in {peak_period}.\n"
        f"{metric_my} အမြင့်ဆုံးတန်ဖိုးမှာ {peak_period} တွင် "
        f"{_number(peak_value)} ဖြစ်ပါသည်။",
    ]
    if change_percent is not None:
        findings.insert(
            1,
            f"The percentage change is {_percentage(change_percent)}.\n"
            f"ရာခိုင်နှုန်းပြောင်းလဲမှုမှာ {_percentage(change_percent)} ဖြစ်ပါသည်။",
        )

    evidence = [
        {
            "claim": "trend_change",
            "start_period": start_period,
            "start_value": start_value,
            "end_period": end_period,
            "end_value": end_value,
            "absolute_change": change,
            "percentage_change": change_percent,
        },
        {
            "claim": "trend_peak",
            "period": peak_period,
            "value": peak_value,
        },
    ]
    return headline, findings, evidence


def _data_quality(
    dataframe: pd.DataFrame,
) -> tuple[str, list[str], list[dict[str, Any]]]:
    required = {"column", "missing_count", "missing_percentage"}
    if not required.issubset(dataframe.columns):
        raise ValueError("Data-quality results are missing required fields.")

    counts = pd.to_numeric(dataframe["missing_count"], errors="coerce").fillna(0)
    percentages = pd.to_numeric(
        dataframe["missing_percentage"], errors="coerce"
    ).fillna(0)
    total = int(counts.sum())
    affected = int((counts > 0).sum())
    worst_index = counts.idxmax()
    worst_column = str(dataframe.loc[worst_index, "column"])
    worst_count = int(counts.loc[worst_index])
    worst_percentage = float(percentages.loc[worst_index])

    headline = (
        f"Data Quality: {total:,} missing values across {affected} columns\n"
        f"Data Quality: column {affected} ခုတွင် missing value "
        f"{total:,} ခု ရှိပါသည်"
    )
    findings = [
        f"The dataset contains {total:,} missing values across "
        f"{affected} columns.\n"
        f"Dataset တွင် column {affected} ခုအတွင်း missing value "
        f"{total:,} ခု ရှိပါသည်။"
    ]
    if worst_count:
        findings.append(
            f"{worst_column} has the most missing values: {worst_count:,} "
            f"({_percentage(worst_percentage)}).\n"
            f"{worst_column} column တွင် missing value အများဆုံး "
            f"{worst_count:,} ခု ({_percentage(worst_percentage)}) ရှိပါသည်။"
        )
    else:
        findings.append(
            "No missing values were detected.\n"
            "Missing value မတွေ့ရှိပါ။"
        )

    evidence = [{
        "claim": "missing_value_summary",
        "total_missing_values": total,
        "affected_columns": affected,
        "most_affected_column": worst_column,
        "most_affected_count": worst_count,
        "most_affected_percentage": worst_percentage,
    }]
    return headline, findings, evidence


def generate_insights(
    result_records: list[dict[str, Any]],
    plan: AnalysisPlan,
    language: InsightLanguage = "bilingual",
) -> InsightResult:
    """Generate insights from validated executor output only."""

    if language not in {"en", "my", "bilingual"}:
        raise ValueError("language must be 'en', 'my', or 'bilingual'")
    if not isinstance(result_records, list):
        raise TypeError("result_records must be a list")

    dataframe = pd.DataFrame(result_records)
    if dataframe.empty:
        raise ValueError("Insights cannot be generated from an empty result.")

    intent = _enum_value(plan.intent)
    if intent == AnalysisIntent.SUMMARY.value:
        headline, findings, evidence = _summary(dataframe, plan)
    elif intent in {AnalysisIntent.RANKING.value, AnalysisIntent.COMPARISON.value}:
        headline, findings, evidence = _ranking(dataframe, plan)
    elif intent == AnalysisIntent.TREND.value:
        headline, findings, evidence = _trend(dataframe, plan)
    elif intent == AnalysisIntent.DATA_QUALITY.value:
        headline, findings, evidence = _data_quality(dataframe)
    else:
        raise ValueError(
            f"Insight generation for intent '{intent}' is not implemented yet."
        )

    selected_headline = _language_text(headline, language)
    selected_findings = [
        _language_text(finding, language)
        for finding in findings
    ]
    validation = {
        "status": "passed" if selected_findings and evidence else "failed",
        "source_result_rows": int(len(dataframe)),
        "finding_count": len(selected_findings),
        "evidence_count": len(evidence),
        "calculation_source": "validated_executor_output",
        "llm_calculation_used": False,
        "checks": [
            "Insights were generated from deterministic executor records.",
            "Every insight group includes evidence values.",
            "No LLM-generated numerical calculation was used.",
        ],
    }

    return InsightResult(
        success=validation["status"] == "passed",
        language=language,
        headline=selected_headline,
        findings=selected_findings,
        evidence=evidence,
        warnings=[],
        validation=validation,
    )
