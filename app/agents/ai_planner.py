"""Qwen-first analytics planner for InsightFlow AI.

The local LLM interprets English, Myanmar, and mixed-language questions. The
LLM selects analytical intent, metric, dimension, filters, ordering, limits,
and visualization. Python reconstructs an approved tool sequence, validates the
plan, and grounds every selected field to the current DataFrame.

The planner never calculates business values. Numerical work remains in trusted
Pandas-based application services.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Literal

import pandas as pd
from pydantic import (
    BaseModel,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)

from app.agents.question_parser import parse_question

try:
    import ollama
except ImportError:  # pragma: no cover
    ollama = None


DEFAULT_MODEL = os.getenv("OLLAMA_MODEL", "qwen3:4b")

SUPPORTED_TOOLS = {
    "inspect_schema",
    "profile_dataset",
    "apply_filters",
    "calculate_summary",
    "calculate_ranking",
    "calculate_trend",
    "calculate_distribution",
    "calculate_correlation",
    "analyze_data_quality",
    "validate_pdf_quality",
    "generate_chart",
}

CALCULATION_TOOL_BY_INTENT = {
    "summary": "calculate_summary",
    "ranking": "calculate_ranking",
    "trend": "calculate_trend",
    "comparison": "calculate_ranking",
    "distribution": "calculate_distribution",
    "data_quality": "analyze_data_quality",
    "correlation": "calculate_correlation",
}

CONFIDENCE_ALIASES = {
    "very_low": 0.20,
    "low": 0.35,
    "medium": 0.60,
    "moderate": 0.60,
    "high": 0.80,
    "strong": 0.90,
    "very_high": 0.95,
    "certain": 0.95,
}


class PlannerFilter(BaseModel):
    """One validated dataset filter selected by the local planner."""

    column: str = Field(min_length=1)
    operator: Literal[
        "equals",
        "not_equals",
        "greater_than",
        "greater_than_or_equal",
        "less_than",
        "less_than_or_equal",
        "contains",
        "in",
        "between",
    ] = "equals"
    value: Any


class ToolStep(BaseModel):
    """One approved deterministic tool invocation."""

    tool: str = Field(min_length=1)
    reason: str = Field(min_length=1)

    @field_validator("tool")
    @classmethod
    def validate_tool(cls, value: str) -> str:
        normalized = value.strip()

        if normalized not in SUPPORTED_TOOLS:
            raise ValueError(
                f"Unsupported tool '{normalized}'. Allowed tools: "
                f"{sorted(SUPPORTED_TOOLS)}"
            )

        return normalized


class AIAnalysisPlan(BaseModel):
    """Structured and validated output from the local Qwen planner."""

    requires_clarification: bool = False
    clarification_question: str | None = None
    language: Literal["en", "my", "mixed", "unknown"] = "unknown"
    intent: Literal[
        "summary",
        "ranking",
        "trend",
        "comparison",
        "distribution",
        "data_quality",
        "correlation",
        "unknown",
    ] = "unknown"
    metric: str | None = None
    dimension: str | None = None
    aggregation: Literal[
        "sum",
        "mean",
        "count",
        "min",
        "max",
    ] = "sum"
    filters: list[PlannerFilter] = Field(default_factory=list)
    sort_direction: Literal[
        "ascending",
        "descending",
    ] | None = None
    limit: int | None = Field(default=None, ge=1, le=1000)
    visualization: Literal[
        "kpi",
        "bar",
        "line",
        "histogram",
        "scatter",
        "table",
    ] = "table"
    tool_steps: list[ToolStep] = Field(default_factory=list)
    reasoning_summary: str = Field(
        default=(
            "The local AI planner selected a grounded analytical "
            "operation using the uploaded dataset schema."
        ),
        min_length=1,
    )
    assumptions: list[str] = Field(default_factory=list)
    confidence: float = Field(default=0.75, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def validate_consistency(self) -> "AIAnalysisPlan":
        if self.requires_clarification:
            if not self.clarification_question:
                raise ValueError(
                    "clarification_question is required when "
                    "requires_clarification is true."
                )

            if self.tool_steps:
                raise ValueError(
                    "A clarification plan must not execute tools."
                )

            return self

        metric_intents = {
            "summary",
            "ranking",
            "trend",
            "comparison",
            "distribution",
        }

        if self.intent in metric_intents and not self.metric:
            raise ValueError(
                f"Intent '{self.intent}' requires a metric."
            )

        if (
            self.intent in {"ranking", "trend", "comparison"}
            and not self.dimension
        ):
            raise ValueError(
                f"Intent '{self.intent}' requires a dimension."
            )

        if self.intent == "ranking" and self.sort_direction is None:
            raise ValueError("Ranking requires sort_direction.")

        if not self.tool_steps:
            raise ValueError(
                "A non-clarification plan requires tool_steps."
            )

        return self


@dataclass
class AIPlannerResult:
    """Planner result with model and fallback provenance."""

    plan: AIAnalysisPlan
    source: str
    model: str
    fallback_reason: str | None = None
    raw_content: str | None = None

    def metadata(self) -> dict[str, Any]:
        """Return JSON-friendly planner metadata."""

        return {
            "source": self.source,
            "model": self.model,
            "fallback_reason": self.fallback_reason,
            "plan": self.plan.model_dump(mode="json"),
        }


def _json_safe(value: Any) -> Any:
    """Convert Pandas and NumPy values to JSON-compatible values."""

    if value is None:
        return None

    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass

    if isinstance(value, pd.Timestamp):
        return value.isoformat()

    if hasattr(value, "item"):
        try:
            return value.item()
        except (TypeError, ValueError):
            pass

    return value


def build_dataset_context(
    dataframe: pd.DataFrame,
    *,
    maximum_sample_values: int = 5,
) -> dict[str, Any]:
    """Build compact schema context without sending the full dataset."""

    if not isinstance(dataframe, pd.DataFrame):
        raise TypeError("dataframe must be a pandas DataFrame")

    columns: list[dict[str, Any]] = []

    for column in dataframe.columns:
        series = dataframe[column]
        samples = [
            _json_safe(value)
            for value in (
                series.dropna()
                .drop_duplicates()
                .head(maximum_sample_values)
                .tolist()
            )
        ]

        columns.append(
            {
                "name": str(column),
                "dtype": str(series.dtype),
                "non_null_count": int(series.notna().sum()),
                "missing_count": int(series.isna().sum()),
                "unique_count": int(series.nunique(dropna=True)),
                "sample_values": samples,
            }
        )

    return {
        "row_count": int(len(dataframe)),
        "column_count": int(len(dataframe.columns)),
        "columns": columns,
        "supported_derived_metrics": {
            "revenue": "quantity * unit_price",
            "profit": "quantity * (unit_price - unit_cost)",
        },
        "approved_tools": sorted(SUPPORTED_TOOLS),
    }


SYSTEM_PROMPT = """
You are the primary planning agent for InsightFlow AI, a local bilingual data
analytics application. Convert the user question and dataset schema into one
analysis plan. Return only one JSON object. Do not repeat the input.

The JSON object must use these top-level fields:
requires_clarification, clarification_question, language, intent, metric,
dimension, aggregation, filters, sort_direction, limit, visualization,
tool_steps, reasoning_summary, assumptions, confidence.

Never copy user_question, dataset_context, columns, approved_tools, or
supported_derived_metrics into the output.

Safety rules:
1. Never calculate, estimate, or invent a business value.
2. Never include totals, percentages, ranked values, or a business answer.
3. Select only tools listed in approved_tools.
4. Use only dataset columns, except supported derived metrics.
5. Ask one concise clarification question instead of guessing ambiguity.
6. Best, performance, weak, strong, successful, and attention are ambiguous
   unless the user states the metric.
7. Never assume currency, units, date meaning, or business definitions.
8. Revenue is quantity * unit_price only when those fields are available.
9. Profit is quantity * (unit_price - unit_cost). It is gross transaction
   profit, not net profit.
10. reasoning_summary describes planning logic only. Do not include hidden
    chain-of-thought or calculated results.
11. confidence must be a number between 0 and 1, not a word.

Planning rules:
- summary: one aggregate, no grouping, visualization=kpi.
- ranking: grouped ordered result, dimension and sort required, bar.
- trend: time grouping, line.
- comparison: grouped comparison, bar.
- distribution: histogram.
- correlation: scatter.
- data_quality: table.

When requires_clarification is false, tool_steps is mandatory. Each tool step
contains exactly tool and reason. Use only these tool names:
inspect_schema, profile_dataset, apply_filters, calculate_summary,
calculate_ranking, calculate_trend, calculate_distribution,
calculate_correlation, analyze_data_quality, validate_pdf_quality,
generate_chart.

When requires_clarification is true, tool_steps must be an empty list.
""".strip()


def ollama_is_available() -> bool:
    """Return whether the Ollama client and local server are available."""

    if ollama is None:
        return False

    try:
        ollama.list()
    except Exception:
        return False

    return True


def _response_content(response: Any) -> str:
    """Read content from dictionary or object Ollama responses."""

    if isinstance(response, dict):
        message = response.get("message", {})

        if isinstance(message, dict):
            return str(
                message.get("content", "") or ""
            ).strip()

        return str(
            getattr(message, "content", "") or ""
        ).strip()

    message = getattr(response, "message", None)

    return str(
        getattr(message, "content", "") or ""
    ).strip()


def build_tool_steps(
    plan_data: dict[str, Any],
) -> list[dict[str, str]]:
    """Build approved tools without calculating or inventing business data."""

    if plan_data.get("requires_clarification", False):
        return []

    intent = str(plan_data.get("intent", "unknown"))

    steps: list[dict[str, str]] = [
        {
            "tool": "inspect_schema",
            "reason": (
                "Verify selected fields against the uploaded dataset."
            ),
        }
    ]

    if plan_data.get("filters"):
        steps.append(
            {
                "tool": "apply_filters",
                "reason": (
                    "Apply filters selected by the local AI planner."
                ),
            }
        )

    calculation_tool = CALCULATION_TOOL_BY_INTENT.get(intent)

    if calculation_tool:
        steps.append(
            {
                "tool": calculation_tool,
                "reason": (
                    "Execute the AI-selected analysis using "
                    "deterministic application code."
                ),
            }
        )

    visualization = str(
        plan_data.get("visualization", "table")
    )

    if calculation_tool and visualization != "table":
        steps.append(
            {
                "tool": "generate_chart",
                "reason": (
                    "Visualize the validated analytical result."
                ),
            }
        )

    return steps


def _normalize_confidence(value: Any) -> float:
    """Normalize numeric and descriptive confidence values to 0..1."""

    if isinstance(value, str):
        normalized = (
            value.strip()
            .casefold()
            .replace(" ", "_")
            .replace("-", "_")
        )

        if normalized in CONFIDENCE_ALIASES:
            confidence = CONFIDENCE_ALIASES[normalized]
        else:
            try:
                confidence = float(normalized)
            except ValueError:
                confidence = 0.75
    elif isinstance(value, (int, float)):
        confidence = float(value)
    else:
        confidence = 0.75

    return max(0.0, min(1.0, confidence))


def normalize_qwen_plan_data(
    content: str,
) -> dict[str, Any]:
    """Normalize safe metadata before strict Pydantic validation."""

    try:
        plan_data = json.loads(content)
    except json.JSONDecodeError as error:
        raise RuntimeError(
            "Qwen did not return valid JSON. "
            f"Content: {content}"
        ) from error

    if not isinstance(plan_data, dict):
        raise RuntimeError(
            "Qwen returned JSON, but the planning response "
            "is not an object."
        )

    if (
        "user_question" in plan_data
        or "dataset_context" in plan_data
    ):
        raise RuntimeError(
            "Qwen repeated the input payload instead of "
            "returning an analysis plan."
        )

    plan_data.setdefault("requires_clarification", False)
    plan_data.setdefault("clarification_question", None)
    plan_data.setdefault("language", "unknown")
    plan_data.setdefault("intent", "unknown")
    plan_data.setdefault("metric", None)
    plan_data.setdefault("dimension", None)
    plan_data.setdefault("aggregation", "sum")
    plan_data.setdefault("filters", [])
    plan_data.setdefault("sort_direction", None)
    plan_data.setdefault("limit", None)
    plan_data.setdefault("visualization", "table")
    plan_data.setdefault("assumptions", [])
    plan_data.setdefault(
        "reasoning_summary",
        (
            "The local AI planner selected a grounded analytical "
            "operation using the uploaded dataset schema."
        ),
    )

    requires_clarification = bool(
        plan_data.get("requires_clarification", False)
    )
    plan_data["requires_clarification"] = (
        requires_clarification
    )

    clarification_question = plan_data.get(
        "clarification_question"
    )

    if isinstance(clarification_question, str):
        clarification_question = (
            clarification_question.strip() or None
        )

    if requires_clarification:
        plan_data["clarification_question"] = (
            clarification_question
            or (
                "Which metric would you like to use "
                "for this analysis?"
            )
        )
        plan_data["tool_steps"] = []
    else:
        plan_data["clarification_question"] = None

        # Never execute arbitrary tool names generated by the LLM.
        # Reconstruct the approved sequence from validated plan fields.
        plan_data["tool_steps"] = build_tool_steps(
            plan_data
        )

    plan_data["confidence"] = _normalize_confidence(
        plan_data.get("confidence", 0.75)
    )

    return plan_data


def _ground_plan_to_dataframe(
    plan: AIAnalysisPlan,
    dataframe: pd.DataFrame,
) -> None:
    """Reject AI-selected fields that are not grounded to the DataFrame."""

    if plan.requires_clarification:
        return

    columns = {
        str(column).strip().casefold()
        for column in dataframe.columns
    }
    allowed_metrics = columns | {"revenue", "profit"}
    temporal_dimensions = {
        "date",
        "month",
        "quarter",
        "year",
    }

    if (
        plan.metric
        and plan.metric.strip().casefold()
        not in allowed_metrics
    ):
        raise ValueError(
            f"AI-selected metric '{plan.metric}' is not "
            "grounded to the dataset."
        )

    if (
        plan.dimension
        and plan.dimension.strip().casefold()
        not in columns | temporal_dimensions
    ):
        raise ValueError(
            f"AI-selected dimension '{plan.dimension}' is not "
            "grounded to the dataset."
        )

    for condition in plan.filters:
        if condition.column.strip().casefold() not in columns:
            raise ValueError(
                f"AI-selected filter column '{condition.column}' "
                "does not exist in the dataset."
            )


def plan_with_qwen(
    question: str,
    dataframe: pd.DataFrame,
    *,
    model: str = DEFAULT_MODEL,
) -> AIPlannerResult:
    """Create, normalize, validate, and ground a local Qwen plan."""

    if not question or not question.strip():
        raise ValueError("question must not be empty")

    if not ollama_is_available():
        raise RuntimeError(
            "The local Ollama server is unavailable."
        )

    columns = [str(column) for column in dataframe.columns]
    column_types = {
        str(column): str(dtype)
        for column, dtype in dataframe.dtypes.items()
    }

    payload = {
        "question": question.strip(),
        "available_columns": columns,
        "column_types": column_types,
        "row_count": int(len(dataframe)),
    }

    user_content = (
        "Return exactly one analytics-plan JSON object. "
        "Do not repeat the input payload. "
        "Do not return dataset rows, sample values, column profiles, "
        "dataset context, Markdown, prose, or explanations. "
        "Do not calculate result values. "
        "For one total or aggregate, use intent='summary'. "
        "For top, bottom, highest, lowest, best, or worst requests, "
        "use intent='ranking'. "
        "For a request over time, use intent='trend'. "
        "For missing values or duplicates, use intent='data_quality'. "
        "For 'top 5 products by revenue', use metric='revenue', "
        "dimension='product', aggregation='sum', limit=5, "
        "sort_direction='descending', and visualization='bar'. "
        "For 'total revenue', use intent='summary', metric='revenue', "
        "dimension=null, aggregation='sum', limit=null, "
        "sort_direction=null, and visualization='kpi'. "
        "Required output fields are requires_clarification, "
        "clarification_question, language, intent, metric, dimension, "
        "aggregation, filters, sort_direction, limit, visualization, "
        "tool_steps, reasoning_summary, assumptions, and confidence."
        "\n\nPlanning input:\n"
        + json.dumps(
            payload,
            ensure_ascii=False,
            default=str,
        )
    )

    response = ollama.chat(
        model=model,
        messages=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": user_content,
            },
        ],
        format="json",
        think=False,
        options={
            "temperature": 0,
            "seed": 42,
            "top_p": 0.1,
            "num_predict": 512,
        },
    )

    content = _response_content(response)


    if not content:
        raise RuntimeError(
            "Qwen returned an empty planning response."
        )

    if os.getenv("INSIGHTFLOW_DEBUG_AI") == "1":
        print("\n" + "=" * 80)
        print("RAW QWEN PLANNER RESPONSE")
        print("=" * 80)
        print(content)
        print("=" * 80)

    plan_data = normalize_qwen_plan_data(content)

    try:
        validated_plan = AIAnalysisPlan.model_validate(
            plan_data
        )
    except ValidationError as error:
        raise RuntimeError(
            "Qwen returned a plan that failed Pydantic "
            "validation after safe normalization: "
            f"{error}"
        ) from error

    _ground_plan_to_dataframe(
        validated_plan,
        dataframe,
    )

    return AIPlannerResult(
        plan=validated_plan,
        source="local_qwen",
        model=model,
        fallback_reason=None,
        raw_content=content,
    )


def _enum_value(value: Any) -> Any:
    """Return an enum value while accepting plain strings."""

    return getattr(value, "value", value)


def _fallback_plan(
    question: str,
) -> AIAnalysisPlan:
    """Convert the deterministic parser output to AIAnalysisPlan."""

    rule_plan = parse_question(question)
    intent = str(_enum_value(rule_plan.intent))
    visualization = str(
        _enum_value(rule_plan.visualization)
    )

    requires_clarification = (
        intent == "unknown"
        or (
            intent
            in {
                "summary",
                "ranking",
                "trend",
                "comparison",
                "distribution",
            }
            and rule_plan.metric is None
        )
    )

    plan_data: dict[str, Any] = {
        "requires_clarification": (
            requires_clarification
        ),
        "clarification_question": (
            "Which metric and comparison would you like "
            "to analyze?"
            if requires_clarification
            else None
        ),
        "language": str(
            _enum_value(rule_plan.language)
        ),
        "intent": intent,
        "metric": rule_plan.metric,
        "dimension": rule_plan.dimension,
        "aggregation": rule_plan.aggregation,
        "filters": [
            item.model_dump()
            for item in rule_plan.filters
        ],
        "sort_direction": (
            str(_enum_value(rule_plan.sort_direction))
            if rule_plan.sort_direction is not None
            else None
        ),
        "limit": rule_plan.limit,
        "visualization": visualization,
        "reasoning_summary": (
            "The offline deterministic parser created "
            "this fallback plan."
        ),
        "assumptions": [],
        "confidence": float(rule_plan.confidence),
    }

    plan_data["tool_steps"] = build_tool_steps(
        plan_data
    )

    return AIAnalysisPlan.model_validate(plan_data)


def create_ai_plan(
    question: str,
    dataframe: pd.DataFrame,
    *,
    model: str = DEFAULT_MODEL,
    allow_rule_fallback: bool = True,
) -> AIPlannerResult:
    """Use Qwen first and deterministic rules only as optional fallback."""

    try:
        return plan_with_qwen(
            question,
            dataframe,
            model=model,
        )
    except Exception as error:
        if not allow_rule_fallback:
            raise

        return AIPlannerResult(
            plan=_fallback_plan(question),
            source="rule_fallback",
            model=model,
            fallback_reason=str(error),
            raw_content=None,
        )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "Create a Qwen-first analysis plan for a CSV dataset."
        )
    )
    parser.add_argument("csv_path")
    parser.add_argument("question")
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
    )
    parser.add_argument(
        "--no-fallback",
        action="store_true",
    )

    arguments = parser.parse_args()
    frame = pd.read_csv(arguments.csv_path)

    result = create_ai_plan(
        arguments.question,
        frame,
        model=arguments.model,
        allow_rule_fallback=(
            not arguments.no_fallback
        ),
    )

    print(
        json.dumps(
            result.metadata(),
            ensure_ascii=False,
            indent=2,
            default=str,
        )
    )
