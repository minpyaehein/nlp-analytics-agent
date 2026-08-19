"""Local Ollama integration for structured analytics-plan extraction."""

from __future__ import annotations

import json
from typing import Literal

from ollama import ResponseError, chat
from pydantic import BaseModel, Field, ValidationError


MODEL_NAME = "qwen3:4b"


class LLMFilter(BaseModel):
    """A filter extracted from an analytics question."""

    column: str
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
    value: str | int | float | list[str] | list[int] | list[float]


class LLMAnalysisPlan(BaseModel):
    """Structured plan produced by the local language model."""

    intent: Literal[
        "ranking",
        "trend",
        "comparison",
        "summary",
        "anomaly",
        "data_quality",
        "correlation",
        "distribution",
        "unknown",
    ]
    metric: str | None = None
    dimension: str | None = None
    aggregation: Literal[
        "sum",
        "mean",
        "median",
        "min",
        "max",
        "count",
        "count_distinct",
    ] = "sum"
    sort_direction: Literal[
        "ascending",
        "descending",
    ] | None = None
    limit: int | None = Field(default=None, ge=1, le=1000)
    visualization: Literal[
        "bar",
        "line",
        "scatter",
        "histogram",
        "table",
        "kpi",
    ] = "table"
    filters: list[LLMFilter] = Field(default_factory=list)
    explanation: str = Field(
        description="A short explanation of the analytical interpretation."
    )


SYSTEM_PROMPT = """
You convert English, Myanmar Burmese, and mixed-language analytics
questions into structured analysis plans.

Never calculate numerical results.
Never invent products, values, records, or dataset columns.
Return only data matching the supplied JSON schema.

Canonical metrics include:
revenue, profit, cost, quantity, order_count, customer_count,
average_order_value, discount.

Canonical dimensions include:
product, category, region, customer, date, month, quarter, year,
channel.

Intent rules:
- top, highest, best, most = ranking
- bottom, lowest, worst, least = ranking
- compare, versus, vs, difference = comparison
- trend, monthly, yearly, over time = trend
- total, overall, sum = summary
- missing, duplicate, null = data_quality
- correlation, relationship = correlation
- distribution, histogram, spread = distribution

Ranking rules:
- top, highest, best, most = descending
- bottom, lowest, worst, least = ascending
- ranking visualization = bar
- extract the requested ranking number as limit

Visualization rules:
- ranking = bar
- comparison = bar
- trend = line
- summary = kpi
- anomaly = scatter
- distribution = histogram
- data_quality = table
- correlation = scatter

Myanmar terminology:
- အမြတ် = profit
- ဝင်ငွေ = revenue
- ရောင်းရငွေ = revenue
- ကုန်ပစ္စည်း = product
- အမျိုးအစား = category
- ဒေသ = region
- ဖောက်သည် = customer
- အများဆုံး = top or highest
- အနည်းဆုံး = bottom or lowest
- လစဉ် = monthly
- နှိုင်းယှဉ် = compare

Examples:

Question: Show the top 5 products by revenue.
Plan: intent=ranking, metric=revenue, dimension=product,
sort_direction=descending, limit=5, visualization=bar.

Question: Show the bottom 3 regions by profit.
Plan: intent=ranking, metric=profit, dimension=region,
sort_direction=ascending, limit=3, visualization=bar.

Question: Show monthly revenue trend.
Plan: intent=trend, metric=revenue, dimension=month,
sort_direction=null, limit=null, visualization=line.

Question: Analyze missing values and duplicate rows.
Plan: intent=data_quality, metric=null, dimension=null,
sort_direction=null, limit=null, visualization=table.
"""


def check_ollama_connection() -> bool:
    """Return True when the configured local Ollama model responds."""

    try:
        response = chat(
            model=MODEL_NAME,
            messages=[
                {
                    "role": "user",
                    "content": "Reply with exactly the word ready.",
                }
            ],
            options={
                "temperature": 0,
                "num_predict": 10,
            },
        )
        return bool(response.message.content.strip())
    except Exception:
        return False


def correct_plan(
    question: str,
    plan: LLMAnalysisPlan,
) -> LLMAnalysisPlan:
    """Apply safe deterministic corrections to common model mistakes."""

    normalized = question.casefold()

    descending_terms = [
        "top",
        "highest",
        "best",
        "most",
        "အများဆုံး",
    ]
    ascending_terms = [
        "bottom",
        "lowest",
        "worst",
        "least",
        "အနည်းဆုံး",
    ]

    if any(term in normalized for term in descending_terms):
        plan.intent = "ranking"
        plan.sort_direction = "descending"
        plan.visualization = "bar"

    if any(term in normalized for term in ascending_terms):
        plan.intent = "ranking"
        plan.sort_direction = "ascending"
        plan.visualization = "bar"

    if any(term in normalized for term in ["trend", "monthly", "yearly"]):
        plan.intent = "trend"
        plan.visualization = "line"

    if any(term in normalized for term in ["missing", "duplicate", "null"]):
        plan.intent = "data_quality"
        plan.visualization = "table"

    return plan


def parse_question_with_llm(question: str) -> LLMAnalysisPlan:
    """Convert a bilingual analytics question into a validated plan."""

    if not isinstance(question, str):
        raise TypeError("question must be a string")

    cleaned_question = question.strip()
    if not cleaned_question:
        raise ValueError("Question cannot be empty.")

    schema = LLMAnalysisPlan.model_json_schema()
    user_prompt = (
        "Convert the following request into an analytics plan.\n\n"
        f"User request:\n{cleaned_question}\n\n"
        "Do not calculate or invent any result.\n"
        "Required JSON schema:\n"
        f"{json.dumps(schema, ensure_ascii=False)}"
    )

    try:
        response = chat(
            model=MODEL_NAME,
            messages=[
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT.strip(),
                },
                {
                    "role": "user",
                    "content": user_prompt,
                },
            ],
            format=schema,
            options={
                "temperature": 0,
                "num_predict": 500,
            },
        )

        plan = LLMAnalysisPlan.model_validate_json(
            response.message.content
        )
        return correct_plan(cleaned_question, plan)

    except ResponseError as error:
        raise RuntimeError(
            f"Ollama returned an error: {error}"
        ) from error
    except ValidationError as error:
        raise RuntimeError(
            "The local model returned JSON that did not match the "
            f"required schema: {error}"
        ) from error
    except Exception as error:
        raise RuntimeError(
            "Unable to process the question with local Ollama. "
            "Make sure Ollama is running and qwen3:4b is installed. "
            f"Original error: {error}"
        ) from error


if __name__ == "__main__":
    test_questions = [
        "Show the top 5 products by revenue.",
        "အမြတ်အများဆုံး ကုန်ပစ္စည်း ၅ ခုကို ပြပါ",
        "Show monthly revenue trend.",
        "Analyze missing values and duplicate rows.",
    ]

    print(f"Model: {MODEL_NAME}")
    print(f"Connection available: {check_ollama_connection()}")

    for test_question in test_questions:
        print("=" * 80)
        print(f"Question: {test_question}")

        try:
            result = parse_question_with_llm(test_question)
            print(result.model_dump_json(indent=2))
        except RuntimeError as error:
            print(f"ERROR: {error}")
