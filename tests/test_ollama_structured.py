"""Optional integration tests for Ollama structured analytics output.

Requirements:
1. The Ollama Python package.
2. A running local Ollama server.
3. The configured Qwen model.

The tests skip automatically when Ollama or the requested model is unavailable.
Ollama translates questions into structured plans only. Financial calculations
remain the responsibility of deterministic application code.
"""

from __future__ import annotations

import json
import os
from typing import Any, Literal

import pytest
from pydantic import BaseModel, Field


ollama = pytest.importorskip(
    "ollama",
    reason="The optional Ollama Python package is not installed.",
)

OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen3:4b")


class StructuredAnalysisPlan(BaseModel):
    """Minimal structured plan produced by the local LLM."""

    intent: Literal[
        "summary",
        "ranking",
        "trend",
        "comparison",
        "distribution",
        "data_quality",
        "correlation",
        "unknown",
    ]
    metric: str | None = None
    dimension: str | None = None
    aggregation: Literal[
        "sum",
        "mean",
        "count",
        "min",
        "max",
    ] = "sum"
    limit: int | None = Field(default=None, ge=1)
    sort_direction: Literal[
        "ascending",
        "descending",
    ] | None = None


SYSTEM_PROMPT = """
You are a strict analytics-plan parser.

Convert the user's question into exactly one structured analytics plan.
Return only JSON matching the supplied schema.

Do not calculate values.
Do not invent financial results.
Do not answer the user's question in prose.
Do not include Markdown.
Do not include explanations.

Intent definitions:

- summary:
  Use when the user asks for one aggregated value, such as total revenue,
  total profit, average price, count of rows, minimum, or maximum.

- ranking:
  Use when the user asks for top, bottom, highest, lowest, best, worst,
  largest, smallest, or a limited ordered list grouped by a dimension.

- trend:
  Use when the user asks for change over time, monthly results, yearly
  results, daily results, growth, decline, or a time series.

- comparison:
  Use when the user asks to compare two or more named groups without asking
  for a top or bottom ranking.

- distribution:
  Use for frequency, histogram, spread, ranges, or value distribution.

- data_quality:
  Use only for missing values, duplicates, invalid data, schema problems,
  completeness, or data-quality assessment.

- correlation:
  Use only for relationships or correlation between numerical variables.

- unknown:
  Use only when none of the supported analytical intentions apply.

Field rules:

1. Revenue metric:
   metric = "revenue"

2. Profit metric:
   metric = "profit"

3. "Total revenue" or "total profit":
   intent = "summary"
   aggregation = "sum"
   dimension = null
   limit = null
   sort_direction = null

4. "Top N products by revenue":
   intent = "ranking"
   metric = "revenue"
   dimension = "product"
   aggregation = "sum"
   limit = N
   sort_direction = "descending"

5. "Bottom N":
   sort_direction = "ascending"

6. Ranking always requires a dimension.

7. Summary without grouping has dimension = null.

Examples:

User: Show total revenue.
Output:
{
  "intent": "summary",
  "metric": "revenue",
  "dimension": null,
  "aggregation": "sum",
  "limit": null,
  "sort_direction": null
}

User: Show total profit.
Output:
{
  "intent": "summary",
  "metric": "profit",
  "dimension": null,
  "aggregation": "sum",
  "limit": null,
  "sort_direction": null
}

User: Show the top 5 products by revenue.
Output:
{
  "intent": "ranking",
  "metric": "revenue",
  "dimension": "product",
  "aggregation": "sum",
  "limit": 5,
  "sort_direction": "descending"
}

User: Analyze missing values and duplicate rows.
Output:
{
  "intent": "data_quality",
  "metric": null,
  "dimension": null,
  "aggregation": "count",
  "limit": null,
  "sort_direction": null
}
""".strip()


def ollama_is_available() -> bool:
    """Return True only when the local Ollama server responds."""

    try:
        ollama.list()
    except Exception:
        return False

    return True


def _model_value(model: Any, field_name: str) -> Any:
    """Read one model field from dict-like and object-like responses."""

    if isinstance(model, dict):
        return model.get(field_name)

    return getattr(model, field_name, None)


def available_model_names() -> set[str]:
    """Return model names reported by current Ollama client versions."""

    try:
        response = ollama.list()
    except Exception:
        return set()

    if isinstance(response, dict):
        models = response.get("models", [])
    else:
        models = getattr(response, "models", [])

    names: set[str] = set()

    for model in models:
        for field_name in ("model", "name"):
            candidate = _model_value(model, field_name)

            if candidate:
                names.add(str(candidate))

    return names


def model_is_available(requested_model: str) -> bool:
    """Return True when the configured model or its base name is installed."""

    available = available_model_names()
    requested_base = requested_model.split(":", maxsplit=1)[0]

    return any(
        model_name == requested_model
        or model_name.split(":", maxsplit=1)[0] == requested_base
        for model_name in available
    )


@pytest.fixture(scope="module", autouse=True)
def require_ollama_server() -> None:
    """Skip this module when Ollama cannot be reached."""

    if not ollama_is_available():
        pytest.skip(
            "Ollama is not running or accessible. Start Ollama to run "
            "local-LLM integration tests.",
            allow_module_level=True,
        )


@pytest.fixture(scope="module")
def installed_model() -> str:
    """Return the configured model or skip when unavailable."""

    if not model_is_available(OLLAMA_MODEL):
        pytest.skip(
            f"Ollama model '{OLLAMA_MODEL}' is not installed."
        )

    return OLLAMA_MODEL


def _response_content(response: Any) -> str:
    """Read assistant content across Ollama response object versions."""

    if isinstance(response, dict):
        message = response.get("message", {})
        content = (
            message.get("content", "")
            if isinstance(message, dict)
            else getattr(message, "content", "")
        )
    else:
        message = getattr(response, "message", None)
        content = getattr(message, "content", "")

    return str(content or "").strip()


def parse_with_ollama(
    model: str,
    question: str,
) -> StructuredAnalysisPlan:
    """Convert one question into a validated structured plan."""

    response = ollama.chat(
        model=model,
        messages=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": question,
            },
        ],
        format=StructuredAnalysisPlan.model_json_schema(),
        options={
            "temperature": 0,
            "seed": 42,
            "top_p": 0.1,
        },
    )

    content = _response_content(response)

    if not content:
        raise AssertionError("Ollama returned empty content.")

    try:
        parsed_json = json.loads(content)
    except json.JSONDecodeError as error:
        raise AssertionError(
            "Ollama did not return valid JSON. "
            f"Content: {content}"
        ) from error

    return StructuredAnalysisPlan.model_validate(parsed_json)


@pytest.mark.integration
@pytest.mark.ollama
def test_ollama_structured_ranking_plan(
    installed_model: str,
) -> None:
    """Verify a top-five request becomes a ranking plan."""

    plan = parse_with_ollama(
        model=installed_model,
        question="Show the top 5 products by revenue.",
    )

    assert plan.intent == "ranking"
    assert plan.metric == "revenue"
    assert plan.dimension == "product"
    assert plan.aggregation == "sum"
    assert plan.limit == 5
    assert plan.sort_direction == "descending"


@pytest.mark.integration
@pytest.mark.ollama
def test_ollama_structured_summary_plan(
    installed_model: str,
) -> None:
    """Verify total revenue becomes a summary plan."""

    plan = parse_with_ollama(
        model=installed_model,
        question="Show total revenue.",
    )

    assert plan.intent == "summary"
    assert plan.metric == "revenue"
    assert plan.dimension is None
    assert plan.aggregation == "sum"
    assert plan.limit is None
    assert plan.sort_direction is None


@pytest.mark.integration
@pytest.mark.ollama
def test_ollama_does_not_calculate_revenue(
    installed_model: str,
) -> None:
    """Verify the local LLM returns a plan and no financial result."""

    plan = parse_with_ollama(
        model=installed_model,
        question="Show total revenue.",
    )
    dumped = plan.model_dump()

    assert "value" not in dumped
    assert "result" not in dumped
    assert "revenue_total" not in dumped
    assert "calculated_value" not in dumped


@pytest.mark.integration
@pytest.mark.ollama
def test_ollama_data_quality_plan(
    installed_model: str,
) -> None:
    """Verify data-quality language maps only to data quality."""

    plan = parse_with_ollama(
        model=installed_model,
        question="Analyze missing values and duplicate rows.",
    )

    assert plan.intent == "data_quality"
    assert plan.metric is None
    assert plan.dimension is None
    assert plan.aggregation == "count"
    assert plan.limit is None
    assert plan.sort_direction is None
