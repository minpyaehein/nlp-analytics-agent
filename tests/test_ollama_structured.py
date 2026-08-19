from typing import Literal

from ollama import chat
from pydantic import BaseModel


class Plan(BaseModel):
    intent: Literal[
        "ranking",
        "trend",
        "comparison",
        "summary",
        "unknown",
    ]

    metric: str | None = None
    dimension: str | None = None

    sort_direction: Literal[
        "ascending",
        "descending",
    ] | None = None

    limit: int | None = None

    visualization: Literal[
        "bar",
        "line",
        "table",
        "kpi",
    ]


response = chat(
    model="qwen3:4b",
    messages=[
        {
            "role": "system",
            "content": (
                "Convert analytics questions into analysis plans. "
                "Never calculate results. "
                "Never invent products, values, columns, or data. "
                "Return only the requested structured plan."
            ),
        },
        {
            "role": "user",
            "content": (
                "Show the top 5 products by revenue."
            ),
        },
    ],
    format=Plan.model_json_schema(),
    options={
        "temperature": 0,
        "num_predict": 300,
    },
)


plan = Plan.model_validate_json(
    response.message.content
)

print(plan.model_dump_json(indent=2))