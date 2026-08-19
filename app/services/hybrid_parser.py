"""
Hybrid analytics question parser.

The deterministic parser handles familiar requests first.
Qwen3 4B is used only when deterministic parsing is incomplete.
"""

from __future__ import annotations

from typing import Any

from app.agents.question_parser import parse_question
from app.models.analysis_plan import AnalysisIntent
from app.services.local_llm import parse_question_with_llm


CONFIDENCE_THRESHOLD = 0.85


def should_use_local_llm(rule_plan: Any) -> bool:
    """
    Determine whether the local LLM fallback is required.
    """

    return (
        rule_plan.confidence < CONFIDENCE_THRESHOLD
        or bool(rule_plan.warnings)
        or rule_plan.intent == AnalysisIntent.UNKNOWN
        or (
            rule_plan.metric is None
            and rule_plan.intent
            not in {
                AnalysisIntent.DATA_QUALITY,
                AnalysisIntent.CORRELATION,
            }
        )
    )


def parse_question_hybrid(
    question: str,
) -> dict[str, Any]:
    """
    Parse a question with deterministic rules first.

    Qwen3 4B is used only if the deterministic plan is incomplete.
    """

    rule_plan = parse_question(question)

    if not should_use_local_llm(rule_plan):
        return {
            "source": "rule_based",
            "fallback_used": False,
            "plan": rule_plan.model_dump(mode="json"),
            "rule_plan": rule_plan.model_dump(mode="json"),
            "llm_plan": None,
            "llm_error": None,
        }

    try:
        llm_plan = parse_question_with_llm(question)

        return {
            "source": "local_llm",
            "fallback_used": True,
            "plan": llm_plan.model_dump(mode="json"),
            "rule_plan": rule_plan.model_dump(mode="json"),
            "llm_plan": llm_plan.model_dump(mode="json"),
            "llm_error": None,
        }

    except RuntimeError as error:
        return {
            "source": "rule_based_fallback",
            "fallback_used": True,
            "plan": rule_plan.model_dump(mode="json"),
            "rule_plan": rule_plan.model_dump(mode="json"),
            "llm_plan": None,
            "llm_error": str(error),
        }


if __name__ == "__main__":
    test_questions = [
        "Show the top 5 products by revenue.",
        "Analyze missing values and duplicate rows.",
        (
            "Which business areas need management attention "
            "because performance is becoming weaker?"
        ),
        (
            "ဘယ်ဒေသတွေမှာ လုပ်ငန်းစွမ်းဆောင်ရည် "
            "အားနည်းနေသလဲ"
        ),
    ]

    for test_question in test_questions:
        result = parse_question_hybrid(test_question)

        print("=" * 80)
        print(f"Question: {test_question}")
        print(f"Source: {result['source']}")
        print(
            f"Fallback used: "
            f"{result['fallback_used']}"
        )
        print(f"Plan: {result['plan']}")

        if result["llm_error"]:
            print(f"LLM error: {result['llm_error']}")