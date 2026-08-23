"""Patch the production AI planner to use a minimal Qwen payload.

Run from the project root:
    python scripts/install_ai_planner_minimal_payload_fix.py

The patch preserves all existing planner models, normalization, grounding,
AIPlannerResult return behavior, deterministic fallback, and CLI handling.
Only the payload, prompt, and Ollama request inside plan_with_qwen() change.
"""

from __future__ import annotations

import ast
import py_compile
import re
import shutil
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PLANNER_PATH = PROJECT_ROOT / "app" / "agents" / "ai_planner.py"
BACKUP_DIRECTORY = PROJECT_ROOT / "backups"

NEW_REQUEST_BLOCK = '''    columns = [str(column) for column in dataframe.columns]
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
        "\\n\\nPlanning input:\\n"
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
'''


def create_backup() -> Path:
    """Create a timestamped backup of the valid planner."""

    BACKUP_DIRECTORY.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = BACKUP_DIRECTORY / f"ai_planner_before_minimal_payload_{timestamp}.py"
    shutil.copy2(PLANNER_PATH, backup)
    return backup


def top_level_function_bounds(source: str, name: str) -> tuple[int, int]:
    """Return character bounds for one top-level function."""

    match = re.search(rf"(?m)^def {re.escape(name)}\s*\(", source)
    if match is None:
        raise RuntimeError(f"Could not locate {name}().")

    remainder = source[match.end() :]
    next_definition = re.search(r"(?m)^(?:def|class)\s+", remainder)
    end = match.end() + next_definition.start() if next_definition else len(source)
    return match.start(), end


def patch_plan_with_qwen(source: str) -> str:
    """Replace only the existing Qwen request block."""

    function_start, function_end = top_level_function_bounds(
        source,
        "plan_with_qwen",
    )
    function_source = source[function_start:function_end]

    already_fixed = (
        '"available_columns": columns' in function_source
        and "think=False" in function_source
        and '"dataset_context": build_dataset_context(dataframe)' not in function_source
    )
    if already_fixed:
        return source

    payload_start = function_source.find("    payload = {")
    if payload_start < 0:
        raise RuntimeError("Could not locate the existing planner payload block.")

    content_statement = "    content = _response_content(response)"
    content_start = function_source.find(content_statement, payload_start)
    if content_start < 0:
        raise RuntimeError("Could not locate the planner response-content statement.")

    content_end = content_start + len(content_statement)

    patched_function = (
        function_source[:payload_start]
        + NEW_REQUEST_BLOCK
        + function_source[content_end:]
    )

    return source[:function_start] + patched_function + source[function_end:]


def validate_source(source: str) -> None:
    """Validate preservation, integration, and Python syntax."""

    required = [
        "class AIAnalysisPlan",
        "class AIPlannerResult",
        "def normalize_qwen_plan_data(",
        "def plan_with_qwen(",
        "def create_ai_plan(",
        '"available_columns": columns',
        '"column_types": column_types',
        "think=False",
        '"num_predict": 512',
        "plan_data = normalize_qwen_plan_data(content)",
        "validated_plan = AIAnalysisPlan.model_validate(",
        "_ground_plan_to_dataframe(",
        "return AIPlannerResult(",
        'source="local_qwen"',
    ]

    missing = [token for token in required if token not in source]
    if missing:
        raise RuntimeError("Incomplete planner patch. Missing: " + ", ".join(missing))

    function_start, function_end = top_level_function_bounds(
        source,
        "plan_with_qwen",
    )
    planner_function = source[function_start:function_end]

    if '"dataset_context": build_dataset_context(dataframe)' in planner_function:
        raise RuntimeError("The production planner still sends full dataset context.")

    if planner_function.count("ollama.chat(") != 1:
        raise RuntimeError("Expected exactly one Ollama call in plan_with_qwen().")

    if planner_function.count("think=False") != 1:
        raise RuntimeError("The production planner must disable Qwen thinking.")

    ast.parse(source)


def main() -> None:
    """Apply the minimal-payload production planner fix."""

    print("=" * 80)
    print("InsightFlow AI Production Planner Minimal-Payload Fix")
    print("=" * 80)

    if not PLANNER_PATH.exists():
        raise FileNotFoundError(f"AI planner not found: {PLANNER_PATH}")

    original = PLANNER_PATH.read_text(encoding="utf-8-sig")
    ast.parse(original)

    modified = patch_plan_with_qwen(original)
    validate_source(modified)

    if modified == original:
        py_compile.compile(str(PLANNER_PATH), doraise=True)
        print("The production planner minimal-payload fix already exists.")
        print("No changes were required.")
        return

    backup = create_backup()
    print(f"Backup created: {backup}")

    PLANNER_PATH.write_bytes(modified.encode("utf-8"))
    py_compile.compile(str(PLANNER_PATH), doraise=True)

    print(f"Updated: {PLANNER_PATH}")
    print("ai_planner.py compilation passed.")
    print()
    print("=" * 80)
    print("PRODUCTION PLANNER MINIMAL-PAYLOAD FIX PASSED")
    print("=" * 80)


if __name__ == "__main__":
    main()
