"""Integrate revenue and profit quality gates into analysis_executor.py.

This script:

1. Locates app/services/analysis_executor.py.
2. Creates a backup before modifying the file.
3. Adds quality-gate imports.
4. Inserts require_revenue_ready() before revenue calculation.
5. Inserts require_profit_ready() before profit calculation.
6. Avoids duplicate integration.
7. Verifies the resulting source code.
8. Compiles the modified executor.

Run from the project root:

    python scripts/integrate_quality_gate.py
"""

from __future__ import annotations

import ast
import py_compile
import re
import shutil
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

EXECUTOR_PATH = (
    PROJECT_ROOT
    / "app"
    / "services"
    / "analysis_executor.py"
)

BACKUP_DIRECTORY = (
    PROJECT_ROOT
    / "backups"
)

BACKUP_PATH = (
    BACKUP_DIRECTORY
    / "analysis_executor_before_quality_gate.py"
)


QUALITY_GATE_IMPORT = """from core.analytics_quality_gate import (
    require_profit_ready,
    require_revenue_ready,
)
"""


REVENUE_GATE_CALL = """require_revenue_ready(
            dataframe=dataframe,
            minimum_usable_ratio=0.80,
            maximum_missing_ratio=0.20,
        )
"""


PROFIT_GATE_CALL = """require_profit_ready(
            dataframe=dataframe,
            minimum_usable_ratio=0.80,
            maximum_missing_ratio=0.20,
        )
"""


def read_source() -> str:
    """Read the current analysis executor."""

    if not EXECUTOR_PATH.exists():
        raise FileNotFoundError(
            "Analysis executor does not exist: "
            f"{EXECUTOR_PATH}"
        )

    source = EXECUTOR_PATH.read_text(
        encoding="utf-8-sig",
    )

    if not source.strip():
        raise ValueError(
            "analysis_executor.py is empty."
        )

    return source


def create_backup() -> None:
    """Create a backup outside the application package."""

    BACKUP_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    shutil.copy2(
        EXECUTOR_PATH,
        BACKUP_PATH,
    )

    print(
        "Backup created: "
        f"{BACKUP_PATH}"
    )


def add_quality_gate_import(
    source: str,
) -> str:
    """Add the quality-gate import after existing imports."""

    if (
        "from core.analytics_quality_gate import"
        in source
    ):
        print(
            "Quality-gate import already exists."
        )

        return source

    lines = source.splitlines(
        keepends=True,
    )

    insertion_index = 0
    inside_multiline_import = False

    for index, line in enumerate(lines):
        stripped = line.strip()

        if stripped.startswith(
            "from __future__ import"
        ):
            insertion_index = index + 1
            continue

        if stripped.startswith("import "):
            insertion_index = index + 1
            continue

        if stripped.startswith("from "):
            insertion_index = index + 1

            if stripped.endswith("("):
                inside_multiline_import = True

            continue

        if inside_multiline_import:
            insertion_index = index + 1

            if stripped == ")":
                inside_multiline_import = False

            continue

        if (
            insertion_index > 0
            and stripped
            and not stripped.startswith("#")
        ):
            break

    import_lines = [
        "\n",
        QUALITY_GATE_IMPORT,
        "\n",
    ]

    lines[
        insertion_index:insertion_index
    ] = import_lines

    print(
        "Added analytics quality-gate import."
    )

    return "".join(lines)


def find_metric_condition(
    source: str,
    metric_name: str,
) -> re.Match[str] | None:
    """Find a revenue or profit conditional branch."""

    escaped_metric = re.escape(
        metric_name
    )

    patterns = [
        rf'(?m)^(?P<indent>\s*)if\s+metric_name\s*==\s*["\']{escaped_metric}["\']\s*:\s*$',
        rf'(?m)^(?P<indent>\s*)elif\s+metric_name\s*==\s*["\']{escaped_metric}["\']\s*:\s*$',
        rf'(?m)^(?P<indent>\s*)if\s+metric\s*==\s*["\']{escaped_metric}["\']\s*:\s*$',
        rf'(?m)^(?P<indent>\s*)elif\s+metric\s*==\s*["\']{escaped_metric}["\']\s*:\s*$',
        rf'(?m)^(?P<indent>\s*)if\s+semantic_metric\s*==\s*["\']{escaped_metric}["\']\s*:\s*$',
        rf'(?m)^(?P<indent>\s*)elif\s+semantic_metric\s*==\s*["\']{escaped_metric}["\']\s*:\s*$',
    ]

    for pattern in patterns:
        match = re.search(
            pattern,
            source,
        )

        if match:
            return match

    return None


def detect_dataframe_variable(
    source: str,
) -> str:
    """Detect whether the executor uses dataframe or df."""

    function_patterns = [
        r"def\s+create_metric_series\s*\([^)]*\bdataframe\b",
        r"def\s+execute_analysis_plan\s*\([^)]*\bdataframe\b",
    ]

    for pattern in function_patterns:
        if re.search(
            pattern,
            source,
            flags=re.DOTALL,
        ):
            return "dataframe"

    df_patterns = [
        r"def\s+create_metric_series\s*\([^)]*\bdf\b",
        r"def\s+execute_analysis_plan\s*\([^)]*\bdf\b",
    ]

    for pattern in df_patterns:
        if re.search(
            pattern,
            source,
            flags=re.DOTALL,
        ):
            return "df"

    if re.search(
        r"\bdataframe\s*\[",
        source,
    ):
        return "dataframe"

    if re.search(
        r"\bdf\s*\[",
        source,
    ):
        return "df"

    raise RuntimeError(
        "Unable to determine the DataFrame variable. "
        "Expected 'dataframe' or 'df'."
    )


def build_gate_call(
    function_name: str,
    dataframe_variable: str,
    indentation: str,
) -> str:
    """Build an indented quality-gate function call."""

    body_indent = indentation + "    "

    return (
        f"{body_indent}{function_name}(\n"
        f"{body_indent}    "
        f"dataframe={dataframe_variable},\n"
        f"{body_indent}    "
        "minimum_usable_ratio=0.80,\n"
        f"{body_indent}    "
        "maximum_missing_ratio=0.20,\n"
        f"{body_indent})\n\n"
    )


def insert_gate_call(
    source: str,
    metric_name: str,
    function_name: str,
    dataframe_variable: str,
) -> str:
    """Insert one gate at the beginning of its metric branch."""

    call_marker = f"{function_name}("

    if call_marker in source:
        print(
            f"{function_name}() already exists."
        )

        return source

    condition = find_metric_condition(
        source=source,
        metric_name=metric_name,
    )

    if condition is None:
        raise RuntimeError(
            f"Could not find the '{metric_name}' "
            "calculation branch in analysis_executor.py."
        )

    line_end = source.find(
        "\n",
        condition.end(),
    )

    if line_end == -1:
        line_end = len(source)
        newline = "\n"
    else:
        line_end += 1
        newline = ""

    indentation = condition.group(
        "indent"
    )

    gate_call = build_gate_call(
        function_name=function_name,
        dataframe_variable=dataframe_variable,
        indentation=indentation,
    )

    updated = (
        source[:line_end]
        + newline
        + gate_call
        + source[line_end:]
    )

    print(
        f"Inserted {function_name}() "
        f"before {metric_name} calculation."
    )

    return updated


def validate_source(
    source: str,
) -> None:
    """Validate the modified Python source structurally."""

    required_tokens = [
        (
            "from core.analytics_quality_gate "
            "import"
        ),
        "require_revenue_ready",
        "require_profit_ready",
        "minimum_usable_ratio=0.80",
        "maximum_missing_ratio=0.20",
    ]

    missing_tokens = [
        token
        for token in required_tokens
        if token not in source
    ]

    if missing_tokens:
        raise RuntimeError(
            "Quality-gate integration is incomplete. "
            "Missing tokens: "
            + ", ".join(missing_tokens)
        )

    if source.count(
        "require_revenue_ready("
    ) != 1:
        raise RuntimeError(
            "Expected exactly one revenue gate call."
        )

    if source.count(
        "require_profit_ready("
    ) != 1:
        raise RuntimeError(
            "Expected exactly one profit gate call."
        )

    try:
        ast.parse(source)
    except SyntaxError as error:
        raise RuntimeError(
            "Modified analysis executor is not valid "
            f"Python: {error}"
        ) from error


def write_source(
    source: str,
) -> None:
    """Write the modified executor with UTF-8 encoding."""

    EXECUTOR_PATH.write_text(
        source,
        encoding="utf-8",
        newline="\n",
    )

    print(
        "Updated: "
        f"{EXECUTOR_PATH}"
    )


def compile_executor() -> None:
    """Compile the updated executor."""

    py_compile.compile(
        str(EXECUTOR_PATH),
        doraise=True,
    )

    print(
        "analysis_executor.py compilation passed."
    )


def main() -> None:
    """Run the complete quality-gate integration."""

    print("=" * 80)
    print(
        "InsightFlow AI Analytics Quality-Gate "
        "Integration"
    )
    print("=" * 80)

    source = read_source()

    dataframe_variable = (
        detect_dataframe_variable(
            source
        )
    )

    print(
        "Detected DataFrame variable: "
        f"{dataframe_variable}"
    )

    create_backup()

    source = add_quality_gate_import(
        source
    )

    source = insert_gate_call(
        source=source,
        metric_name="revenue",
        function_name=(
            "require_revenue_ready"
        ),
        dataframe_variable=(
            dataframe_variable
        ),
    )

    source = insert_gate_call(
        source=source,
        metric_name="profit",
        function_name=(
            "require_profit_ready"
        ),
        dataframe_variable=(
            dataframe_variable
        ),
    )

    validate_source(source)
    write_source(source)
    compile_executor()

    print()
    print("=" * 80)
    print(
        "QUALITY-GATE INTEGRATION PASSED"
    )
    print("=" * 80)


if __name__ == "__main__":
    main()