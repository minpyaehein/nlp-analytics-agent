"""Fix stale PDF quality gates in the unified-loader Streamlit app.

Run from the project root:
    python scripts/install_unified_loader_quality_gate_fix.py

This installer updates app/frontend/streamlit_app.py so that revenue and
profit readiness are recalculated from the exact DataFrame passed to
execute_filtered_analysis(). Upload-result readiness remains available as
processing evidence, but it no longer decides calculation eligibility.
"""

from __future__ import annotations

import ast
import py_compile
import re
import shutil
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
APP_PATH = PROJECT_ROOT / "app" / "frontend" / "streamlit_app.py"
BACKUP_DIRECTORY = PROJECT_ROOT / "backups"

QUALITY_IMPORT = '''from core.analytics_quality_gate import (
    validate_profit_readiness,
    validate_revenue_readiness,
)
'''

NEW_GATE_FUNCTION = '''def enforce_metric_quality_gate(
    plan: AnalysisPlan,
    dataframe: pd.DataFrame,
    upload_result: Any,
) -> dict[str, Any] | None:
    """Validate the exact current DataFrame before financial calculation.

    PDF/OCR readiness stored in ``upload_result`` is retained as provenance,
    but it is not trusted as the final calculation gate. This function always
    recalculates readiness from ``dataframe``, which is the same object passed
    to the deterministic executor.
    """

    metric = (plan.metric or "").strip().casefold()

    if metric not in {"revenue", "profit"}:
        return None

    if metric == "revenue":
        gate = validate_revenue_readiness(dataframe)
    else:
        gate = validate_profit_readiness(dataframe)

    if not gate.analytics_ready:
        source_label = str(
            getattr(upload_result, "source_type", "dataset")
        ).upper()

        raise ValueError(
            f"{source_label} extraction is not ready for {metric} "
            "calculation. Current DataFrame quality gate: "
            f"{gate.to_dict()}"
        )

    return gate.to_dict()
'''

NEW_CALL_BLOCK = '''                    current_quality_gate = enforce_metric_quality_gate(
                        plan=plan,
                        dataframe=dataframe,
                        upload_result=upload_result,
                    )

                    if current_quality_gate is not None:
                        if not hasattr(upload_result, "evidence"):
                            pass
                        elif isinstance(upload_result.evidence, dict):
                            upload_result.evidence[
                                "pre_analysis_quality_gate"
                            ] = current_quality_gate

                    execution_result = execute_filtered_analysis(
'''


def timestamped_backup(path: Path) -> Path:
    """Create a timestamped backup outside the application package."""

    BACKUP_DIRECTORY.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = BACKUP_DIRECTORY / f"{path.stem}_before_unified_gate_{timestamp}.py"
    shutil.copy2(path, backup)
    return backup


def add_quality_import(source: str) -> str:
    """Add analytics-quality imports once."""

    if "from core.analytics_quality_gate import" in source:
        return source

    marker = "from core.data_profiler import profile_dataframe\n"
    if marker not in source:
        raise RuntimeError("Could not locate core.data_profiler import.")

    return source.replace(marker, marker + QUALITY_IMPORT, 1)


def replace_gate_function(source: str) -> str:
    """Replace the old upload-result gate with current-DataFrame validation."""

    pattern = re.compile(
        r"(?ms)^def enforce_metric_quality_gate\(.*?"
        r"(?=^def display_upload_evidence\()"
    )
    match = pattern.search(source)

    if match is None:
        raise RuntimeError(
            "Could not locate enforce_metric_quality_gate() followed by "
            "display_upload_evidence()."
        )

    return (
        source[: match.start()]
        + NEW_GATE_FUNCTION.rstrip()
        + "\n\n\n"
        + source[match.end() :]
    )


def replace_gate_call(source: str) -> str:
    """Pass the exact current DataFrame into the new gate."""

    if "current_quality_gate = enforce_metric_quality_gate(" in source:
        return source

    old = '''                    enforce_metric_quality_gate(plan, upload_result)
                    execution_result = execute_filtered_analysis(
'''

    if old not in source:
        raise RuntimeError(
            "Could not locate the old enforce_metric_quality_gate call."
        )

    return source.replace(old, NEW_CALL_BLOCK, 1)


def validate_source(source: str) -> None:
    """Perform structural and syntax validation before writing."""

    required = [
        "validate_revenue_readiness",
        "validate_profit_readiness",
        "dataframe: pd.DataFrame",
        "Current DataFrame quality gate",
        "current_quality_gate = enforce_metric_quality_gate(",
        "dataframe=dataframe",
        "pre_analysis_quality_gate",
        "execution_result = execute_filtered_analysis(",
    ]
    missing = [token for token in required if token not in source]

    if missing:
        raise RuntimeError("Incomplete Streamlit patch. Missing: " + ", ".join(missing))

    if source.count("def enforce_metric_quality_gate(") != 1:
        raise RuntimeError("Expected exactly one enforce_metric_quality_gate function.")

    if source.count("current_quality_gate = enforce_metric_quality_gate(") != 1:
        raise RuntimeError("Expected exactly one current quality-gate call.")

    ast.parse(source)


def main() -> None:
    """Install the current-DataFrame gate into the unified-loader app."""

    print("=" * 80)
    print("InsightFlow AI Unified-Loader Quality-Gate Fix")
    print("=" * 80)

    if not APP_PATH.exists():
        raise FileNotFoundError(f"Streamlit application not found: {APP_PATH}")

    original = APP_PATH.read_text(encoding="utf-8-sig")

    already_fixed = (
        "Current DataFrame quality gate" in original
        and "current_quality_gate = enforce_metric_quality_gate(" in original
        and "dataframe=dataframe" in original
    )

    if already_fixed:
        validate_source(original)
        py_compile.compile(str(APP_PATH), doraise=True)
        print("The unified-loader current-DataFrame gate already exists.")
        print("No changes were required.")
        return

    modified = add_quality_import(original)
    modified = replace_gate_function(modified)
    modified = replace_gate_call(modified)
    validate_source(modified)

    backup = timestamped_backup(APP_PATH)
    print(f"Backup created: {backup}")

    APP_PATH.write_bytes(modified.encode("utf-8"))
    py_compile.compile(str(APP_PATH), doraise=True)

    print(f"Updated: {APP_PATH}")
    print("streamlit_app.py compilation passed.")
    print()
    print("=" * 80)
    print("UNIFIED-LOADER QUALITY-GATE FIX PASSED")
    print("=" * 80)


if __name__ == "__main__":
    main()
